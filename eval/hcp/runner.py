import copy
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import yaml

from util.dynamics_inference import _build_dynamics, _build_tokenizer, _load_checkpoint_weights, _load_config

from .ar import first_prep_rows_by_run, generate_fixation_ar_windows, save_fixation_ar_stats
from .bank import (
    build_bank_prep,
    build_hcp_dataset,
    build_ibc_subject_context_map,
    build_ibc_task_bank,
    build_subject_context_map,
    build_task_bank,
    generate_future_windows,
    generate_rollout_bank,
)
from .common import base_subject_id, pick, sanitize_name
from .eval_setup import build_eval_banks, build_ibc_eval_datasets
from .glm import (
    HCP_CONTRASTS,
    HCP_TASKS,
    average_glm_phrase_variant_outputs,
    run_glm_for_ibc_task,
    run_glm_for_task,
    save_glm,
    save_glm_group_only,
    save_glm_guidance_compare,
)
from .ibc_contrasts import IBC_CONTRASTS
from .outputs import (
    add_hcp_task_summary_stats,
    append_hcp_report_rows,
    append_ibc_relevance_report_row,
    append_ibc_report_rows,
    glm_report_images,
    hcp_task_summary_display,
    level1_summary,
    write_eval_outputs,
    write_fc_only_outputs,
)
from .relevance_input import load_relevance_input_config
from .report import (
    compute_averaged_metrics,
    compute_level1_metrics,
    load_roi_labels,
    save_averaged,
    save_grid_timeseries_stats,
    save_level1,
    save_subject_id,
)
from .stats import add_stats_value
from .subject_id import run_subject_id_eval


@dataclass(frozen=True)
class EvalOptions:
    relevance_input_cfg: dict
    default_relevance_mode: str
    relevance_grid: list
    run_ibc_glm: bool
    phrase_variant_indices: list
    glm_guidance_grid: list
    glm_piece_drop_grid: list
    current_phrase_variant_idx: object
    subject_id_grid_enabled: bool
    guidance_scale: float
    guidance_scale_grid: list
    context_guidance_scale: float
    context_guidance_scale_grid: list
    num_steps: int
    stochastic_euler_eps: float
    stochastic_euler_noise_space: str
    euler_stop_sigma: float
    batch_size: int
    rollout_counts: list
    ar_rollout_cfg: object


@dataclass(frozen=True)
class HcpEvalRuntime:
    cfg: dict
    eval_cfg: dict
    out_dir: Path
    holdout_subject_ids_path: str
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    relevance_input_cfg: dict
    ts_cache: dict


def _scale_tag(prefix, value):
    return f"{prefix}_{str(value).replace('.', 'p')}"


def _glm_phrase_variant_indices(eval_cfg):
    phrase_cfg = eval_cfg.get("glm_phrase_variants", {"kind": "disabled"})
    kind = str(phrase_cfg["kind"])
    if kind == "disabled":
        return []
    if kind == "fixed_indices":
        indices = [int(x) for x in phrase_cfg["indices"]]
        assert len(indices) == len(set(indices))
        return indices
    raise ValueError(f"Unknown evaluation.hcp.glm_phrase_variants.kind={kind!r}")


def _glm_guidance_grid(eval_cfg):
    grid_cfg = eval_cfg.get("glm_guidance_grid", {"kind": "disabled"})
    kind = str(grid_cfg["kind"])
    if kind == "disabled":
        return []
    if kind == "cartesian":
        pairs = [
            (float(guidance), float(context_guidance))
            for guidance in grid_cfg["guidance_scales"]
            for context_guidance in grid_cfg["context_guidance_scales"]
        ]
        assert len(pairs) == len(set(pairs))
        return pairs
    raise ValueError(f"Unknown evaluation.hcp.glm_guidance_grid.kind={kind!r}")


def _glm_piece_drop_grid(eval_cfg):
    grid_cfg = eval_cfg.get("glm_piece_drop_grid", {"kind": "disabled"})
    kind = str(grid_cfg["kind"])
    if kind == "disabled":
        return []
    if kind == "fixed":
        drops = [_normalize_piece_drop(piece) for piece in grid_cfg["pieces"]]
        assert len(drops) == len(set(drops))
        return drops
    raise ValueError(f"Unknown evaluation.hcp.glm_piece_drop_grid.kind={kind!r}")


def _normalize_piece_drop(piece):
    if isinstance(piece, str):
        pieces = (piece,)
    else:
        pieces = tuple(piece)
    pieces = tuple("instruction" if str(curr) == "instruct" else str(curr) for curr in pieces)
    assert len(pieces) > 0
    assert len(pieces) == len(set(pieces))
    for curr in pieces:
        assert curr in {"instruction", "sensory", "response"}
    return pieces


def _piece_drop_label(pieces):
    return "_".join(pieces)


@contextmanager
def _force_condition_piece_drop(model, pieces):
    assert bool(getattr(model, "condition_tripartite", False))
    piece_indices = [{"instruction": 0, "sensory": 1, "response": 2}[piece] for piece in pieces]
    restore_piece_mask = model._sample_condition_piece_drop_mask

    def _fixed_piece_drop_mask(drop_mask, B, T, device):
        base = restore_piece_mask(drop_mask, B, T, device)
        forced = torch.zeros((B, T, 3), dtype=torch.bool, device=device)
        forced[:, :, piece_indices] = True
        return base | forced

    model._sample_condition_piece_drop_mask = _fixed_piece_drop_mask
    try:
        yield
    finally:
        model._sample_condition_piece_drop_mask = restore_piece_mask


def _ar_rollout_cfg(eval_cfg):
    cfg = eval_cfg.get("ar_rollout", {"kind": "disabled"})
    kind = str(cfg["kind"])
    if kind == "disabled":
        return None
    if kind == "fixation":
        steps = int(cfg["steps"])
        assert steps > 0
        return {"kind": kind, "steps": steps, "seed": int(cfg["seed"])}
    raise ValueError(f"Unknown evaluation.hcp.ar_rollout.kind={kind!r}")


def _guidance_pair_key(guidance_scale, context_guidance_scale):
    return f"g{float(guidance_scale):.1f}_ctx{float(context_guidance_scale):.1f}"


def _guidance_pair_dir(guidance_scale, context_guidance_scale):
    return (
        f"guidance_{str(float(guidance_scale)).replace('.', 'p')}"
        f"__context_{str(float(context_guidance_scale)).replace('.', 'p')}"
    )


def _run_subject_id_grid(glm_guidance_grid):
    if not glm_guidance_grid:
        return False
    context_scales = sorted({float(context_guidance_scale) for _, context_guidance_scale in glm_guidance_grid})
    return context_scales != [1.0]


def _save_glm_phrase_variant_summaries(
    phrase_outputs_by_variant,
    roi_labels,
    guidance_scale,
    context_guidance_scale,
    stats_bundle,
    stats_prefix_base,
):
    phrase_summary_records = {}
    for phrase_variant_idx in sorted(phrase_outputs_by_variant):
        phrase_key = f"phrase_variant_{phrase_variant_idx}"
        phrase_summary = save_glm_group_only(
            phrase_outputs_by_variant[phrase_variant_idx],
            guidance_scale=guidance_scale,
            context_guidance_scale=context_guidance_scale,
            phrase_variant_idx=phrase_variant_idx,
            bundle=stats_bundle,
            stats_prefix=f"{stats_prefix_base}__{phrase_key}",
        )
        phrase_summary_records[phrase_key] = json.loads(phrase_summary.to_json(orient="records"))
    if phrase_outputs_by_variant:
        phrase_summary = save_glm_group_only(
            average_glm_phrase_variant_outputs(phrase_outputs_by_variant, roi_labels),
            guidance_scale=guidance_scale,
            context_guidance_scale=context_guidance_scale,
            phrase_variant_idx=-1,
            bundle=stats_bundle,
            stats_prefix=f"{stats_prefix_base}__mean",
        )
        phrase_summary_records["mean"] = json.loads(phrase_summary.to_json(orient="records"))
    return phrase_summary_records


def _with_glm_phrase_variant(cfg, phrase_variant_idx):
    out = copy.deepcopy(cfg)
    data_cfg = out["data"]
    lang_cfgs = [data_cfg["val_condition_cont_language_events"]]
    for spec in data_cfg["val_datasets"]:
        if "condition_cont_language_events" in spec:
            lang_cfgs.append(spec["condition_cont_language_events"])
    for lang_cfg in lang_cfgs:
        lang_cfg["variant_sampling"] = "fixed"
        lang_cfg["fixed_variant_idx"] = int(phrase_variant_idx)
    return out


def _current_fixed_phrase_variant(cfg):
    lang_cfg = cfg["data"]["val_condition_cont_language_events"]
    if str(lang_cfg["variant_sampling"]) != "fixed":
        return None
    return int(lang_cfg["fixed_variant_idx"])


def _resolve_eval_options(cfg, eval_cfg):
    relevance_input_cfg = load_relevance_input_config(eval_cfg)
    default_relevance_mode = relevance_input_cfg["default"]
    relevance_grid = list(relevance_input_cfg["grid"])
    run_ibc_glm = bool(eval_cfg.get("run_ibc_glm", False))
    phrase_variant_indices = _glm_phrase_variant_indices(eval_cfg)
    glm_guidance_grid = _glm_guidance_grid(eval_cfg)
    glm_piece_drop_grid = _glm_piece_drop_grid(eval_cfg)
    if glm_piece_drop_grid:
        assert not bool(eval_cfg.get("force_drop_text_condition_token", False))
    current_phrase_variant_idx = _current_fixed_phrase_variant(cfg)
    if glm_guidance_grid:
        assert current_phrase_variant_idx is not None
        assert not phrase_variant_indices
    subject_id_grid_enabled = _run_subject_id_grid(glm_guidance_grid)

    guidance_scale = float(eval_cfg.get("guidance_scale", 1.0))
    guidance_scale_grid = [float(x) for x in eval_cfg.get("guidance_scale_grid", [])]
    guidance_scale_grid = [x for x in guidance_scale_grid if x != guidance_scale]
    context_guidance_scale = float(eval_cfg.get("context_guidance_scale", 1.0))
    context_guidance_scale_grid = [float(x) for x in eval_cfg.get("context_guidance_scale_grid", [])]
    context_guidance_scale_grid = [x for x in context_guidance_scale_grid if x != context_guidance_scale]

    return EvalOptions(
        relevance_input_cfg=relevance_input_cfg,
        default_relevance_mode=default_relevance_mode,
        relevance_grid=relevance_grid,
        run_ibc_glm=run_ibc_glm,
        phrase_variant_indices=phrase_variant_indices,
        glm_guidance_grid=glm_guidance_grid,
        glm_piece_drop_grid=glm_piece_drop_grid,
        current_phrase_variant_idx=current_phrase_variant_idx,
        subject_id_grid_enabled=subject_id_grid_enabled,
        guidance_scale=guidance_scale,
        guidance_scale_grid=guidance_scale_grid,
        context_guidance_scale=context_guidance_scale,
        context_guidance_scale_grid=context_guidance_scale_grid,
        num_steps=int(eval_cfg.get("num_steps", 64)),
        stochastic_euler_eps=float(eval_cfg.get("stochastic_euler_eps", 0.0)),
        stochastic_euler_noise_space=str(eval_cfg.get("stochastic_euler_noise_space", "latent")).lower(),
        euler_stop_sigma=float(eval_cfg.get("euler_stop_sigma", 1.0)),
        batch_size=int(eval_cfg.get("batch_size", 32)),
        rollout_counts=list(eval_cfg.get("rollout_counts", [1, 2, 5, 10, 25, 50])),
        ar_rollout_cfg=_ar_rollout_cfg(eval_cfg),
    )


def _init_stats_bundle(tasks, roi_labels, options):
    stats_bundle = {}
    add_stats_value(stats_bundle, "meta", "schema_version", value=1)
    add_stats_value(stats_bundle, "meta", "tasks", value=np.asarray(tasks, dtype=str))
    add_stats_value(stats_bundle, "meta", "roi_labels", value=np.asarray(roi_labels, dtype=str))
    add_stats_value(stats_bundle, "meta", "rollout_counts", value=np.asarray(options.rollout_counts, dtype=np.int64))
    add_stats_value(stats_bundle, "meta", "default_relevance_mode", value=str(options.default_relevance_mode))
    add_stats_value(stats_bundle, "meta", "relevance_grid", value=np.asarray(options.relevance_grid, dtype=str))
    add_stats_value(
        stats_bundle,
        "meta",
        "glm_phrase_variant_indices",
        value=np.asarray(options.phrase_variant_indices, dtype=np.int64),
    )
    add_stats_value(stats_bundle, "meta", "glm_guidance_grid", value=np.asarray(options.glm_guidance_grid, dtype=np.float32))
    add_stats_value(
        stats_bundle,
        "meta",
        "glm_piece_drop_grid",
        value=np.asarray([_piece_drop_label(pieces) for pieces in options.glm_piece_drop_grid], dtype=str),
    )
    add_stats_value(
        stats_bundle,
        "meta",
        "guidance_scales",
        value=np.asarray([options.guidance_scale] + options.guidance_scale_grid, dtype=np.float32),
    )
    add_stats_value(
        stats_bundle,
        "meta",
        "context_guidance_scales",
        value=np.asarray([options.context_guidance_scale] + options.context_guidance_scale_grid, dtype=np.float32),
    )
    add_stats_value(stats_bundle, "meta", "timeseries_context", value="continuation")
    add_stats_value(stats_bundle, "meta", "glm_context", value="pretask_fixation")
    add_stats_value(
        stats_bundle,
        "meta",
        "ar_rollout_kind",
        value="disabled" if options.ar_rollout_cfg is None else options.ar_rollout_cfg["kind"],
    )
    if options.ar_rollout_cfg is not None:
        add_stats_value(stats_bundle, "meta", "ar_rollout_steps", value=options.ar_rollout_cfg["steps"])
    return stats_bundle


def _run_fc_eval(runtime, fc_cfg):
    from .fc import FcEvalRuntime, run_fc_eval

    run_fc_eval(FcEvalRuntime(
        cfg=runtime.cfg,
        eval_cfg={**runtime.eval_cfg, "fc_eval": fc_cfg},
        out_dir=runtime.out_dir,
        holdout_subject_ids_path=runtime.holdout_subject_ids_path,
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        device=runtime.device,
        relevance_input_cfg=runtime.relevance_input_cfg,
        ts_cache=runtime.ts_cache,
    ))


def run_hcp_eval(config_path, checkpoint_path, output_dir):
    cfg = _load_config(config_path)
    eval_cfg = cfg.get("evaluation", {}).get("hcp")
    assert isinstance(eval_cfg, dict)
    assert bool(eval_cfg.get("enabled", False))
    assert str(eval_cfg.get("run_at", "final")) == "final"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    tasks = [str(task).upper() for task in eval_cfg["tasks"]]
    for task in tasks:
        assert task in HCP_TASKS

    holdout_subject_ids_path = str(eval_cfg["holdout_subject_ids_path"])
    atlas_names = list(cfg["data"]["atlas_names"])
    device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = _build_tokenizer(cfg, device)
    model = _build_dynamics(cfg, tokenizer, device)
    _load_checkpoint_weights(model, checkpoint_path, device)
    model.eval()
    if bool(eval_cfg.get("force_drop_text_condition_token", False)):
        model.p_drop_text_condition_token = 1.0
    options = _resolve_eval_options(cfg, eval_cfg)
    relevance_input_cfg = options.relevance_input_cfg
    default_relevance_mode = options.default_relevance_mode
    relevance_grid = options.relevance_grid
    run_ibc_glm = options.run_ibc_glm
    phrase_variant_indices = options.phrase_variant_indices
    glm_guidance_grid = options.glm_guidance_grid
    glm_piece_drop_grid = options.glm_piece_drop_grid
    current_phrase_variant_idx = options.current_phrase_variant_idx
    subject_id_grid_enabled = options.subject_id_grid_enabled

    context_frames = int(cfg["dynamics"].get("context_frames", 8))
    generation_frames = int(cfg["dynamics"].get("generation_frames", 16))
    ts_cache = {}
    runtime = HcpEvalRuntime(
        cfg=cfg,
        eval_cfg=eval_cfg,
        out_dir=out_dir,
        holdout_subject_ids_path=holdout_subject_ids_path,
        model=model,
        tokenizer=tokenizer,
        device=device,
        relevance_input_cfg=relevance_input_cfg,
        ts_cache=ts_cache,
    )
    fc_cfg = eval_cfg.get("fc_eval", {"enabled": False})
    if bool(fc_cfg.get("enabled", False)) and bool(fc_cfg.get("only", False)):
        _run_fc_eval(runtime, fc_cfg)
        write_fc_only_outputs(out_dir, fc_cfg, tasks)
        return

    eval_banks = build_eval_banks(
        cfg,
        holdout_subject_ids_path,
        relevance_input_cfg,
        ts_cache,
        tasks,
        atlas_names,
        context_frames,
        run_ibc_glm,
    )
    banks = eval_banks.banks
    ibc_datasets = eval_banks.ibc_datasets
    roi_labels = load_roi_labels(atlas_names, eval_banks.roi_counts)

    tr_seconds = float(pick(cfg["data"], "val", "tr_seconds", 1.0))
    guidance_scale = options.guidance_scale
    guidance_scale_grid = options.guidance_scale_grid
    context_guidance_scale = options.context_guidance_scale
    context_guidance_scale_grid = options.context_guidance_scale_grid
    num_steps = options.num_steps
    stochastic_euler_eps = options.stochastic_euler_eps
    stochastic_euler_noise_space = options.stochastic_euler_noise_space
    euler_stop_sigma = options.euler_stop_sigma
    batch_size = options.batch_size
    rollout_counts = options.rollout_counts
    ar_rollout_cfg = options.ar_rollout_cfg
    summary_rows = []
    summary_json = {"tasks": {}, "ibc": {}}
    stats_bundle = _init_stats_bundle(tasks, roi_labels, options)

    if bool(fc_cfg.get("enabled", False)):
        _run_fc_eval(runtime, fc_cfg)

    for task in tasks:
        bank = banks[task]
        task_key = sanitize_name(task)
        task_prefix = f"task__{task_key}"
        primary_contrast = HCP_CONTRASTS[task][0]["name"]
        task_dir = out_dir / task.lower()
        prep_cont = build_bank_prep(
            bank,
            "continuation",
            model,
            tokenizer,
            device,
            batch_size,
            relevance_mode=default_relevance_mode,
            relevance_input_cfg=relevance_input_cfg,
        )
        prep_glm = build_bank_prep(
            bank,
            "glm",
            model,
            tokenizer,
            device,
            batch_size,
            relevance_mode=default_relevance_mode,
            relevance_input_cfg=relevance_input_cfg,
        )
        ar_summary_records = []
        if ar_rollout_cfg is not None:
            ar_prep, ar_row_indices = first_prep_rows_by_run(prep_glm, bank["run_ids"])
            curr_ar_cfg = {
                **ar_rollout_cfg,
                "num_steps": num_steps,
                "batch_size": batch_size,
                "guidance_scale": guidance_scale,
                "context_guidance_scale": context_guidance_scale,
                "stochastic_euler_eps": stochastic_euler_eps,
                "stochastic_euler_noise_space": stochastic_euler_noise_space,
                "euler_stop_sigma": euler_stop_sigma,
            }
            print(
                f"[hcp-eval] {task}: fixation AR rollout "
                f"{curr_ar_cfg['steps']} steps for {ar_prep['z_context'].shape[0]} runs",
                flush=True,
            )
            ar_windows = generate_fixation_ar_windows(
                model,
                tokenizer,
                ar_prep,
                device,
                curr_ar_cfg,
                progress_label=f"{task} fixation AR",
            )
            ar_summary = pd.DataFrame(save_fixation_ar_stats(
                stats_bundle,
                f"{task_prefix}__ar_fixation",
                ar_windows,
                tr_seconds,
                ar_row_indices,
            ))
            ar_summary_records = json.loads(ar_summary.to_json(orient="records"))
            summary_rows.append({
                "title": f"{task} Fixation AR",
                "table": ar_summary,
                "images": [],
            })
        max_rollouts = max(int(x) for x in rollout_counts)

        print(
            f"[hcp-eval] {task}: generating continuation rollout bank with {max_rollouts} samples for {len(bank['run_ids'])} windows",
            flush=True,
        )
        rollout_bank = generate_rollout_bank(
            model,
            tokenizer,
            prep_cont,
            device,
            count=max_rollouts,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            context_guidance_scale=context_guidance_scale,
            batch_size=batch_size,
            progress_label=f"{task} continuation rollout bank",
            condition_guidance_scope="all_timepoints",
            stochastic_euler_eps=stochastic_euler_eps,
            stochastic_euler_noise_space=stochastic_euler_noise_space,
            euler_stop_sigma=euler_stop_sigma,
        )
        rollout_bank_by_context_guidance = {}
        for curr_context_guidance_scale in context_guidance_scale_grid:
            print(f"[hcp-eval] {task}: continuation context guidance {curr_context_guidance_scale}", flush=True)
            rollout_bank_by_context_guidance[curr_context_guidance_scale] = generate_rollout_bank(
                model,
                tokenizer,
                prep_cont,
                device,
                count=1,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                context_guidance_scale=curr_context_guidance_scale,
                batch_size=batch_size,
                progress_label=f"{task} continuation context guidance {curr_context_guidance_scale}",
                condition_guidance_scope="all_timepoints",
                stochastic_euler_eps=stochastic_euler_eps,
                stochastic_euler_noise_space=stochastic_euler_noise_space,
                euler_stop_sigma=euler_stop_sigma,
            )[0]

        real_full = bank["signal_cont"][:, :, :, 0]
        real_future = bank["future_signal"][:, :, :, 0]
        condition_names = bank["block_names"].tolist()

        level1 = compute_level1_metrics(real_future, rollout_bank[0], tr_seconds, condition_names)
        save_level1(task_dir / "level1", task, level1, roi_labels, bundle=stats_bundle, stats_prefix=f"{task_prefix}__level1")

        averaged = compute_averaged_metrics(real_future, rollout_bank, rollout_counts, condition_names)
        save_averaged(task_dir / "averaged", averaged, roi_labels, bundle=stats_bundle, stats_prefix=f"{task_prefix}__averaged")

        glm_outputs_by_scale = {}
        default_glm_outputs = None
        default_glm_future = None
        for curr_guidance_scale in [guidance_scale] + guidance_scale_grid:
            print(f"[hcp-eval] {task}: glm guidance {curr_guidance_scale}", flush=True)
            synth_future_glm = generate_future_windows(
                model,
                tokenizer,
                prep_glm,
                device,
                num_steps=num_steps,
                guidance_scale=curr_guidance_scale,
                context_guidance_scale=context_guidance_scale,
                seed=0,
                batch_size=batch_size,
                progress_label=f"{task} glm guidance {curr_guidance_scale}",
                condition_guidance_scope="all_timepoints",
                stochastic_euler_eps=stochastic_euler_eps,
                stochastic_euler_noise_space=stochastic_euler_noise_space,
                euler_stop_sigma=euler_stop_sigma,
            )
            outputs = run_glm_for_task(
                task,
                bank,
                roi_labels,
                tr_seconds=tr_seconds,
                generation_frames=generation_frames,
                synth_future=synth_future_glm,
            )
            glm_outputs_by_scale[curr_guidance_scale] = outputs
            if curr_guidance_scale == guidance_scale:
                default_glm_outputs = outputs
                default_glm_future = synth_future_glm
                save_glm(
                    task_dir / "glm",
                    outputs,
                    guidance_scale=curr_guidance_scale,
                    context_guidance_scale=context_guidance_scale,
                    bundle=stats_bundle,
                    stats_prefix=f"{task_prefix}__glm_variants__default",
                    image_contrasts={primary_contrast},
                )
            else:
                save_glm(
                    task_dir / "glm_guidance" / f"scale_{str(curr_guidance_scale).replace('.', 'p')}",
                    outputs,
                    guidance_scale=curr_guidance_scale,
                    context_guidance_scale=context_guidance_scale,
                    bundle=stats_bundle,
                    stats_prefix=f"{task_prefix}__glm_variants__{_scale_tag('guidance', curr_guidance_scale)}",
                    image_contrasts={primary_contrast},
                )
        save_glm_guidance_compare(
            glm_outputs_by_scale,
            bundle=stats_bundle,
            stats_prefix=f"{task_prefix}__glm_guidance_compare",
        )
        assert default_glm_outputs is not None
        assert default_glm_future is not None
        glm_piece_drop_records = []
        glm_piece_drop_images = []
        for pieces in glm_piece_drop_grid:
            piece_label = _piece_drop_label(pieces)
            piece_key = sanitize_name(piece_label)
            print(f"[hcp-eval] {task}: glm piece drop {piece_label}", flush=True)
            with _force_condition_piece_drop(model, pieces):
                piece_future = generate_future_windows(
                    model,
                    tokenizer,
                    prep_glm,
                    device,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    context_guidance_scale=context_guidance_scale,
                    seed=0,
                    batch_size=batch_size,
                    progress_label=f"{task} glm piece drop {piece_label}",
                    condition_guidance_scope="all_timepoints",
                    stochastic_euler_eps=stochastic_euler_eps,
                    stochastic_euler_noise_space=stochastic_euler_noise_space,
                    euler_stop_sigma=euler_stop_sigma,
                )
            piece_outputs = run_glm_for_task(
                task,
                bank,
                roi_labels,
                tr_seconds=tr_seconds,
                generation_frames=generation_frames,
                synth_future=piece_future,
            )
            piece_summary = save_glm(
                task_dir / "glm_piece_drop" / piece_key,
                piece_outputs,
                guidance_scale=guidance_scale,
                context_guidance_scale=context_guidance_scale,
                bundle=stats_bundle,
                stats_prefix=f"{task_prefix}__glm_piece_drop__{piece_key}",
                image_contrasts={primary_contrast},
            )
            glm_piece_drop_records.extend(
                {"piece": piece_label, **row}
                for row in json.loads(piece_summary.to_json(orient="records"))
            )
            glm_piece_drop_images.extend(glm_report_images(
                task_dir / "glm_piece_drop" / piece_key,
                primary_contrast,
                f"piece drop={piece_label}",
            ))
        glm_outputs_by_relevance = {default_relevance_mode: default_glm_outputs}
        glm_guidance_grid_records = []
        glm_guidance_grid_images = []
        for curr_guidance_scale, curr_context_guidance_scale in glm_guidance_grid:
            pair_key = _guidance_pair_dir(curr_guidance_scale, curr_context_guidance_scale)
            grid_stats_prefix = f"{task_prefix}__glm_guidance_grid__{pair_key}"
            if curr_guidance_scale == guidance_scale and curr_context_guidance_scale == context_guidance_scale:
                outputs = default_glm_outputs
                grid_future = default_glm_future
            else:
                print(
                    f"[hcp-eval] {task}: glm guidance grid "
                    f"guidance={curr_guidance_scale} context={curr_context_guidance_scale}",
                    flush=True,
                )
                grid_future = generate_future_windows(
                    model,
                    tokenizer,
                    prep_glm,
                    device,
                    num_steps=num_steps,
                    guidance_scale=curr_guidance_scale,
                    context_guidance_scale=curr_context_guidance_scale,
                    seed=0,
                    batch_size=batch_size,
                    progress_label=(
                        f"{task} glm guidance grid "
                        f"guidance={curr_guidance_scale} context={curr_context_guidance_scale}"
                    ),
                    condition_guidance_scope="all_timepoints",
                    stochastic_euler_eps=stochastic_euler_eps,
                    stochastic_euler_noise_space=stochastic_euler_noise_space,
                    euler_stop_sigma=euler_stop_sigma,
                )
                outputs = run_glm_for_task(
                    task,
                    bank,
                    roi_labels,
                    tr_seconds=tr_seconds,
                    generation_frames=generation_frames,
                    synth_future=grid_future,
                )
            timeseries_summary = save_grid_timeseries_stats(stats_bundle, grid_stats_prefix, real_future, grid_future)
            grid_summary = save_glm(
                task_dir / "glm_guidance_grid" / pair_key,
                outputs,
                guidance_scale=curr_guidance_scale,
                context_guidance_scale=curr_context_guidance_scale,
                bundle=stats_bundle,
                stats_prefix=grid_stats_prefix,
                image_contrasts={primary_contrast},
            )
            grid_summary = grid_summary.assign(**timeseries_summary)
            glm_guidance_grid_records.extend(json.loads(grid_summary.to_json(orient="records")))
            glm_guidance_grid_images.extend(glm_report_images(
                task_dir / "glm_guidance_grid" / pair_key,
                primary_contrast,
                f"guidance={curr_guidance_scale}, context guidance={curr_context_guidance_scale}",
            ))
        phrase_outputs_by_variant = {}
        for phrase_variant_idx in phrase_variant_indices:
            phrase_key = f"phrase_variant_{phrase_variant_idx}"
            if phrase_variant_idx == current_phrase_variant_idx:
                outputs = default_glm_outputs
            else:
                phrase_cfg = _with_glm_phrase_variant(cfg, phrase_variant_idx)
                phrase_datasets = {
                    hcp_task: build_hcp_dataset(phrase_cfg, hcp_task, holdout_subject_ids_path, relevance_input_cfg)
                    for hcp_task in HCP_TASKS
                }
                phrase_context_map = build_subject_context_map(phrase_datasets, context_frames)
                phrase_bank = build_task_bank(task, phrase_datasets, phrase_context_map, phrase_cfg, ts_cache, relevance_input_cfg)
                phrase_prep_glm = build_bank_prep(
                    phrase_bank,
                    "glm",
                    model,
                    tokenizer,
                    device,
                    batch_size,
                    relevance_mode=default_relevance_mode,
                    relevance_input_cfg=relevance_input_cfg,
                )
                print(f"[hcp-eval] {task}: glm {phrase_key}", flush=True)
                phrase_future = generate_future_windows(
                    model,
                    tokenizer,
                    phrase_prep_glm,
                    device,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    context_guidance_scale=context_guidance_scale,
                    seed=0,
                    batch_size=batch_size,
                    progress_label=f"{task} glm {phrase_key}",
                    condition_guidance_scope="all_timepoints",
                    stochastic_euler_eps=stochastic_euler_eps,
                    stochastic_euler_noise_space=stochastic_euler_noise_space,
                    euler_stop_sigma=euler_stop_sigma,
                )
                outputs = run_glm_for_task(
                    task,
                    phrase_bank,
                    roi_labels,
                    tr_seconds=tr_seconds,
                    generation_frames=generation_frames,
                    synth_future=phrase_future,
                )
            phrase_outputs_by_variant[phrase_variant_idx] = outputs
        phrase_summary_records = _save_glm_phrase_variant_summaries(
            phrase_outputs_by_variant,
            roi_labels,
            guidance_scale,
            context_guidance_scale,
            stats_bundle,
            f"{task_prefix}__glm_phrase_variants",
        )
        glm_context_outputs_by_scale = {}
        for curr_context_guidance_scale in [context_guidance_scale] + context_guidance_scale_grid:
            if curr_context_guidance_scale == context_guidance_scale:
                outputs = default_glm_outputs
            else:
                print(f"[hcp-eval] {task}: glm context guidance {curr_context_guidance_scale}", flush=True)
                synth_future_glm = generate_future_windows(
                    model,
                    tokenizer,
                    prep_glm,
                    device,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    context_guidance_scale=curr_context_guidance_scale,
                    seed=0,
                    batch_size=batch_size,
                    progress_label=f"{task} glm context guidance {curr_context_guidance_scale}",
                    condition_guidance_scope="all_timepoints",
                    stochastic_euler_eps=stochastic_euler_eps,
                    stochastic_euler_noise_space=stochastic_euler_noise_space,
                    euler_stop_sigma=euler_stop_sigma,
                )
                outputs = run_glm_for_task(
                    task,
                    bank,
                    roi_labels,
                    tr_seconds=tr_seconds,
                    generation_frames=generation_frames,
                    synth_future=synth_future_glm,
                )
            glm_context_outputs_by_scale[curr_context_guidance_scale] = outputs
            if curr_context_guidance_scale != context_guidance_scale:
                save_glm(
                    task_dir / "glm_context_guidance" / f"scale_{str(curr_context_guidance_scale).replace('.', 'p')}",
                    outputs,
                    guidance_scale=guidance_scale,
                    context_guidance_scale=curr_context_guidance_scale,
                    bundle=stats_bundle,
                    stats_prefix=f"{task_prefix}__glm_variants__{_scale_tag('context_guidance', curr_context_guidance_scale)}",
                    image_contrasts={primary_contrast},
                )
        save_glm_guidance_compare(
            glm_context_outputs_by_scale,
            bundle=stats_bundle,
            stats_prefix=f"{task_prefix}__glm_context_guidance_compare",
        )

        task_summary = {
            "relevance_input_default": default_relevance_mode,
            "relevance_input_grid": relevance_grid,
            "mean_ks": level1["overall"]["mean_ks"],
            "mmd": level1["overall"]["mmd"],
            "paired_fc_corr_mean": level1["overall"]["paired_fc_corr_mean"],
            "paired_fc_corr_std": level1["overall"]["paired_fc_corr_std"],
            "real_mean_fc_edge_weight": level1["overall"]["real_mean_fc_edge_weight"],
            "real_std_fc_edge_weight": level1["overall"]["real_std_fc_edge_weight"],
            "synth_mean_fc_edge_weight": level1["overall"]["synth_mean_fc_edge_weight"],
            "synth_std_fc_edge_weight": level1["overall"]["synth_std_fc_edge_weight"],
            "glm_guidance_scales": [guidance_scale] + guidance_scale_grid,
            "glm_context_guidance_scales": [context_guidance_scale] + context_guidance_scale_grid,
            "glm_guidance_grid": glm_guidance_grid_records,
            "glm_piece_drop": glm_piece_drop_records,
            "glm_phrase_variants": phrase_summary_records,
            "ar_fixation": ar_summary_records,
            "timeseries_context": "continuation",
            "glm_context": "pretask_fixation",
        }
        subject_cfg = eval_cfg.get("subject_id", {})
        if bool(subject_cfg.get("enabled", True)):
            print(f"[hcp-eval] {task}: subject classifier start", flush=True)
            subject_start = time.time()
            train_real = []
            train_labels = []
            for other_task in HCP_TASKS:
                if other_task == task:
                    continue
                train_real.append(banks[other_task]["signal_cont"][:, :, :, 0])
                train_labels.extend([base_subject_id(run_id) for run_id in banks[other_task]["run_ids"].tolist()])
            train_real = np.concatenate(train_real, axis=0)
            train_subjects = sorted({base_subject_id(run_id) for run_id in bank["run_ids"].tolist()})
            subject_to_idx = {subject_id: idx for idx, subject_id in enumerate(train_subjects)}
            train_label_idx = np.asarray([subject_to_idx[sid] for sid in train_labels], dtype=np.int64)
            eval_label_idx = np.asarray([subject_to_idx[base_subject_id(run_id)] for run_id in bank["run_ids"].tolist()], dtype=np.int64)
            if subject_id_grid_enabled:
                eval_synth_sequences_by_name = {}
                for curr_guidance_scale, curr_context_guidance_scale in glm_guidance_grid:
                    synth_name = _guidance_pair_key(curr_guidance_scale, curr_context_guidance_scale)
                    if curr_guidance_scale == guidance_scale and curr_context_guidance_scale == context_guidance_scale:
                        synth_future = rollout_bank[0]
                    else:
                        print(
                            f"[hcp-eval] {task}: subject ID grid "
                            f"guidance={curr_guidance_scale} context={curr_context_guidance_scale}",
                            flush=True,
                        )
                        synth_future = generate_future_windows(
                            model,
                            tokenizer,
                            prep_cont,
                            device,
                            num_steps=num_steps,
                            guidance_scale=curr_guidance_scale,
                            context_guidance_scale=curr_context_guidance_scale,
                            seed=0,
                            batch_size=batch_size,
                            progress_label=(
                                f"{task} subject ID grid "
                                f"guidance={curr_guidance_scale} context={curr_context_guidance_scale}"
                            ),
                            condition_guidance_scope="all_timepoints",
                            stochastic_euler_eps=stochastic_euler_eps,
                            stochastic_euler_noise_space=stochastic_euler_noise_space,
                            euler_stop_sigma=euler_stop_sigma,
                        )
                    eval_synth_sequences_by_name[synth_name] = np.concatenate(
                        [real_full[:, :context_frames], synth_future],
                        axis=1,
                    )
            else:
                eval_synth_sequences_by_name = {
                    f"synth_ctx_{context_guidance_scale:.1f}": np.concatenate([real_full[:, :context_frames], rollout_bank[0]], axis=1),
                }
                for curr_context_guidance_scale, synth_future in rollout_bank_by_context_guidance.items():
                    eval_synth_sequences_by_name[f"synth_ctx_{curr_context_guidance_scale:.1f}"] = np.concatenate(
                        [real_full[:, :context_frames], synth_future],
                        axis=1,
                    )
            subject_results = run_subject_id_eval(
                train_real,
                train_label_idx,
                real_full,
                eval_synth_sequences_by_name,
                eval_label_idx,
                num_rois=real_full.shape[-1],
                num_subjects=len(train_subjects),
                train_steps=int(subject_cfg.get("train_steps", 1000)),
                batch_size=int(subject_cfg.get("batch_size", 64)),
                lr=float(subject_cfg.get("lr", 1e-3)),
                seeds=list(subject_cfg.get("seeds", [0, 1, 2])),
                device=device,
            )
            print(f"[hcp-eval] {task}: subject classifier done in {time.time() - subject_start:.1f}s", flush=True)
            save_subject_id(
                task_dir / "subject_id",
                subject_results,
                context_frames=context_frames,
                bundle=stats_bundle,
                stats_prefix=f"{task_prefix}__subject_id",
            )
            task_summary["subject_id_real_mean"] = float(subject_results["real_mean"].mean())
            for name, synth in subject_results["synth"].items():
                task_summary[f"{name}_mean"] = float(synth["mean"].mean())

        relevance_compare = {
            default_relevance_mode: level1_summary(level1),
        }
        relevance_images = glm_report_images(task_dir / "glm", primary_contrast, f"relevance={default_relevance_mode}")
        for relevance_mode in relevance_grid:
            prep_cont_variant = build_bank_prep(
                bank,
                "continuation",
                model,
                tokenizer,
                device,
                batch_size,
                relevance_mode=relevance_mode,
                relevance_input_cfg=relevance_input_cfg,
            )
            prep_glm_variant = build_bank_prep(
                bank,
                "glm",
                model,
                tokenizer,
                device,
                batch_size,
                relevance_mode=relevance_mode,
                relevance_input_cfg=relevance_input_cfg,
            )

            print(f"[hcp-eval] {task}: relevance input {relevance_mode}", flush=True)
            variant_future = generate_future_windows(
                model,
                tokenizer,
                prep_cont_variant,
                device,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                context_guidance_scale=context_guidance_scale,
                seed=0,
                batch_size=batch_size,
                progress_label=f"{task} relevance input {relevance_mode}",
                condition_guidance_scope="all_timepoints",
                stochastic_euler_eps=stochastic_euler_eps,
                stochastic_euler_noise_space=stochastic_euler_noise_space,
                euler_stop_sigma=euler_stop_sigma,
            )
            variant_level1 = compute_level1_metrics(real_future, variant_future, tr_seconds, condition_names)
            variant_dir = task_dir / "relevance_input" / relevance_mode
            save_level1(
                variant_dir / "level1",
                task,
                variant_level1,
                roi_labels,
                bundle=stats_bundle,
                stats_prefix=f"{task_prefix}__relevance_variants__{sanitize_name(relevance_mode)}__level1",
            )

            variant_glm_future = generate_future_windows(
                model,
                tokenizer,
                prep_glm_variant,
                device,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                context_guidance_scale=context_guidance_scale,
                seed=0,
                batch_size=batch_size,
                progress_label=f"{task} glm relevance input {relevance_mode}",
                condition_guidance_scope="all_timepoints",
                stochastic_euler_eps=stochastic_euler_eps,
                stochastic_euler_noise_space=stochastic_euler_noise_space,
                euler_stop_sigma=euler_stop_sigma,
            )
            variant_glm = run_glm_for_task(
                task,
                bank,
                roi_labels,
                tr_seconds=tr_seconds,
                generation_frames=generation_frames,
                synth_future=variant_glm_future,
            )
            glm_outputs_by_relevance[relevance_mode] = variant_glm
            save_glm(
                variant_dir / "glm",
                variant_glm,
                guidance_scale=guidance_scale,
                context_guidance_scale=context_guidance_scale,
                bundle=stats_bundle,
                stats_prefix=f"{task_prefix}__glm_variants__relevance_{sanitize_name(relevance_mode)}",
                image_contrasts={primary_contrast},
            )
            relevance_compare[relevance_mode] = level1_summary(variant_level1)
            relevance_images.extend(glm_report_images(variant_dir / "glm", primary_contrast, f"relevance={relevance_mode}"))

        phrase_relevance_records = {}
        if phrase_variant_indices:
            for relevance_mode in relevance_grid:
                relevance_phrase_outputs_by_variant = {}
                for phrase_variant_idx in phrase_variant_indices:
                    phrase_key = f"phrase_variant_{phrase_variant_idx}"
                    if phrase_variant_idx == current_phrase_variant_idx:
                        outputs = glm_outputs_by_relevance[relevance_mode]
                    else:
                        phrase_cfg = _with_glm_phrase_variant(cfg, phrase_variant_idx)
                        phrase_datasets = {
                            hcp_task: build_hcp_dataset(phrase_cfg, hcp_task, holdout_subject_ids_path, relevance_input_cfg)
                            for hcp_task in HCP_TASKS
                        }
                        phrase_context_map = build_subject_context_map(phrase_datasets, context_frames)
                        phrase_bank = build_task_bank(
                            task,
                            phrase_datasets,
                            phrase_context_map,
                            phrase_cfg,
                            ts_cache,
                            relevance_input_cfg,
                        )
                        phrase_prep_glm = build_bank_prep(
                            phrase_bank,
                            "glm",
                            model,
                            tokenizer,
                            device,
                            batch_size,
                            relevance_mode=relevance_mode,
                            relevance_input_cfg=relevance_input_cfg,
                        )
                        print(f"[hcp-eval] {task}: glm relevance input {relevance_mode} {phrase_key}", flush=True)
                        phrase_future = generate_future_windows(
                            model,
                            tokenizer,
                            phrase_prep_glm,
                            device,
                            num_steps=num_steps,
                            guidance_scale=guidance_scale,
                            context_guidance_scale=context_guidance_scale,
                            seed=0,
                            batch_size=batch_size,
                            progress_label=f"{task} glm relevance input {relevance_mode} {phrase_key}",
                            condition_guidance_scope="all_timepoints",
                            stochastic_euler_eps=stochastic_euler_eps,
                            stochastic_euler_noise_space=stochastic_euler_noise_space,
                            euler_stop_sigma=euler_stop_sigma,
                        )
                        outputs = run_glm_for_task(
                            task,
                            phrase_bank,
                            roi_labels,
                            tr_seconds=tr_seconds,
                            generation_frames=generation_frames,
                            synth_future=phrase_future,
                        )
                    relevance_phrase_outputs_by_variant[phrase_variant_idx] = outputs
                relevance_key = sanitize_name(relevance_mode)
                phrase_relevance_records[relevance_mode] = _save_glm_phrase_variant_summaries(
                    relevance_phrase_outputs_by_variant,
                    roi_labels,
                    guidance_scale,
                    context_guidance_scale,
                    stats_bundle,
                    f"{task_prefix}__glm_variants__relevance_{relevance_key}__glm_phrase_variants",
                )

        task_summary["relevance_input_compare"] = relevance_compare
        task_summary["glm_phrase_variants_relevance"] = phrase_relevance_records
        task_summary_display = hcp_task_summary_display(task_summary)
        add_hcp_task_summary_stats(
            stats_bundle,
            task_prefix,
            task_summary_display,
            default_relevance_mode,
            relevance_grid,
            relevance_compare,
        )

        summary_json["tasks"][task] = task_summary

        append_hcp_report_rows(
            summary_rows,
            task,
            task_dir,
            primary_contrast,
            task_summary_display,
            guidance_scale,
            guidance_scale_grid,
            context_guidance_scale_grid,
            glm_guidance_grid_records,
            glm_guidance_grid_images,
            relevance_grid,
            default_relevance_mode,
            relevance_compare,
            relevance_images,
            glm_piece_drop_records,
            glm_piece_drop_images,
        )

    if run_ibc_glm:
        if ibc_datasets is None:
            ibc_datasets = build_ibc_eval_datasets(cfg, relevance_input_cfg)
        assert len(ibc_datasets) > 0
        ibc_tasks = sorted(task for task in ibc_datasets if task in IBC_CONTRASTS)
        skipped_ibc_tasks = sorted(task for task in ibc_datasets if task not in IBC_CONTRASTS)
        if skipped_ibc_tasks:
            print(
                f"[hcp-eval] skipping IBC GLM for tasks with no registered contrasts: {', '.join(skipped_ibc_tasks)}",
                flush=True,
            )
        assert len(ibc_tasks) > 0, "run_ibc_glm is enabled, but no IBC tasks have registered contrasts"
        add_stats_value(stats_bundle, "meta", "ibc_tasks", value=np.asarray(ibc_tasks, dtype=str))
        add_stats_value(stats_bundle, "meta", "ibc_tasks_skipped", value=np.asarray(skipped_ibc_tasks, dtype=str))
        ibc_context_map = build_ibc_subject_context_map(ibc_datasets, context_frames)
        ibc_banks = {
            task: build_ibc_task_bank(
                task,
                ibc_datasets[task],
                ibc_datasets,
                ibc_context_map,
                context_frames,
                generation_frames,
                ts_cache,
                relevance_input_cfg,
            )
            for task in ibc_tasks
        }
        for task in ibc_tasks:
            contrasts = list(IBC_CONTRASTS[task])
            assert len(contrasts) > 0

            bank = ibc_banks[task]
            task_key = sanitize_name(task)
            task_prefix = f"ibc__{task_key}"
            task_dir = out_dir / "ibc" / task_key
            prep_glm = build_bank_prep(
                bank,
                "glm",
                model,
                tokenizer,
                device,
                batch_size,
                relevance_mode=default_relevance_mode,
                relevance_input_cfg=relevance_input_cfg,
            )
            curr_tr_seconds = float(tr_seconds if bank["dataset"].tr_seconds is None else bank["dataset"].tr_seconds)

            print(f"[hcp-eval] IBC {task}: glm generation", flush=True)
            synth_future_glm = generate_future_windows(
                model,
                tokenizer,
                prep_glm,
                device,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                context_guidance_scale=context_guidance_scale,
                seed=0,
                batch_size=batch_size,
                progress_label=f"IBC {task} glm",
                condition_guidance_scope="all_timepoints",
                stochastic_euler_eps=stochastic_euler_eps,
                stochastic_euler_noise_space=stochastic_euler_noise_space,
                euler_stop_sigma=euler_stop_sigma,
            )
            outputs = run_glm_for_ibc_task(
                task,
                bank,
                roi_labels,
                tr_seconds=curr_tr_seconds,
                contrasts=contrasts,
                synth_future=synth_future_glm,
            )
            glm_summary = save_glm(
                task_dir / "glm",
                outputs,
                guidance_scale=guidance_scale,
                context_guidance_scale=context_guidance_scale,
                bundle=stats_bundle,
                stats_prefix=f"{task_prefix}__glm",
                image_contrasts={contrasts[0]["name"]},
            )
            glm_outputs_by_relevance = {default_relevance_mode: outputs}
            glm_piece_drop_records = []
            glm_piece_drop_images = []
            for pieces in glm_piece_drop_grid:
                piece_label = _piece_drop_label(pieces)
                piece_key = sanitize_name(piece_label)
                print(f"[hcp-eval] IBC {task}: glm piece drop {piece_label}", flush=True)
                with _force_condition_piece_drop(model, pieces):
                    piece_future = generate_future_windows(
                        model,
                        tokenizer,
                        prep_glm,
                        device,
                        num_steps=num_steps,
                        guidance_scale=guidance_scale,
                        context_guidance_scale=context_guidance_scale,
                        seed=0,
                        batch_size=batch_size,
                        progress_label=f"IBC {task} glm piece drop {piece_label}",
                        condition_guidance_scope="all_timepoints",
                        stochastic_euler_eps=stochastic_euler_eps,
                        stochastic_euler_noise_space=stochastic_euler_noise_space,
                        euler_stop_sigma=euler_stop_sigma,
                    )
                piece_outputs = run_glm_for_ibc_task(
                    task,
                    bank,
                    roi_labels,
                    tr_seconds=curr_tr_seconds,
                    contrasts=contrasts,
                    synth_future=piece_future,
                )
                piece_summary = save_glm(
                    task_dir / "glm_piece_drop" / piece_key,
                    piece_outputs,
                    guidance_scale=guidance_scale,
                    context_guidance_scale=context_guidance_scale,
                    bundle=stats_bundle,
                    stats_prefix=f"{task_prefix}__glm_piece_drop__{piece_key}",
                    image_contrasts={contrasts[0]["name"]},
                )
                glm_piece_drop_records.extend(
                    {"piece": piece_label, **row}
                    for row in json.loads(piece_summary.to_json(orient="records"))
                )
                glm_piece_drop_images.extend(glm_report_images(
                    task_dir / "glm_piece_drop" / piece_key,
                    contrasts[0]["name"],
                    f"piece drop={piece_label}",
                    prefix="IBC GLM",
                ))
            glm_guidance_grid_records = []
            glm_guidance_grid_images = []
            for curr_guidance_scale, curr_context_guidance_scale in glm_guidance_grid:
                pair_key = _guidance_pair_dir(curr_guidance_scale, curr_context_guidance_scale)
                grid_stats_prefix = f"{task_prefix}__glm_guidance_grid__{pair_key}"
                if curr_guidance_scale == guidance_scale and curr_context_guidance_scale == context_guidance_scale:
                    grid_outputs = outputs
                    grid_future = synth_future_glm
                else:
                    print(
                        f"[hcp-eval] IBC {task}: glm guidance grid "
                        f"guidance={curr_guidance_scale} context={curr_context_guidance_scale}",
                        flush=True,
                    )
                    grid_future = generate_future_windows(
                        model,
                        tokenizer,
                        prep_glm,
                        device,
                        num_steps=num_steps,
                        guidance_scale=curr_guidance_scale,
                        context_guidance_scale=curr_context_guidance_scale,
                        seed=0,
                        batch_size=batch_size,
                        progress_label=(
                            f"IBC {task} glm guidance grid "
                            f"guidance={curr_guidance_scale} context={curr_context_guidance_scale}"
                        ),
                        condition_guidance_scope="all_timepoints",
                        stochastic_euler_eps=stochastic_euler_eps,
                        stochastic_euler_noise_space=stochastic_euler_noise_space,
                        euler_stop_sigma=euler_stop_sigma,
                    )
                    grid_outputs = run_glm_for_ibc_task(
                        task,
                        bank,
                        roi_labels,
                        tr_seconds=curr_tr_seconds,
                        contrasts=contrasts,
                        synth_future=grid_future,
                    )
                timeseries_summary = save_grid_timeseries_stats(
                    stats_bundle,
                    grid_stats_prefix,
                    bank["future_signal"][:, :, :, 0],
                    grid_future,
                )
                grid_summary = save_glm(
                    task_dir / "glm_guidance_grid" / pair_key,
                    grid_outputs,
                    guidance_scale=curr_guidance_scale,
                    context_guidance_scale=curr_context_guidance_scale,
                    bundle=stats_bundle,
                    stats_prefix=grid_stats_prefix,
                    image_contrasts={contrasts[0]["name"]},
                )
                grid_summary = grid_summary.assign(**timeseries_summary)
                glm_guidance_grid_records.extend(json.loads(grid_summary.to_json(orient="records")))
                glm_guidance_grid_images.extend(glm_report_images(
                    task_dir / "glm_guidance_grid" / pair_key,
                    contrasts[0]["name"],
                    f"guidance={curr_guidance_scale}, context guidance={curr_context_guidance_scale}",
                    prefix="IBC GLM",
                ))
            phrase_outputs_by_variant = {}
            for phrase_variant_idx in phrase_variant_indices:
                phrase_key = f"phrase_variant_{phrase_variant_idx}"
                if phrase_variant_idx == current_phrase_variant_idx:
                    phrase_outputs = outputs
                else:
                    phrase_cfg = _with_glm_phrase_variant(cfg, phrase_variant_idx)
                    phrase_ibc_datasets = build_ibc_eval_datasets(phrase_cfg, relevance_input_cfg)
                    assert task in phrase_ibc_datasets
                    phrase_context_map = build_ibc_subject_context_map(phrase_ibc_datasets, context_frames)
                    phrase_bank = build_ibc_task_bank(
                        task,
                        phrase_ibc_datasets[task],
                        phrase_ibc_datasets,
                        phrase_context_map,
                        context_frames,
                        generation_frames,
                        ts_cache,
                        relevance_input_cfg,
                    )
                    phrase_prep_glm = build_bank_prep(
                        phrase_bank,
                        "glm",
                        model,
                        tokenizer,
                        device,
                        batch_size,
                        relevance_mode=default_relevance_mode,
                        relevance_input_cfg=relevance_input_cfg,
                    )
                    phrase_tr_seconds = float(
                        tr_seconds if phrase_bank["dataset"].tr_seconds is None else phrase_bank["dataset"].tr_seconds
                    )
                    print(f"[hcp-eval] IBC {task}: glm {phrase_key}", flush=True)
                    phrase_future = generate_future_windows(
                        model,
                        tokenizer,
                        phrase_prep_glm,
                        device,
                        num_steps=num_steps,
                        guidance_scale=guidance_scale,
                        context_guidance_scale=context_guidance_scale,
                        seed=0,
                        batch_size=batch_size,
                        progress_label=f"IBC {task} glm {phrase_key}",
                        condition_guidance_scope="all_timepoints",
                        stochastic_euler_eps=stochastic_euler_eps,
                        stochastic_euler_noise_space=stochastic_euler_noise_space,
                        euler_stop_sigma=euler_stop_sigma,
                    )
                    phrase_outputs = run_glm_for_ibc_task(
                        task,
                        phrase_bank,
                        roi_labels,
                        tr_seconds=phrase_tr_seconds,
                        contrasts=contrasts,
                        synth_future=phrase_future,
                    )
                phrase_outputs_by_variant[phrase_variant_idx] = phrase_outputs
            phrase_summary_records = _save_glm_phrase_variant_summaries(
                phrase_outputs_by_variant,
                roi_labels,
                guidance_scale,
                context_guidance_scale,
                stats_bundle,
                f"{task_prefix}__glm_phrase_variants",
            )
            ibc_task_summary = {
                "glm": json.loads(glm_summary.to_json(orient="records")),
                "relevance_input_default": default_relevance_mode,
                "relevance_input_grid": relevance_grid,
                "glm_guidance_grid": glm_guidance_grid_records,
                "glm_piece_drop": glm_piece_drop_records,
                "glm_phrase_variants": phrase_summary_records,
                "glm_relevance": {
                    default_relevance_mode: json.loads(glm_summary.to_json(orient="records")),
                },
            }
            append_ibc_report_rows(
                summary_rows,
                task,
                task_dir,
                contrasts[0]["name"],
                glm_summary,
                guidance_scale,
                glm_piece_drop_records,
                glm_piece_drop_images,
                glm_guidance_grid_records,
                glm_guidance_grid_images,
            )
            if relevance_grid:
                relevance_rows = [
                    {"mode": default_relevance_mode, **row}
                    for row in json.loads(glm_summary.to_json(orient="records"))
                ]
                relevance_images = []
                for relevance_mode in relevance_grid:
                    prep_glm_variant = build_bank_prep(
                        bank,
                        "glm",
                        model,
                        tokenizer,
                        device,
                        batch_size,
                        relevance_mode=relevance_mode,
                        relevance_input_cfg=relevance_input_cfg,
                    )
                    print(f"[hcp-eval] IBC {task}: relevance input {relevance_mode}", flush=True)
                    variant_glm_future = generate_future_windows(
                        model,
                        tokenizer,
                        prep_glm_variant,
                        device,
                        num_steps=num_steps,
                        guidance_scale=guidance_scale,
                        context_guidance_scale=context_guidance_scale,
                        seed=0,
                        batch_size=batch_size,
                        progress_label=f"IBC {task} glm relevance input {relevance_mode}",
                        condition_guidance_scope="all_timepoints",
                        stochastic_euler_eps=stochastic_euler_eps,
                        stochastic_euler_noise_space=stochastic_euler_noise_space,
                        euler_stop_sigma=euler_stop_sigma,
                    )
                    variant_glm = run_glm_for_ibc_task(
                        task,
                        bank,
                        roi_labels,
                        tr_seconds=curr_tr_seconds,
                        contrasts=contrasts,
                        synth_future=variant_glm_future,
                    )
                    glm_outputs_by_relevance[relevance_mode] = variant_glm
                    variant_summary = save_glm(
                        task_dir / "relevance_input" / relevance_mode / "glm",
                        variant_glm,
                        guidance_scale=guidance_scale,
                        context_guidance_scale=context_guidance_scale,
                        bundle=stats_bundle,
                        stats_prefix=f"{task_prefix}__glm_variants__relevance_{sanitize_name(relevance_mode)}",
                        image_contrasts={contrasts[0]["name"]},
                    )
                    variant_records = json.loads(variant_summary.to_json(orient="records"))
                    ibc_task_summary["glm_relevance"][relevance_mode] = variant_records
                    relevance_rows.extend({"mode": relevance_mode, **row} for row in variant_records)
                    relevance_images.extend(glm_report_images(
                        task_dir / "relevance_input" / relevance_mode / "glm",
                        contrasts[0]["name"],
                        f"relevance={relevance_mode}",
                        prefix="IBC GLM",
                    ))
                append_ibc_relevance_report_row(summary_rows, task, relevance_rows, relevance_images)
            phrase_relevance_records = {}
            if phrase_variant_indices:
                for relevance_mode in relevance_grid:
                    relevance_phrase_outputs_by_variant = {}
                    for phrase_variant_idx in phrase_variant_indices:
                        phrase_key = f"phrase_variant_{phrase_variant_idx}"
                        if phrase_variant_idx == current_phrase_variant_idx:
                            phrase_outputs = glm_outputs_by_relevance[relevance_mode]
                        else:
                            phrase_cfg = _with_glm_phrase_variant(cfg, phrase_variant_idx)
                            phrase_ibc_datasets = build_ibc_eval_datasets(phrase_cfg, relevance_input_cfg)
                            assert task in phrase_ibc_datasets
                            phrase_context_map = build_ibc_subject_context_map(phrase_ibc_datasets, context_frames)
                            phrase_bank = build_ibc_task_bank(
                                task,
                                phrase_ibc_datasets[task],
                                phrase_ibc_datasets,
                                phrase_context_map,
                                context_frames,
                                generation_frames,
                                ts_cache,
                                relevance_input_cfg,
                            )
                            phrase_prep_glm = build_bank_prep(
                                phrase_bank,
                                "glm",
                                model,
                                tokenizer,
                                device,
                                batch_size,
                                relevance_mode=relevance_mode,
                                relevance_input_cfg=relevance_input_cfg,
                            )
                            phrase_tr_seconds = float(
                                tr_seconds if phrase_bank["dataset"].tr_seconds is None else phrase_bank["dataset"].tr_seconds
                            )
                            print(f"[hcp-eval] IBC {task}: glm relevance input {relevance_mode} {phrase_key}", flush=True)
                            phrase_future = generate_future_windows(
                                model,
                                tokenizer,
                                phrase_prep_glm,
                                device,
                                num_steps=num_steps,
                                guidance_scale=guidance_scale,
                                context_guidance_scale=context_guidance_scale,
                                seed=0,
                                batch_size=batch_size,
                                progress_label=f"IBC {task} glm relevance input {relevance_mode} {phrase_key}",
                                condition_guidance_scope="all_timepoints",
                                stochastic_euler_eps=stochastic_euler_eps,
                                stochastic_euler_noise_space=stochastic_euler_noise_space,
                                euler_stop_sigma=euler_stop_sigma,
                            )
                            phrase_outputs = run_glm_for_ibc_task(
                                task,
                                phrase_bank,
                                roi_labels,
                                tr_seconds=phrase_tr_seconds,
                                contrasts=contrasts,
                                synth_future=phrase_future,
                            )
                        relevance_phrase_outputs_by_variant[phrase_variant_idx] = phrase_outputs
                    relevance_key = sanitize_name(relevance_mode)
                    phrase_relevance_records[relevance_mode] = _save_glm_phrase_variant_summaries(
                        relevance_phrase_outputs_by_variant,
                        roi_labels,
                        guidance_scale,
                        context_guidance_scale,
                        stats_bundle,
                        f"{task_prefix}__glm_variants__relevance_{relevance_key}__glm_phrase_variants",
                    )
            ibc_task_summary["glm_phrase_variants_relevance"] = phrase_relevance_records
            summary_json["ibc"][task] = ibc_task_summary

    write_eval_outputs(out_dir, summary_json, stats_bundle, summary_rows)
