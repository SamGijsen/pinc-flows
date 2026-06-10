import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .bank import (
    build_bank_prep,
    build_hcp_dataset,
    build_subject_context_map,
    build_task_bank,
    generate_future_windows,
)
from .common import base_subject_id
from .glm import HCP_TASKS


CONTEXT_CONDITIONS = (
    "correct",
    "same_subject_misaligned",
    "different_subject",
    "no_context",
)
CONTEXT_KEYS = (
    "z_context",
    "cond_disc_ctx",
    "cond_disc_weight_ctx",
    "cond_cont_ctx",
    "cond_mode_ctx",
    "drop_mask_ctx",
    "relevance_scores_ctx",
    "drop_mask_relevance_ctx",
)


@dataclass(frozen=True)
class FcEvalRuntime:
    cfg: dict
    eval_cfg: dict
    out_dir: Path
    holdout_subject_ids_path: str
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    relevance_input_cfg: dict
    ts_cache: dict


def _with_phrase_variant(cfg, phrase_variant_id):
    out = copy.deepcopy(cfg)
    data_cfg = out["data"]
    lang_cfgs = [data_cfg["val_condition_cont_language_events"]]
    for spec in data_cfg["val_datasets"]:
        if "condition_cont_language_events" in spec:
            lang_cfgs.append(spec["condition_cont_language_events"])
    for lang_cfg in lang_cfgs:
        lang_cfg["variant_sampling"] = "fixed"
        lang_cfg["fixed_variant_idx"] = int(phrase_variant_id)
    return out


def _build_bank(runtime, task, phrase_variant_id):
    cfg = _with_phrase_variant(runtime.cfg, phrase_variant_id)
    context_frames = int(cfg["dynamics"]["context_frames"])
    datasets = {
        hcp_task: build_hcp_dataset(
            cfg,
            hcp_task,
            runtime.holdout_subject_ids_path,
            runtime.relevance_input_cfg,
        )
        for hcp_task in HCP_TASKS
    }
    context_map = build_subject_context_map(datasets, context_frames)
    return build_task_bank(
        task,
        datasets,
        context_map,
        cfg,
        runtime.ts_cache,
        runtime.relevance_input_cfg,
    )


def _prep_with_context_rows(prep, rows):
    out = dict(prep)
    for key in CONTEXT_KEYS:
        value = prep[key]
        out[key] = None if value is None else value[rows]
    return out


def _donor_rows(bank, context_condition, rng, min_gap_trs):
    n = len(bank["run_ids"])
    if context_condition == "correct":
        return np.arange(n, dtype=np.int64)
    if context_condition == "no_context":
        return np.full(n, -1, dtype=np.int64)

    subject = np.asarray([base_subject_id(run_id) for run_id in bank["run_ids"].tolist()], dtype=str)
    run_id = np.asarray([str(x) for x in bank["run_ids"].tolist()], dtype=str)
    block_start = np.asarray(bank["block_starts"], dtype=np.int64)
    rows = np.empty(n, dtype=np.int64)

    for i in range(n):
        if context_condition == "same_subject_misaligned":
            candidates = np.flatnonzero(
                (subject == subject[i])
                & (run_id == run_id[i])
                & (np.abs(block_start - block_start[i]) >= int(min_gap_trs))
            )
        elif context_condition == "different_subject":
            candidates = np.flatnonzero(subject != subject[i])
        else:
            raise ValueError(f"Unknown FC context_condition={context_condition!r}")
        assert candidates.size > 0
        rows[i] = int(rng.choice(candidates))
    return rows


def _real_pairs(subject_id, condition_id):
    pairs = []
    for subject in sorted(set(subject_id.tolist())):
        for condition in sorted(set(condition_id.tolist())):
            rows = np.flatnonzero((subject_id == subject) & (condition_id == condition))
            for i, row_a in enumerate(rows):
                for row_b in rows[i + 1:]:
                    pairs.append((int(row_a), int(row_b)))
    if not pairs:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    arr = np.asarray(pairs, dtype=np.int64)
    return arr[:, 0], arr[:, 1]


def _assert_same_real_rows(first, curr):
    np.testing.assert_array_equal(first["run_ids"], curr["run_ids"])
    np.testing.assert_array_equal(first["block_names"], curr["block_names"])
    np.testing.assert_array_equal(first["block_starts"], curr["block_starts"])
    np.testing.assert_allclose(first["future_signal"], curr["future_signal"])


def _save_task_npz(path, arrays, dtype_name):
    dtype = {"float32": np.float32, "float16": np.float16}[str(dtype_name)]
    arrays["generated_timeseries"] = arrays["generated_timeseries"].astype(dtype)
    arrays["real_timeseries"] = arrays["real_timeseries"].astype(dtype)
    np.savez_compressed(path, **arrays)


def run_fc_eval(runtime):
    fc_cfg = runtime.eval_cfg["fc_eval"]
    if not bool(fc_cfg["enabled"]):
        return

    tasks = [str(task).upper() for task in fc_cfg.get("tasks", runtime.eval_cfg["tasks"])]
    phrase_variant_ids = np.asarray(fc_cfg.get("phrase_variant_indices", [0, 1, 2, 3, 4, 5]), dtype=np.int64)
    assert len(tasks) > 0
    assert phrase_variant_ids.size > 0
    for task in tasks:
        assert task in HCP_TASKS
    min_gap_trs = int(fc_cfg.get("same_subject_min_gap_trs", 45))
    output_dtype = str(fc_cfg.get("output_dtype", "float32"))
    assert output_dtype in ("float32", "float16")

    eval_cfg = runtime.eval_cfg
    num_steps = int(eval_cfg["num_steps"])
    guidance_scale = float(eval_cfg["guidance_scale"])
    context_guidance_scale = float(eval_cfg["context_guidance_scale"])
    batch_size = int(eval_cfg["batch_size"])
    stochastic_euler_eps = float(eval_cfg.get("stochastic_euler_eps", 0.0))
    stochastic_euler_noise_space = str(eval_cfg.get("stochastic_euler_noise_space", "latent")).lower()
    euler_stop_sigma = float(eval_cfg.get("euler_stop_sigma", 1.0))
    seed = int(fc_cfg.get("seed", 0))

    for task in tasks:
        first_bank = None
        generated = []
        generated_real_index = []
        gen_subject_id = []
        gen_condition_id = []
        gen_context_condition = []
        gen_phrase_variant_id = []
        gen_window_id = []
        gen_task = []
        gen_run_id = []
        gen_block_start = []
        context_source_window_id = []
        context_source_run_id = []
        context_source_block_start = []

        for phrase_variant_id in phrase_variant_ids.tolist():
            bank = _build_bank(runtime, task, int(phrase_variant_id))
            if first_bank is None:
                first_bank = bank
            else:
                _assert_same_real_rows(first_bank, bank)

            prep = build_bank_prep(
                bank,
                "continuation",
                runtime.model,
                runtime.tokenizer,
                runtime.device,
                batch_size,
                relevance_mode=runtime.relevance_input_cfg["default"],
                relevance_input_cfg=runtime.relevance_input_cfg,
            )
            n = len(bank["run_ids"])

            for context_condition in CONTEXT_CONDITIONS:
                donor_rows = _donor_rows(
                    bank,
                    context_condition,
                    np.random.default_rng(seed + CONTEXT_CONDITIONS.index(context_condition)),
                    min_gap_trs,
                )
                curr_prep = (
                    prep
                    if context_condition in ("correct", "no_context")
                    else _prep_with_context_rows(prep, donor_rows)
                )
                synth = generate_future_windows(
                    runtime.model,
                    runtime.tokenizer,
                    curr_prep,
                    runtime.device,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    context_guidance_scale=context_guidance_scale,
                    seed=seed,
                    batch_size=batch_size,
                    progress_label=f"{task} FC {context_condition} phrase {phrase_variant_id}",
                    condition_guidance_scope="all_timepoints",
                    drop_latent_context=context_condition == "no_context",
                    stochastic_euler_eps=stochastic_euler_eps,
                    stochastic_euler_noise_space=stochastic_euler_noise_space,
                    euler_stop_sigma=euler_stop_sigma,
                )
                generated.append(synth)
                generated_real_index.append(np.arange(n, dtype=np.int64))
                gen_subject_id.extend(base_subject_id(x) for x in bank["run_ids"].tolist())
                gen_condition_id.extend(str(x) for x in bank["block_names"].tolist())
                gen_context_condition.extend([context_condition] * n)
                gen_phrase_variant_id.extend([int(phrase_variant_id)] * n)
                gen_window_id.extend(np.arange(n, dtype=np.int64).tolist())
                gen_task.extend([task] * n)
                gen_run_id.extend(str(x) for x in bank["run_ids"].tolist())
                gen_block_start.extend(bank["block_starts"].astype(np.int64).tolist())
                context_source_window_id.extend(donor_rows.astype(np.int64).tolist())
                context_source_run_id.extend(
                    "" if row < 0 else str(bank["run_ids"][row])
                    for row in donor_rows.tolist()
                )
                context_source_block_start.extend(
                    -1 if row < 0 else int(bank["block_starts"][row])
                    for row in donor_rows.tolist()
                )

        assert first_bank is not None
        real = first_bank["future_signal"][:, :, :, 0].astype(np.float32)
        real_subject_id = np.asarray(
            [base_subject_id(run_id) for run_id in first_bank["run_ids"].tolist()],
            dtype=str,
        )
        real_condition_id = np.asarray([str(x) for x in first_bank["block_names"].tolist()], dtype=str)
        real_pair_a, real_pair_b = _real_pairs(real_subject_id, real_condition_id)

        task_dir = runtime.out_dir / task.lower()
        task_dir.mkdir(parents=True, exist_ok=True)
        out_path = task_dir / "fc_eval.npz"
        _save_task_npz(
            out_path,
            {
                "generated_timeseries": np.concatenate(generated, axis=0),
                "generated_real_index": np.concatenate(generated_real_index, axis=0),
                "real_timeseries": real,
                "real_pair_a": real_pair_a,
                "real_pair_b": real_pair_b,
                "subject_id": np.asarray(gen_subject_id, dtype=str),
                "condition_id": np.asarray(gen_condition_id, dtype=str),
                "context_condition": np.asarray(gen_context_condition, dtype=str),
                "phrase_variant_id": np.asarray(gen_phrase_variant_id, dtype=np.int64),
                "window_id": np.asarray(gen_window_id, dtype=np.int64),
                "task": np.asarray(gen_task, dtype=str),
                "run_id": np.asarray(gen_run_id, dtype=str),
                "block_start": np.asarray(gen_block_start, dtype=np.int64),
                "context_source_window_id": np.asarray(context_source_window_id, dtype=np.int64),
                "context_source_run_id": np.asarray(context_source_run_id, dtype=str),
                "context_source_block_start": np.asarray(context_source_block_start, dtype=np.int64),
                "real_subject_id": real_subject_id,
                "real_condition_id": real_condition_id,
                "real_window_id": np.arange(real.shape[0], dtype=np.int64),
                "real_task": np.asarray([task] * real.shape[0], dtype=str),
                "real_run_id": np.asarray([str(x) for x in first_bank["run_ids"].tolist()], dtype=str),
                "real_block_start": first_bank["block_starts"].astype(np.int64),
                "context_conditions": np.asarray(CONTEXT_CONDITIONS, dtype=str),
                "phrase_variant_ids": phrase_variant_ids.astype(np.int64),
                "same_subject_min_gap_trs": np.asarray(min_gap_trs, dtype=np.int64),
            },
            output_dtype,
        )
        print(f"[hcp-eval] {task}: saved FC eval to {out_path}", flush=True)
