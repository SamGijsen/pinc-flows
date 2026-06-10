from pathlib import Path
import re

import numpy as np
import torch

from datasets.dynamics_dataset import SequenceDataset
from util.dynamics_inference import (
    _batch_to_prep,
    _concat_prep_batches,
    _decode_to_bold,
    _sample_future,
    _slice_prep,
    add_stochastic_eval_noise,
)

from .common import base_subject_id, pick
from .glm import get_block_start_event_starts
from .relevance_input import get_requested_relevance_modes, resolve_relevance_input


HCP_TASK_PATH_RE = re.compile(r"^HCP_YA_[A-Z]+_")


def _task_h5_fp(cfg, task):
    hcp_paths = [
        Path(str(ds["path"]))
        for ds in cfg["data"]["val_datasets"]
        if str(ds.get("name", "")).lower() == "hcp" or str(ds["path"]).split("/")[-1].startswith("HCP_YA_")
    ]
    assert hcp_paths

    for path in hcp_paths:
        if path.name.startswith(f"HCP_YA_{task}_"):
            return path

    template = hcp_paths[0]
    name = HCP_TASK_PATH_RE.sub(f"HCP_YA_{task}_", template.name, count=1)
    assert name != template.name
    return template.parent / name


def build_hcp_dataset(cfg, task, holdout_subject_ids_path, relevance_input_cfg=None):
    data_cfg = cfg["data"]
    dyn_cfg = cfg["dynamics"]
    tok_cfg = cfg.get("tokenizer", {})
    atlas_names = pick(data_cfg, "val", "atlas_names", data_cfg.get("atlas_names"))
    context_frames = int(dyn_cfg.get("context_frames", 8))
    generation_frames = int(dyn_cfg.get("generation_frames", 16))
    sequence_length = context_frames + generation_frames
    crop_length = int(tok_cfg.get("input_timesteps", 1))
    input_stride = int(pick(data_cfg, "val", "input_stride", 1))
    assert crop_length == 1
    assert input_stride == 1
    condition_cont_language_events = pick(data_cfg, "val", "condition_cont_language_events")
    relevance_cfg = dyn_cfg.get("relevance")
    if condition_cont_language_events is not None and isinstance(relevance_cfg, dict):
        relevance_h5_group = relevance_cfg.get("h5_group")
        if relevance_h5_group is not None:
            assert isinstance(relevance_h5_group, str) and relevance_h5_group.strip()
            condition_cont_language_events = dict(condition_cont_language_events)
            condition_cont_language_events["relevance_h5_group"] = relevance_h5_group
            if relevance_input_cfg is not None and relevance_input_cfg.get("ood_roi_h5_group") is not None:
                condition_cont_language_events["relevance_ood_h5_group"] = relevance_input_cfg["ood_roi_h5_group"]

    return SequenceDataset(
        data_path=str(_task_h5_fp(cfg, task)),
        sequence_length=sequence_length,
        crop_length=crop_length,
        condition_label_name=pick(data_cfg, "val", "condition_label_name"),
        num_conditions=int(dyn_cfg.get("num_conditions", 1)),
        atlas_names=atlas_names,
        subject_ids_path=holdout_subject_ids_path,
        input_stride=input_stride,
        unlabeled_condition=int(pick(data_cfg, "val", "unlabeled_condition", 0)),
        condition_mode=pick(data_cfg, "val", "condition_mode", data_cfg.get("condition_mode", "cont")),
        condition_cont_label_name=pick(data_cfg, "val", "condition_cont_label_name"),
        condition_cont_idx_label_name=pick(data_cfg, "val", "condition_cont_idx_label_name"),
        condition_cont_scale_label_name=pick(data_cfg, "val", "condition_cont_scale_label_name"),
        condition_cont_embeddings_name=pick(data_cfg, "val", "condition_cont_embeddings_name"),
        condition_cont_language_events=condition_cont_language_events,
        condition_relevance_path=None,
        condition_cont_dim=pick(data_cfg, "val", "condition_cont_dim", dyn_cfg.get("condition_cont_dim")),
        event_mapping=pick(data_cfg, "val", "event_mapping"),
        tr_seconds=pick(data_cfg, "val", "tr_seconds"),
        sample_mode="anchored_eval",
        ood_tag_label_name=pick(data_cfg, "val", "ood_tag_label_name"),
        allowed_ood_tags=pick(data_cfg, "val", "allowed_ood_tags"),
        anchor_tag_label_name=pick(data_cfg, "val", "anchor_tag_label_name"),
        anchor_tags=pick(data_cfg, "val", "anchor_tags"),
        anchor_crop_index=pick(data_cfg, "val", "anchor_crop_index", context_frames),
        subject_token_enabled=bool(dyn_cfg.get("subject_token_enabled", False)),
        subject_context_length=dyn_cfg.get("subject_context_length"),
        subject_min_gap=pick(data_cfg, "val", "subject_min_gap", data_cfg.get("subject_min_gap")),
        age_label_name=pick(data_cfg, "val", "age_label_name"),
        sex_label_name=pick(data_cfg, "val", "sex_label_name"),
        motion_label_name=pick(data_cfg, "val", "motion_label_name"),
        field_strength_t=pick(data_cfg, "val", "field_strength_t"),
    )


def get_zscored_ts(cache, task, dataset, subj_idx):
    key = (task, int(subj_idx))
    if key not in cache:
        cache[key] = dataset._zscore_roiwise(dataset._load_subject_timeseries(int(subj_idx)))
    return cache[key]


def _first_nonfixation_start(dataset, subj_idx):
    first = None
    for event_name in dataset.file["events"].keys():
        if event_name == "fixation":
            continue
        arr = np.asarray(dataset.file["events"][event_name][subj_idx], dtype=np.float32)
        idx = np.flatnonzero(arr > 1e-6)
        if idx.size == 0:
            continue
        value = int(idx[0])
        if first is None or value < first:
            first = value
    return first


def build_subject_context_map(datasets, context_frames):
    candidates = {}
    for task, dataset in datasets.items():
        for local_idx, run_id in enumerate(dataset.subject_ids):
            subj_idx = int(dataset.subject_indices[local_idx])
            first = _first_nonfixation_start(dataset, subj_idx)
            if first is None or first < int(context_frames):
                continue
            base_subject = base_subject_id(run_id)
            candidates.setdefault(base_subject, []).append((task, subj_idx, str(run_id)))

    rng = np.random.default_rng(0)
    out = {}
    for base_subject, rows in candidates.items():
        pick_idx = int(rng.integers(0, len(rows)))
        out[base_subject] = rows[pick_idx]
    return out


def build_ibc_subject_context_map(datasets, context_frames):
    candidates = {}
    for task, dataset in datasets.items():
        for local_idx, run_id in enumerate(dataset.subject_ids):
            subj_idx = int(dataset.subject_indices[local_idx])
            valid_tp = int(dataset.file["valid_timepoints"][subj_idx])
            if valid_tp < int(context_frames):
                continue
            base_subject = base_subject_id(run_id)
            candidates.setdefault(base_subject, []).append((task, subj_idx, str(run_id)))

    rng = np.random.default_rng(0)
    out = {}
    for base_subject, rows in candidates.items():
        pick_idx = int(rng.integers(0, len(rows)))
        out[base_subject] = rows[pick_idx]
    return out


def build_condition_arrays(dataset, subj_idx, crop_starts, relevance_subject_id, relevance_source="train"):
    if dataset.condition_mode == "cont" and dataset.condition_cont_source == "language_event_pool":
        variants = dataset._sample_language_event_variants_for_sequence()
    else:
        variants = None

    disc = []
    disc_weight = []
    cont = []
    mode = []
    relevance = [] if dataset.has_direct_relevance_scores() else None
    relevance_embedding_type = None
    for crop_start in crop_starts:
        crop_start = int(crop_start)
        if dataset.condition_mode == "cont":
            if dataset.condition_cont_source == "language_event_pool":
                cond_disc, cond_disc_weight, cond_cont = dataset._get_volume_language_event_condition_parts(
                    subj_idx,
                    crop_start,
                    language_event_sequence_variants=variants,
                )
                disc.append(cond_disc)
                disc_weight.append(cond_disc_weight)
                cont.append(cond_cont)
                mode.append(1)
                if relevance is not None:
                    scores, relevance_embedding_type = dataset._get_volume_relevance_scores(
                        subj_idx,
                        crop_start,
                        language_event_sequence_variants=variants,
                        relevance_subject_id=relevance_subject_id,
                        relevance_source=relevance_source,
                    )
                    relevance.append(scores)
                continue

            disc.append(-1)
            disc_weight.append(np.zeros(dataset.num_conditions, dtype=np.float32))
            cont.append(dataset._get_volume_condition_cont(subj_idx, crop_start, variants))
            mode.append(1)
            if relevance is not None:
                relevance.append(np.zeros(dataset.num_rois, dtype=np.float32))
            continue

        crop_end = crop_start + dataset.crop_length
        disc_id = dataset._get_crop_condition(subj_idx, crop_start, crop_end)
        disc.append(disc_id)
        disc_weight.append(np.zeros(dataset.num_conditions, dtype=np.float32))
        cont.append(np.zeros(dataset.condition_cont_dim, dtype=np.float32))
        mode.append(0)
        if relevance is not None:
            relevance.append(np.zeros(dataset.num_rois, dtype=np.float32))

    relevance_arr = None if relevance is None else np.stack(relevance, axis=0).astype(np.float32)
    return {
        "condition_disc": np.asarray(disc, dtype=np.int64),
        "condition_disc_weight": np.stack(disc_weight, axis=0).astype(np.float32),
        "condition_cont": np.stack(cont, axis=0).astype(np.float32),
        "condition_mode": np.asarray(mode, dtype=np.int64),
        "relevance_scores": relevance_arr,
        "relevance_embedding_type": relevance_embedding_type,
    }


def _build_social_fixation_context_conds(social_dataset, subj_idx, context_frames, relevance_source):
    if social_dataset.condition_mode != "cont":
        return None
    if social_dataset.condition_cont_source != "language_event_pool":
        return None

    variant_idx = social_dataset.language_event_fixed_variant_idx
    if variant_idx is None:
        variant_idx = 0
    piece_dim = int(social_dataset.language_event_piece_dim)
    instruction = social_dataset._get_language_event_pool_row("instruction:fixation", variant_idx)
    sensory = social_dataset._get_language_event_pool_row("sensory:fixation", variant_idx)
    response = np.zeros(piece_dim, dtype=np.float32)
    response_special = np.asarray([1.0, 0.0], dtype=np.float32)
    condition_cont = np.concatenate([instruction, sensory, response, response_special], axis=0)

    relevance_scores = None
    relevance_embedding_type = None
    if social_dataset.has_direct_relevance_scores():
        if social_dataset.language_event_h5_relevance_events:
            relevance_scores = np.asarray(
                social_dataset.language_event_h5_relevance_events["events/fixation"][subj_idx],
                dtype=np.float32,
            )
            relevance_embedding_type = 0
        else:
            instruction_rel = social_dataset._get_language_event_relevance_row("instruction:fixation", variant_idx)
            sensory_rel = social_dataset._get_language_event_relevance_row("sensory:fixation", variant_idx)
            response_rel = np.zeros(social_dataset.num_rois, dtype=np.float32)
            relevance_scores = np.stack([instruction_rel, sensory_rel, response_rel], axis=0).astype(np.float32)
            relevance_embedding_type = 0

    return {
        "condition_disc": np.full(context_frames, -1, dtype=np.int64),
        "condition_disc_weight": np.zeros((context_frames, social_dataset.num_conditions), dtype=np.float32),
        "condition_cont": np.repeat(condition_cont[None, :], context_frames, axis=0).astype(np.float32),
        "condition_mode": np.ones(context_frames, dtype=np.int64),
        "relevance_scores": None if relevance_scores is None else np.repeat(relevance_scores[None, ...], context_frames, axis=0),
        "relevance_embedding_type": relevance_embedding_type,
    }


def build_task_bank(task, datasets, context_map, cfg, ts_cache, relevance_input_cfg):
    dataset = datasets[task]
    context_frames = int(cfg["dynamics"].get("context_frames", 8))
    generation_frames = int(cfg["dynamics"].get("generation_frames", 16))
    requested_relevance_modes = get_requested_relevance_modes(relevance_input_cfg)
    need_ood_roi = "ood_roi" in requested_relevance_modes
    social_dataset = datasets["SOCIAL"]
    social_fixation_subject_idx = None
    if (
        social_dataset.condition_mode == "cont"
        and social_dataset.condition_cont_source == "language_event_pool"
    ):
        social_fixation_subject_idx = {
            base_subject_id(run_id): int(social_dataset.subject_indices[local_idx])
            for local_idx, run_id in enumerate(social_dataset.subject_ids)
        }

    signal_cont = []
    signal_glm = []
    future_signal = []
    run_ids = []
    run_indices = []
    block_names = []
    block_starts = []
    condition_disc_cont = []
    condition_disc_weight_cont = []
    condition_cont_cont = []
    condition_mode_cont = []
    condition_disc_glm = []
    condition_disc_weight_glm = []
    condition_cont_glm = []
    condition_mode_glm = []
    relevance_scores_cont = []
    relevance_scores_glm = []
    relevance_scores_ood_roi_cont = []
    relevance_scores_ood_roi_glm = []
    has_direct_relevance = dataset.has_direct_relevance_scores()
    has_relevance = None
    has_ood_roi = need_ood_roi and has_direct_relevance and bool(dataset.language_event_h5_relevance_ood_events)
    runs = {}

    for local_idx, run_id in enumerate(dataset.subject_ids):
        subj_idx = int(dataset.subject_indices[local_idx])
        base_subject = base_subject_id(run_id)
        assert base_subject in context_map
        ctx_task, ctx_subj_idx, ctx_run_id = context_map[base_subject]
        ctx_dataset = datasets[ctx_task]

        starts_by_name = get_block_start_event_starts(dataset.file, task, subj_idx)
        ordered = sorted((int(start), str(name)) for name, starts in starts_by_name.items() for start in starts)
        if not ordered:
            continue

        valid_tp = int(dataset.file["valid_timepoints"][subj_idx])
        target_ts = get_zscored_ts(ts_cache, task, dataset, subj_idx)
        pretask_ts = get_zscored_ts(ts_cache, ctx_task, ctx_dataset, ctx_subj_idx)

        if social_fixation_subject_idx is None:
            glm_ctx_conds = build_condition_arrays(
                ctx_dataset,
                ctx_subj_idx,
                np.arange(context_frames, dtype=np.int64),
                relevance_subject_id=str(ctx_run_id),
            )
        else:
            assert base_subject in social_fixation_subject_idx
            glm_ctx_conds = _build_social_fixation_context_conds(
                social_dataset,
                social_fixation_subject_idx[base_subject],
                context_frames,
                "true_roi",
            )
        glm_ctx_signal = pretask_ts[:, :context_frames]

        run_row_indices = []
        for start, block_name in ordered:
            if start < context_frames or start + generation_frames > valid_tp:
                continue

            cont_ctx_signal = target_ts[:, start - context_frames:start]
            cont_ctx_conds = build_condition_arrays(
                dataset,
                subj_idx,
                np.arange(start - context_frames, start, dtype=np.int64),
                relevance_subject_id=str(run_id),
                relevance_source="true_roi",
            )
            fut_signal = target_ts[:, start:start + generation_frames]
            fut_conds = build_condition_arrays(
                dataset,
                subj_idx,
                np.arange(start, start + generation_frames, dtype=np.int64),
                relevance_subject_id=str(run_id),
                relevance_source="true_roi",
            )

            signal_cont.append(np.concatenate([cont_ctx_signal, fut_signal], axis=1).T[:, :, None].astype(np.float32))
            signal_glm.append(np.concatenate([glm_ctx_signal, fut_signal], axis=1).T[:, :, None].astype(np.float32))
            future_signal.append(fut_signal.T[:, :, None].astype(np.float32))

            condition_disc_cont.append(np.concatenate([cont_ctx_conds["condition_disc"], fut_conds["condition_disc"]], axis=0))
            condition_disc_weight_cont.append(
                np.concatenate([cont_ctx_conds["condition_disc_weight"], fut_conds["condition_disc_weight"]], axis=0)
            )
            condition_cont_cont.append(np.concatenate([cont_ctx_conds["condition_cont"], fut_conds["condition_cont"]], axis=0))
            condition_mode_cont.append(np.concatenate([cont_ctx_conds["condition_mode"], fut_conds["condition_mode"]], axis=0))

            condition_disc_glm.append(np.concatenate([glm_ctx_conds["condition_disc"], fut_conds["condition_disc"]], axis=0))
            condition_disc_weight_glm.append(
                np.concatenate([glm_ctx_conds["condition_disc_weight"], fut_conds["condition_disc_weight"]], axis=0)
            )
            condition_cont_glm.append(np.concatenate([glm_ctx_conds["condition_cont"], fut_conds["condition_cont"]], axis=0))
            condition_mode_glm.append(np.concatenate([glm_ctx_conds["condition_mode"], fut_conds["condition_mode"]], axis=0))

            full_rel_cont = None if cont_ctx_conds["relevance_scores"] is None else np.concatenate(
                [cont_ctx_conds["relevance_scores"], fut_conds["relevance_scores"]], axis=0
            )
            full_rel_glm = None if glm_ctx_conds["relevance_scores"] is None else np.concatenate(
                [glm_ctx_conds["relevance_scores"], fut_conds["relevance_scores"]], axis=0
            )
            if has_relevance is None:
                has_relevance = full_rel_cont is not None
            if has_relevance:
                assert full_rel_cont is not None
                assert full_rel_glm is not None
                relevance_scores_cont.append(full_rel_cont)
                relevance_scores_glm.append(full_rel_glm)
            if has_ood_roi:
                ood_cont_ctx = build_condition_arrays(
                    dataset,
                    subj_idx,
                    np.arange(start - context_frames, start, dtype=np.int64),
                    relevance_subject_id=str(run_id),
                    relevance_source="ood_roi",
                )
                ood_fut = build_condition_arrays(
                    dataset,
                    subj_idx,
                    np.arange(start, start + generation_frames, dtype=np.int64),
                    relevance_subject_id=str(run_id),
                    relevance_source="ood_roi",
                )
                relevance_scores_ood_roi_cont.append(np.concatenate(
                    [ood_cont_ctx["relevance_scores"], ood_fut["relevance_scores"]], axis=0))
                glm_ctx_rel = full_rel_glm[:context_frames] if full_rel_glm is not None else np.zeros(
                    (context_frames,) + ood_fut["relevance_scores"].shape[1:], dtype=np.float32)
                relevance_scores_ood_roi_glm.append(np.concatenate(
                    [glm_ctx_rel, ood_fut["relevance_scores"]], axis=0))


            idx = len(run_ids)
            run_ids.append(str(run_id))
            run_indices.append(int(subj_idx))
            block_names.append(str(block_name))
            block_starts.append(int(start))
            run_row_indices.append(idx)

        if run_row_indices:
            runs[str(run_id)] = run_row_indices

    assert len(run_ids) > 0
    return {
        "task": task,
        "dataset": dataset,
        "signal_cont": np.stack(signal_cont, axis=0).astype(np.float32),
        "signal_glm": np.stack(signal_glm, axis=0).astype(np.float32),
        "future_signal": np.stack(future_signal, axis=0).astype(np.float32),
        "run_ids": np.asarray(run_ids, dtype=object),
        "run_indices": np.asarray(run_indices, dtype=np.int64),
        "block_names": np.asarray(block_names, dtype=object),
        "block_starts": np.asarray(block_starts, dtype=np.int64),
        "condition_disc_cont": np.stack(condition_disc_cont, axis=0).astype(np.int64),
        "condition_disc_weight_cont": np.stack(condition_disc_weight_cont, axis=0).astype(np.float32),
        "condition_cont_cont": np.stack(condition_cont_cont, axis=0).astype(np.float32),
        "condition_mode_cont": np.stack(condition_mode_cont, axis=0).astype(np.int64),
        "condition_disc_glm": np.stack(condition_disc_glm, axis=0).astype(np.int64),
        "condition_disc_weight_glm": np.stack(condition_disc_weight_glm, axis=0).astype(np.float32),
        "condition_cont_glm": np.stack(condition_cont_glm, axis=0).astype(np.float32),
        "condition_mode_glm": np.stack(condition_mode_glm, axis=0).astype(np.int64),
        "relevance_scores_cont": None if not has_relevance else np.stack(relevance_scores_cont, axis=0).astype(np.float32),
        "relevance_scores_glm": None if not has_relevance else np.stack(relevance_scores_glm, axis=0).astype(np.float32),
        "relevance_scores_ood_roi_cont": None if not has_ood_roi else np.stack(relevance_scores_ood_roi_cont, axis=0).astype(np.float32),
        "relevance_scores_ood_roi_glm": None if not has_ood_roi else np.stack(relevance_scores_ood_roi_glm, axis=0).astype(np.float32),
        "runs": runs,
    }


def build_ibc_task_bank(task, dataset, datasets, context_map, context_frames, generation_frames, ts_cache, relevance_input_cfg):
    requested_relevance_modes = get_requested_relevance_modes(relevance_input_cfg)
    need_ood_roi = "ood_roi" in requested_relevance_modes

    signal_glm = []
    future_signal = []
    run_ids = []
    run_indices = []
    condition_disc_glm = []
    condition_disc_weight_glm = []
    condition_cont_glm = []
    condition_mode_glm = []
    condition_drop_mask_glm = []
    relevance_scores_glm = []
    relevance_scores_ood_roi_glm = []
    drop_mask_relevance_glm = []
    has_direct_relevance = dataset.has_direct_relevance_scores()
    has_relevance = None
    has_ood_roi = need_ood_roi and has_direct_relevance and bool(dataset.language_event_h5_relevance_ood_events)
    runs = {}

    for local_idx, run_id in enumerate(dataset.subject_ids):
        subj_idx = int(dataset.subject_indices[local_idx])
        base_subject = base_subject_id(run_id)
        assert base_subject in context_map
        ctx_task, ctx_subj_idx, ctx_run_id = context_map[base_subject]
        ctx_dataset = datasets[ctx_task]

        valid_tp = int(dataset.file["valid_timepoints"][subj_idx])
        trunc_tp = valid_tp - (valid_tp % int(generation_frames))
        assert trunc_tp > 0

        ctx_signal = get_zscored_ts(ts_cache, ctx_task, ctx_dataset, ctx_subj_idx)[:, :context_frames]
        target_ts = get_zscored_ts(ts_cache, task, dataset, subj_idx)
        ctx_conds = build_condition_arrays(
            ctx_dataset,
            ctx_subj_idx,
            np.arange(context_frames, dtype=np.int64),
            relevance_subject_id=str(ctx_run_id),
            relevance_source="true_roi",
        )

        run_row_indices = []
        for start in range(0, trunc_tp, int(generation_frames)):
            fut_conds = build_condition_arrays(
                dataset,
                subj_idx,
                np.arange(start, start + generation_frames, dtype=np.int64),
                relevance_subject_id=str(run_id),
                relevance_source="true_roi",
            )
            fut_signal = target_ts[:, start:start + generation_frames]

            signal_glm.append(np.concatenate([ctx_signal, fut_signal], axis=1).T[:, :, None].astype(np.float32))
            future_signal.append(fut_signal.T[:, :, None].astype(np.float32))
            condition_disc_glm.append(np.concatenate([ctx_conds["condition_disc"], fut_conds["condition_disc"]], axis=0))
            condition_disc_weight_glm.append(
                np.concatenate([ctx_conds["condition_disc_weight"], fut_conds["condition_disc_weight"]], axis=0)
            )
            condition_cont_glm.append(np.concatenate([ctx_conds["condition_cont"], fut_conds["condition_cont"]], axis=0))
            condition_mode_glm.append(np.concatenate([ctx_conds["condition_mode"], fut_conds["condition_mode"]], axis=0))
            condition_drop_mask_glm.append(
                np.concatenate(
                    [
                        np.ones(context_frames, dtype=bool),
                        np.zeros(generation_frames, dtype=bool),
                    ],
                    axis=0,
                )
            )

            full_rel_glm = None if ctx_conds["relevance_scores"] is None else np.concatenate(
                [ctx_conds["relevance_scores"], fut_conds["relevance_scores"]],
                axis=0,
            )
            if has_relevance is None:
                has_relevance = full_rel_glm is not None
            if has_relevance:
                assert full_rel_glm is not None
                relevance_scores_glm.append(full_rel_glm)
                drop_mask_relevance_glm.append(
                    np.concatenate(
                        [
                            np.ones(context_frames, dtype=bool),
                            np.zeros(generation_frames, dtype=bool),
                        ],
                        axis=0,
                    )
                )

            if has_ood_roi:
                ood_fut = build_condition_arrays(
                    dataset,
                    subj_idx,
                    np.arange(start, start + generation_frames, dtype=np.int64),
                    relevance_subject_id=str(run_id),
                    relevance_source="ood_roi",
                )
                glm_ctx_rel = np.zeros((context_frames,) + ood_fut["relevance_scores"].shape[1:], dtype=np.float32)
                relevance_scores_ood_roi_glm.append(np.concatenate([glm_ctx_rel, ood_fut["relevance_scores"]], axis=0))

            idx = len(run_ids)
            run_ids.append(str(run_id))
            run_indices.append(int(subj_idx))
            run_row_indices.append(idx)

        runs[str(run_id)] = run_row_indices

    return {
        "task": task,
        "dataset": dataset,
        "signal_glm": np.stack(signal_glm, axis=0).astype(np.float32),
        "future_signal": np.stack(future_signal, axis=0).astype(np.float32),
        "run_ids": np.asarray(run_ids, dtype=object),
        "run_indices": np.asarray(run_indices, dtype=np.int64),
        "condition_disc_glm": np.stack(condition_disc_glm, axis=0).astype(np.int64),
        "condition_disc_weight_glm": np.stack(condition_disc_weight_glm, axis=0).astype(np.float32),
        "condition_cont_glm": np.stack(condition_cont_glm, axis=0).astype(np.float32),
        "condition_mode_glm": np.stack(condition_mode_glm, axis=0).astype(np.int64),
        "condition_drop_mask_glm": np.stack(condition_drop_mask_glm, axis=0),
        "relevance_scores_glm": None if not has_relevance else np.stack(relevance_scores_glm, axis=0).astype(np.float32),
        "relevance_scores_ood_roi_glm": None if not has_ood_roi else np.stack(relevance_scores_ood_roi_glm, axis=0).astype(np.float32),
        "drop_mask_relevance_glm": None if not has_relevance else np.stack(drop_mask_relevance_glm, axis=0),
        "runs": runs,
    }


def _prep_to_cpu(prep):
    out = {}
    for key, value in prep.items():
        out[key] = value.cpu() if torch.is_tensor(value) else value
    return out


def _move_prep_to_device(prep, device):
    out = {}
    for key, value in prep.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def build_bank_prep(bank, protocol, model, tokenizer, device, batch_size, relevance_mode, relevance_input_cfg):
    suffix = "cont" if protocol == "continuation" else "glm"
    signal = bank[f"signal_{suffix}"]
    condition_disc = bank[f"condition_disc_{suffix}"]
    condition_disc_weight = bank[f"condition_disc_weight_{suffix}"]
    condition_cont = bank[f"condition_cont_{suffix}"]
    condition_mode = bank[f"condition_mode_{suffix}"]
    condition_drop_mask = bank.get(f"condition_drop_mask_{suffix}")
    relevance_scores = bank[f"relevance_scores_{suffix}"]
    relevance_scores_ood_roi = bank.get(f"relevance_scores_ood_roi_{suffix}", bank.get(f"relevance_scores_ood_{suffix}"))
    drop_mask_relevance_fixed = bank.get(f"drop_mask_relevance_{suffix}")
    relevance_scores, drop_mask_relevance, relevance_embedding_type = resolve_relevance_input(
        mode=relevance_mode,
        task=bank["task"],
        condition_cont=condition_cont,
        condition_mode=condition_mode,
        relevance_scores=relevance_scores,
        predict_cfg=relevance_input_cfg["predict"],
        num_rois=int(model.simtok_num_rois),
        relevance_scores_ood_roi=relevance_scores_ood_roi,
    )
    if drop_mask_relevance_fixed is not None:
        if drop_mask_relevance is None:
            drop_mask_relevance = drop_mask_relevance_fixed
        else:
            drop_mask_relevance = drop_mask_relevance | drop_mask_relevance_fixed

    prep_batches = []
    total = signal.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = {
            "signal": torch.from_numpy(np.ascontiguousarray(signal[start:end])).contiguous(),
            "condition": torch.from_numpy(np.ascontiguousarray(condition_disc[start:end])).contiguous(),
            "condition_disc": torch.from_numpy(np.ascontiguousarray(condition_disc[start:end])).contiguous(),
            "condition_disc_weight": torch.from_numpy(np.ascontiguousarray(condition_disc_weight[start:end])).contiguous(),
            "condition_cont": torch.from_numpy(np.ascontiguousarray(condition_cont[start:end])).contiguous(),
            "condition_mode": torch.from_numpy(np.ascontiguousarray(condition_mode[start:end])).contiguous(),
        }
        if relevance_scores is not None:
            batch["relevance_scores"] = torch.from_numpy(np.ascontiguousarray(relevance_scores[start:end])).contiguous()
            batch["relevance_embedding_type"] = torch.full(
                (end - start,),
                int(relevance_embedding_type),
                dtype=torch.long,
            )
        if condition_drop_mask is not None:
            batch["drop_mask"] = torch.from_numpy(np.ascontiguousarray(condition_drop_mask[start:end])).contiguous()
        if drop_mask_relevance is not None:
            batch["drop_mask_relevance"] = torch.from_numpy(np.ascontiguousarray(drop_mask_relevance[start:end])).contiguous()
        prep_batches.append(_prep_to_cpu(_batch_to_prep(batch, model, tokenizer, device)))
    return _concat_prep_batches(prep_batches)


def generate_future_windows(
    model,
    tokenizer,
    prep,
    device,
    num_steps,
    guidance_scale,
    seed,
    batch_size,
    condition_guidance_scope,
    progress_label=None,
    context_guidance_scale=1.0,
    drop_latent_context=False,
    stochastic_euler_eps=0.0,
    stochastic_euler_noise_space="latent",
    euler_stop_sigma=1.0,
):
    futures = []
    total = prep["z_context"].shape[0]
    halfway = max(1, total // 2)
    printed_halfway = False
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        prep_chunk = _move_prep_to_device(_slice_prep(prep, start, end), device)
        z_future = _sample_future(
            model,
            prep_chunk,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            context_guidance_scale=context_guidance_scale,
            seed=int(seed + start),
            condition_guidance_scope=condition_guidance_scope,
            drop_latent_context=drop_latent_context,
            stochastic_euler_eps=stochastic_euler_eps,
            stochastic_euler_noise_space=stochastic_euler_noise_space,
            euler_stop_sigma=euler_stop_sigma,
        )
        if str(stochastic_euler_noise_space).lower() == "roi":
            bold = add_stochastic_eval_noise(
                _decode_to_bold(tokenizer, z_future, model=model),
                stochastic_euler_eps,
            ).cpu().numpy().astype(np.float32)
        else:
            bold = _decode_to_bold(tokenizer, z_future, model=model).cpu().numpy().astype(np.float32)
        futures.append(bold)
        if progress_label is not None and not printed_halfway and end >= halfway:
            print(f"[hcp-eval] {progress_label}: 50% ({end}/{total})", flush=True)
            printed_halfway = True
    if progress_label is not None:
        print(f"[hcp-eval] {progress_label}: 100% ({total}/{total})", flush=True)
    return np.concatenate(futures, axis=0)


def generate_rollout_bank(
    model,
    tokenizer,
    prep,
    device,
    count,
    num_steps,
    guidance_scale,
    batch_size,
    progress_label,
    condition_guidance_scope,
    context_guidance_scale=1.0,
    stochastic_euler_eps=0.0,
    stochastic_euler_noise_space="latent",
    euler_stop_sigma=1.0,
):
    rollouts = []
    halfway = max(1, int(count) // 2)
    for rollout_idx in range(int(count)):
        rollouts.append(
            generate_future_windows(
                model,
                tokenizer,
                prep,
                device,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                context_guidance_scale=context_guidance_scale,
                seed=rollout_idx,
                batch_size=batch_size,
                condition_guidance_scope=condition_guidance_scope,
                stochastic_euler_eps=stochastic_euler_eps,
                stochastic_euler_noise_space=stochastic_euler_noise_space,
                euler_stop_sigma=euler_stop_sigma,
            )
        )
        n = rollout_idx + 1
        if n == halfway:
            print(f"[hcp-eval] {progress_label}: 50% ({n}/{count})", flush=True)
    print(f"[hcp-eval] {progress_label}: 100% ({count}/{count})", flush=True)
    return np.stack(rollouts, axis=0).astype(np.float32)
