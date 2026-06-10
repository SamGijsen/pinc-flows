from dataclasses import dataclass
from pathlib import Path

from util.dataset_sampling import build_expanded_val_datasets
from util.dynamics_runtime import resolve_relevance_runtime_config

from .bank import build_hcp_dataset, build_subject_context_map, build_task_bank
from .common import pick
from .glm import HCP_TASKS


@dataclass(frozen=True)
class EvalBanks:
    banks: dict
    ibc_datasets: object
    roi_counts: list


def _resolve_val_dataset_relevance_args(cfg, relevance_input_cfg=None):
    dynamics_cfg = cfg["dynamics"]
    relevance = resolve_relevance_runtime_config(dynamics_cfg)
    unconditioned_pretraining = bool(dynamics_cfg.get("unconditioned_pretraining", False))
    relevance_ood_h5_group = relevance.cfg.get("ood_roi_h5_group")
    if relevance_input_cfg is not None:
        relevance_ood_h5_group = relevance_input_cfg.get("ood_roi_h5_group", relevance_ood_h5_group)

    return {
        "relevance_mode": relevance.mode,
        "relevance_precomputed_path": relevance.precomputed_path,
        "relevance_h5_group": relevance.h5_group,
        "relevance_ood_h5_group": relevance_ood_h5_group,
        "unconditioned_pretraining": unconditioned_pretraining,
    }


def build_ibc_eval_datasets(cfg, relevance_input_cfg):
    dynamics_cfg = cfg["dynamics"]
    data_cfg = cfg["data"]
    tokenizer_cfg = cfg.get("tokenizer", {})
    context_frames = int(dynamics_cfg.get("context_frames", 8))
    generation_frames = int(dynamics_cfg.get("generation_frames", 16))
    build_args = _resolve_val_dataset_relevance_args(cfg, relevance_input_cfg)
    specs_and_datasets = build_expanded_val_datasets(
        data_cfg=data_cfg,
        dynamics_cfg=dynamics_cfg,
        tokenizer_cfg=tokenizer_cfg,
        subject_ids_path=data_cfg.get("val_subject_ids_path"),
        data_paths=data_cfg.get("val_path"),
        default_input_stride=data_cfg.get("input_stride", 1),
        default_anchor_crop_index=context_frames,
        sequence_length=context_frames + generation_frames,
        num_conditions=int(dynamics_cfg.get("num_conditions", 1)),
        subject_token_enabled=bool(dynamics_cfg.get("subject_token_enabled", False)),
        subject_context_length=dynamics_cfg.get("subject_context_length"),
        subject_min_gap=pick(data_cfg, "val", "subject_min_gap", data_cfg.get("subject_min_gap")),
        splice_context_frames=context_frames,
        splice_generation_frames=generation_frames,
        **build_args,
    )
    out = {}
    for spec, dataset in specs_and_datasets:
        if str(spec["name"]).lower() != "ibc":
            continue
        task = Path(str(spec["path"])).stem
        assert task not in out
        out[task] = dataset
    return out


def build_eval_banks(
    cfg,
    holdout_subject_ids_path,
    relevance_input_cfg,
    ts_cache,
    tasks,
    atlas_names,
    context_frames,
    run_ibc_glm,
):
    if tasks:
        datasets = {
            task: build_hcp_dataset(cfg, task, holdout_subject_ids_path, relevance_input_cfg)
            for task in HCP_TASKS
        }
        roi_counts = [datasets[HCP_TASKS[0]].file[f"timeseries/{atlas}"].shape[1] for atlas in atlas_names]
        context_map = build_subject_context_map(datasets, context_frames)
        banks = {
            task: build_task_bank(task, datasets, context_map, cfg, ts_cache, relevance_input_cfg)
            for task in HCP_TASKS
        }
        return EvalBanks(
            banks=banks,
            ibc_datasets=None,
            roi_counts=roi_counts,
        )

    assert run_ibc_glm, "No HCP tasks requested and run_ibc_glm is disabled"
    ibc_datasets = build_ibc_eval_datasets(cfg, relevance_input_cfg)
    assert len(ibc_datasets) > 0, "No IBC datasets found for IBC-only eval"
    first_ibc_dataset = next(iter(ibc_datasets.values()))
    roi_counts = [first_ibc_dataset.file[f"timeseries/{atlas}"].shape[1] for atlas in atlas_names]
    return EvalBanks(
        banks={},
        ibc_datasets=ibc_datasets,
        roi_counts=roi_counts,
    )
