"""Shared dynamics runtime construction.

This module owns config normalization, tokenizer/model construction, and
checkpoint weight loading for both training and HCP/IBC evaluation.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import yaml

from model.dynamics import FMRIDynamics
from util.dynamics_roi_utils import is_raw_roi_encoder_config, normalize_roi_encoder_config
from util.dynamics_tokenizer_utils import PassthroughEncoder
from util.tokenizer_language import (
    collect_condition_cont_language_event_cfgs,
    normalize_roi_embeddings_shared_mean_from_condition_cfgs,
)


@dataclass(frozen=True)
class RelevanceRuntimeConfig:
    cfg: dict
    mode: str
    enabled: bool
    precomputed_path: Optional[str]
    h5_group: Optional[str]
    include_condition_token: bool
    level_type_embedding_enabled: bool


@dataclass(frozen=True)
class DynamicsRuntimeBuild:
    model: torch.nn.Module
    tokenizer: object
    relevance: RelevanceRuntimeConfig
    pretrain_dynamics_enabled: bool
    unconditioned_pretraining: bool
    one_roi_one_token: bool
    roi_encoder: object
    subject_token_enabled: bool
    subject_context_length: object
    subject_allow_missing_token: bool
    subject_min_gap: object
    global_condition_enabled: bool
    condition_cont_layout: str
    num_conditions: int
    context_frames: int
    generation_frames: int
    sequence_length: int


def _log(log_fn, message):
    if log_fn is not None:
        log_fn(message)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def config_uses_pretrain_dynamics(data_cfg):
    keys = ("pretrain_dynamics", "train_pretrain_dynamics", "val_pretrain_dynamics")
    if any(data_cfg.get(key) is not None for key in keys):
        return True
    for split in ("train", "val"):
        specs = data_cfg.get(f"{split}_datasets")
        if not isinstance(specs, (list, tuple)):
            continue
        if any(isinstance(spec, dict) and spec.get("pretrain_dynamics") is not None for spec in specs):
            return True
    return False


def _dataset_specs_has_global_meta(data_cfg, spec_key):
    specs = data_cfg.get(spec_key)
    if not isinstance(specs, (list, tuple)):
        return False
    for item in specs:
        if not isinstance(item, dict):
            continue
        if (
            item.get("age_label_name") is not None
            or item.get("sex_label_name") is not None
            or item.get("motion_label_name") is not None
            or item.get("field_strength_t") is not None
        ):
            return True
    return False


def _resolve_global_condition_enabled(dynamics_cfg, data_cfg):
    global_keys = [
        "age_label_name",
        "sex_label_name",
        "motion_label_name",
        "field_strength_t",
        "train_age_label_name",
        "train_sex_label_name",
        "train_motion_label_name",
        "train_field_strength_t",
        "val_age_label_name",
        "val_sex_label_name",
        "val_motion_label_name",
        "val_field_strength_t",
    ]
    has_global_condition_config = (
        any(data_cfg.get(k) is not None for k in global_keys)
        or _dataset_specs_has_global_meta(data_cfg, "train_datasets")
        or _dataset_specs_has_global_meta(data_cfg, "val_datasets")
    )
    return bool(dynamics_cfg.get("global_condition_enabled", has_global_condition_config))


def resolve_relevance_runtime_config(dynamics_cfg):
    simtok_raw = dynamics_cfg.get("simtok", False)
    simtok_cfg = simtok_raw if isinstance(simtok_raw, dict) else {}
    relevance_raw = dynamics_cfg.get("relevance")

    if relevance_raw is None:
        relevance_cfg = {
            "mode": "fixed" if bool(simtok_cfg.get("enabled", simtok_raw)) else "none",
            "projection": "truncate",
        }
        roi_paths_cfg = simtok_cfg.get("roi_embeddings_path")
        if roi_paths_cfg is not None:
            relevance_cfg["roi_embeddings_path"] = roi_paths_cfg
    elif isinstance(relevance_raw, dict):
        relevance_cfg = dict(relevance_raw)
        for legacy_key in ("v2_empirical", "training"):
            legacy_cfg = relevance_cfg.get(legacy_key)
            if not isinstance(legacy_cfg, dict):
                continue
            for key in ("h5_group", "ood_roi_h5_group", "ood_network_h5_group"):
                if key in legacy_cfg and key not in relevance_cfg:
                    relevance_cfg[key] = legacy_cfg[key]
    elif relevance_raw in (False, None):
        relevance_cfg = {"mode": "none"}
    elif isinstance(relevance_raw, str):
        relevance_cfg = {"mode": relevance_raw}
    else:
        raise TypeError(
            "dynamics.relevance must be a dict, mode string, or false/null, "
            f"got {type(relevance_raw).__name__}"
        )

    if "mode" not in relevance_cfg and "enabled" in relevance_cfg:
        relevance_cfg["mode"] = "fixed" if bool(relevance_cfg["enabled"]) else "none"
    relevance_mode = str(relevance_cfg.get("mode", "none")).lower()
    if relevance_mode not in ("none", "fixed"):
        raise ValueError(
            f"Unsupported dynamics.relevance.mode={relevance_mode!r}. "
            "Expected one of: none, fixed."
        )

    relevance_model = str(relevance_cfg.get("model", "standard")).lower()
    if relevance_model != "standard":
        raise ValueError("public paper release supports dynamics.relevance.model='standard' only")

    relevance_precomputed_cfg = relevance_cfg.get("use_precomputed_relevance")
    if relevance_precomputed_cfg in (None, False):
        relevance_precomputed_path = None
    elif isinstance(relevance_precomputed_cfg, str):
        if len(relevance_precomputed_cfg.strip()) == 0:
            raise ValueError("dynamics.relevance.use_precomputed_relevance cannot be an empty string")
        relevance_precomputed_path = relevance_precomputed_cfg
    elif isinstance(relevance_precomputed_cfg, bool):
        raise TypeError(
            "dynamics.relevance.use_precomputed_relevance must be a .npy path string "
            "(or null/false), not a boolean true"
        )
    else:
        raise TypeError(
            "dynamics.relevance.use_precomputed_relevance must be a .npy path string "
            f"(or null/false), got {type(relevance_precomputed_cfg).__name__}"
        )
    if relevance_precomputed_path is not None and relevance_mode != "fixed":
        raise ValueError(
            "dynamics.relevance.use_precomputed_relevance requires dynamics.relevance.mode='fixed'"
        )

    relevance_h5_group = relevance_cfg.get("h5_group")
    if relevance_h5_group is not None:
        if not isinstance(relevance_h5_group, str) or len(relevance_h5_group.strip()) == 0:
            raise ValueError("dynamics.relevance.h5_group must be a non-empty string")
        if relevance_mode != "fixed":
            raise ValueError("dynamics.relevance.h5_group requires dynamics.relevance.mode='fixed'")
        if relevance_precomputed_path is not None:
            raise ValueError(
                "dynamics.relevance.h5_group does not support dynamics.relevance.use_precomputed_relevance"
            )

    return RelevanceRuntimeConfig(
        cfg=relevance_cfg,
        mode=relevance_mode,
        enabled=relevance_mode != "none",
        precomputed_path=relevance_precomputed_path,
        h5_group=relevance_h5_group,
        include_condition_token=bool(relevance_cfg.get("include_condition_token", True)),
        level_type_embedding_enabled=bool(relevance_cfg.get("level_type_embedding", True)),
    )


def build_tokenizer(cfg, device, log_fn: Optional[Callable[[str], None]] = None):
    data_cfg = cfg["data"]
    dyn_cfg = cfg["dynamics"]
    tok_cfg = cfg.get("tokenizer", {})
    roi_encoder = normalize_roi_encoder_config(
        dynamics_cfg=dyn_cfg,
        tokenizer_cfg=tok_cfg,
        data_cfg=data_cfg,
    )

    if is_raw_roi_encoder_config(roi_encoder):
        return None
    if bool(dyn_cfg.get("one_roi_one_token", False)):
        crop_length = tok_cfg.get("input_timesteps", data_cfg.get("patch_size", 1))
        if int(crop_length) != 1:
            raise ValueError(f"one_roi_one_token requires tokenizer.input_timesteps=1, got {crop_length}")
        return None

    num_rois = data_cfg.get("num_rois", 450)
    crop_length = tok_cfg.get("input_timesteps", data_cfg.get("patch_size", 1))
    num_latents = dyn_cfg.get("num_latents", 1)
    tok = PassthroughEncoder(
        num_rois=num_rois,
        crop_length=crop_length,
        num_latents=num_latents,
    ).to(device)
    tok.eval()
    _log(log_fn, f"Using PassthroughEncoder: {num_rois}x{crop_length} -> {num_latents}x{tok.latent_dim}")
    return tok


def _load_roi_rows(roi_paths_cfg):
    roi_paths = [roi_paths_cfg] if isinstance(roi_paths_cfg, str) else list(roi_paths_cfg)
    return np.concatenate([np.asarray(np.load(path), dtype=np.float32) for path in roi_paths], axis=0)


def _first_shared_mean_roi_paths(relevance_cfg, language_event_cfgs):
    roi_paths_cfg = relevance_cfg.get("roi_embeddings_path")
    if roi_paths_cfg is not None:
        return roi_paths_cfg
    for language_event_cfg in language_event_cfgs:
        norm_cfg = language_event_cfg.get("normalization", {})
        if norm_cfg.get("mode") == "shared_mean_l2":
            roi_paths_cfg = norm_cfg.get("roi_embeddings_path")
            if roi_paths_cfg is not None:
                return roi_paths_cfg
    return None


def build_dynamics(
    cfg,
    tokenizer,
    device,
    *,
    eval_mode=True,
    log_fn: Optional[Callable[[str], None]] = None,
):
    dynamics_cfg = cfg["dynamics"]
    data_cfg = cfg["data"]
    tokenizer_cfg = cfg.get("tokenizer", {})

    pretrain_dynamics_enabled = config_uses_pretrain_dynamics(data_cfg)
    unconditioned_pretraining = bool(dynamics_cfg.get("unconditioned_pretraining", False))
    relevance = resolve_relevance_runtime_config(dynamics_cfg)

    if pretrain_dynamics_enabled:
        crop_length = tokenizer_cfg.get("input_timesteps", data_cfg.get("patch_size", 1))
        if int(crop_length) != 1:
            raise ValueError(f"pretrain_dynamics requires tokenizer.input_timesteps=1, got {crop_length}")
        if relevance.precomputed_path is not None:
            raise ValueError(
                "pretrain_dynamics does not support dynamics.relevance.use_precomputed_relevance; "
                "relevance scores come directly from the dataset"
            )
    if unconditioned_pretraining and pretrain_dynamics_enabled:
        raise ValueError("dynamics.unconditioned_pretraining does not support pretrain_dynamics")

    one_roi_one_token = bool(dynamics_cfg.get("one_roi_one_token", False))
    roi_encoder = normalize_roi_encoder_config(
        dynamics_cfg=dynamics_cfg,
        tokenizer_cfg=tokenizer_cfg,
        data_cfg=data_cfg,
    )
    if roi_encoder is not None and one_roi_one_token:
        raise ValueError("dynamics.roi_encoder and dynamics.one_roi_one_token cannot both be enabled")

    subject_token_enabled = bool(dynamics_cfg.get("subject_token_enabled", False))
    subject_context_length = dynamics_cfg.get("subject_context_length")
    subject_allow_missing_token = bool(dynamics_cfg.get("subject_allow_missing_token", False))
    subject_min_gap = data_cfg.get("subject_min_gap")
    if subject_token_enabled:
        if subject_context_length is None or int(subject_context_length) <= 0:
            raise ValueError("dynamics.subject_context_length must be > 0 when subject_token_enabled=True")
        if subject_min_gap is None or int(subject_min_gap) < 0:
            raise ValueError("data.subject_min_gap must be >= 0 when subject_token_enabled=True")

    roi_simtok_mixer = bool(dynamics_cfg.get("roi_simtok_mixer", False))
    roi_language_embeddings = None
    relevance_roi_embeddings = None
    simtok_num_rois = None

    language_event_cfgs = collect_condition_cont_language_event_cfgs(data_cfg)
    condition_cont_layout = "standard"
    if language_event_cfgs:
        condition_cont_layout = "h5_v2"
        if dynamics_cfg.get("condition_cont_dim") is None:
            dynamics_cfg["condition_cont_dim"] = 3074

    if one_roi_one_token:
        crop_length = tokenizer_cfg.get("input_timesteps", data_cfg.get("patch_size", 1))
        if int(crop_length) != 1:
            raise ValueError(
                f"dynamics.one_roi_one_token=True requires tokenizer.input_timesteps=1, got {crop_length}"
            )
        roi_rows = _load_roi_rows(dynamics_cfg["roi_language_embeddings_path"])
        num_rois = int(data_cfg.get("num_rois", roi_rows.shape[0]))
        if roi_rows.shape[0] != num_rois:
            raise ValueError(
                f"roi_language_embeddings rows must match data.num_rois={num_rois}, got {roi_rows.shape[0]}"
            )
        target_dim = int(dynamics_cfg.get("d_model", dynamics_cfg.get("latent_dim", 1)))
        roi_language_embeddings = normalize_roi_embeddings_shared_mean_from_condition_cfgs(
            roi_rows,
            model_embedding_dim=target_dim,
            condition_cont_language_event_cfgs=language_event_cfgs,
        )
        dyn_num_latents = num_rois
        dyn_latent_dim = 1
        dynamics_cfg["num_latents"] = dyn_num_latents
        dynamics_cfg["latent_dim"] = dyn_latent_dim
        _log(
            log_fn,
            "Using one_roi_one_token mode: "
            f"num_rois={num_rois}, roi_language_embeddings={tuple(roi_language_embeddings.shape)}",
        )
    elif is_raw_roi_encoder_config(roi_encoder):
        dyn_num_latents = int(roi_encoder["num_tokens"])
        dyn_latent_dim = int(roi_encoder["input_dim"])
        dynamics_cfg["num_latents"] = dyn_num_latents
        dynamics_cfg["latent_dim"] = dyn_latent_dim
        dynamics_cfg["d_model"] = roi_encoder["emb_dim"]
        _log(
            log_fn,
            "Using ROI encoder: "
            f"mode={roi_encoder['mode']}, num_tokens={roi_encoder['num_tokens']}, "
            f"input_dim={roi_encoder['input_dim']}, emb_dim={roi_encoder['emb_dim']}",
        )
    else:
        tok_latent_dim = int(tokenizer.latent_dim)
        cfg_latent_dim = int(dynamics_cfg.get("latent_dim", tok_latent_dim))
        if cfg_latent_dim != tok_latent_dim:
            raise ValueError(f"Dynamics latent_dim={cfg_latent_dim} != tokenizer latent dim={tok_latent_dim}")
        tok_num_latents = int(tokenizer.num_latents)
        cfg_num_latents = int(dynamics_cfg.get("num_latents", 1))
        if tok_num_latents != cfg_num_latents:
            raise ValueError(f"Dynamics num_latents={cfg_num_latents} != tokenizer num_latents={tok_num_latents}")
        dyn_num_latents = cfg_num_latents
        dyn_latent_dim = tok_latent_dim
        dynamics_cfg["latent_dim"] = tok_latent_dim

    global_condition_enabled = _resolve_global_condition_enabled(dynamics_cfg, data_cfg)

    if relevance.enabled:
        if one_roi_one_token:
            raise ValueError("relevance is not supported with one_roi_one_token mode")
        if is_raw_roi_encoder_config(roi_encoder):
            if int(dyn_num_latents) != int(roi_encoder["num_tokens"]):
                raise ValueError("relevance requires roi_encoder-derived num_latents")
        elif int(dynamics_cfg.get("num_latents", 1)) != 1:
            raise ValueError("relevance requires dynamics.num_latents=1")

        simtok_num_rois = int(data_cfg.get("num_rois"))
        if relevance.h5_group is not None:
            relevance_roi_embeddings = None
            _log(
                log_fn,
                "Using H5-backed relevance scoring: "
                f"mode={relevance.mode}, h5_group={relevance.h5_group}, "
                f"num_rois={simtok_num_rois}, include_condition_token={relevance.include_condition_token}",
            )
        elif relevance.precomputed_path is not None:
            _log(
                log_fn,
                "Using precomputed relevance scoring: "
                f"mode={relevance.mode}, path={relevance.precomputed_path}, num_rois={simtok_num_rois}, "
                f"include_condition_token={relevance.include_condition_token}",
            )
        else:
            roi_paths_cfg = _first_shared_mean_roi_paths(relevance.cfg, language_event_cfgs)
            if (
                pretrain_dynamics_enabled
                and relevance.mode == "fixed"
                and roi_paths_cfg is None
                and len(language_event_cfgs) == 0
            ):
                relevance_roi_embeddings = None
                simtok_num_rois = int(data_cfg.get("num_rois"))
                _log(
                    log_fn,
                    "Using dataset-provided relevance priors: "
                    f"mode={relevance.mode}, num_rois={simtok_num_rois}, "
                    f"include_condition_token={relevance.include_condition_token}",
                )
            else:
                if roi_paths_cfg is None:
                    raise ValueError("relevance enabled but no roi_embeddings_path found")
                condition_cont_dim = int(dynamics_cfg.get("condition_cont_dim", data_cfg.get("condition_cont_dim")))
                roi_rows = _load_roi_rows(roi_paths_cfg)
                relevance_roi_embeddings = normalize_roi_embeddings_shared_mean_from_condition_cfgs(
                    roi_rows,
                    model_embedding_dim=condition_cont_dim,
                    condition_cont_language_event_cfgs=language_event_cfgs,
                )
                simtok_num_rois = int(relevance_roi_embeddings.shape[0])
                _log(
                    log_fn,
                    "Using relevance scoring: "
                    f"mode={relevance.mode}, include_condition_token={relevance.include_condition_token}, "
                    f"roi_embeddings={tuple(relevance_roi_embeddings.shape)}, condition_cont_dim={condition_cont_dim}",
                )

    num_conditions = int(dynamics_cfg.get("num_conditions", 3))
    model_kwargs = dict(
        num_latents=dyn_num_latents,
        latent_dim=dyn_latent_dim,
        d_model=dynamics_cfg.get("d_model", None),
        num_heads=dynamics_cfg.get("num_heads", 8),
        num_layers=dynamics_cfg.get("num_layers", 6),
        num_conditions=num_conditions,
        condition_cont_dim=dynamics_cfg.get("condition_cont_dim"),
        condition_cont_layout=condition_cont_layout,
        condition_cont_use_projection=dynamics_cfg.get("condition_cont_use_projection", True),
        condition_cont_proj_bias=dynamics_cfg.get("condition_cont_proj_bias", True),
        mlp_ratio=dynamics_cfg.get("mlp_ratio", 4.0),
        dropout=dynamics_cfg.get("dropout", 0.0),
        max_context_length=dynamics_cfg.get("max_context_length", 16),
        p_drop_context=dynamics_cfg.get("p_drop_context", 0.0),
        p_drop_condition_and_context=dynamics_cfg.get("p_drop_condition_and_context", 0.0),
        p_drop_condition_context=dynamics_cfg.get("p_drop_condition_context", 0.0),
        p_drop_condition_future=dynamics_cfg.get("p_drop_condition_future", 0.0),
        p_drop_global_condition=dynamics_cfg.get("p_drop_global_condition", 0.0),
        p_drop_instruct=dynamics_cfg.get("p_drop_instruct", 0.0),
        p_drop_sensory=dynamics_cfg.get("p_drop_sensory", 0.0),
        p_drop_response=dynamics_cfg.get("p_drop_response", 0.0),
        p_drop_relevance=dynamics_cfg.get("p_drop_relevance", 0.0),
        p_drop_text_condition_token=dynamics_cfg.get("p_drop_text_condition_token", 0.0),
        ramp_loss_weight=dynamics_cfg.get("ramp_loss_weight", False),
        ar_training_denoise_steps=dynamics_cfg.get("ar_training_denoise_steps", 2),
        context_sigma=dynamics_cfg.get("context_sigma", 1.0),
        soft_cap=dynamics_cfg.get("soft_cap", 30.0),
        num_registers=dynamics_cfg.get("num_registers", 0),
        register_temporal_embed=dynamics_cfg.get("register_temporal_embed", False),
        prediction_type=dynamics_cfg.get("prediction_type", "x"),
        parallel_shared_sigma=dynamics_cfg.get("parallel_shared_sigma", False),
        context_frames=dynamics_cfg.get("context_frames", 8),
        generation_frames=dynamics_cfg.get("generation_frames", 16),
        subject_token_enabled=subject_token_enabled,
        subject_context_length=subject_context_length,
        subject_encoder_layers=dynamics_cfg.get("subject_encoder_layers", 2),
        subject_allow_missing_token=subject_allow_missing_token,
        global_condition_enabled=global_condition_enabled,
        one_roi_one_token=one_roi_one_token,
        roi_language_embeddings=roi_language_embeddings,
        simtok_enabled=relevance.enabled,
        simtok_roi_embeddings=relevance_roi_embeddings,
        roi_simtok_mixer=roi_simtok_mixer,
        simtok_num_rois=simtok_num_rois,
        unconditioned_pretraining=unconditioned_pretraining,
        relevance_mode=relevance.mode,
        relevance_include_condition_token=relevance.include_condition_token,
        relevance_level_type_embedding_enabled=relevance.level_type_embedding_enabled,
        factorized_attention=dynamics_cfg.get("factorized_attention", None),
    )
    if is_raw_roi_encoder_config(roi_encoder):
        model_kwargs["roi_encoder"] = roi_encoder

    model = FMRIDynamics(**model_kwargs).to(device)
    if eval_mode:
        model.eval()

    context_frames = int(dynamics_cfg.get("context_frames", 8))
    generation_frames = int(dynamics_cfg.get("generation_frames", 16))
    return DynamicsRuntimeBuild(
        model=model,
        tokenizer=tokenizer,
        relevance=relevance,
        pretrain_dynamics_enabled=pretrain_dynamics_enabled,
        unconditioned_pretraining=unconditioned_pretraining,
        one_roi_one_token=one_roi_one_token,
        roi_encoder=roi_encoder,
        subject_token_enabled=subject_token_enabled,
        subject_context_length=subject_context_length,
        subject_allow_missing_token=subject_allow_missing_token,
        subject_min_gap=subject_min_gap,
        global_condition_enabled=global_condition_enabled,
        condition_cont_layout=condition_cont_layout,
        num_conditions=num_conditions,
        context_frames=context_frames,
        generation_frames=generation_frames,
        sequence_length=context_frames + generation_frames,
    )


def build_dynamics_runtime(
    cfg,
    device,
    *,
    eval_mode=True,
    log_fn: Optional[Callable[[str], None]] = None,
):
    tokenizer = build_tokenizer(cfg, device, log_fn=log_fn)
    return build_dynamics(cfg, tokenizer, device, eval_mode=eval_mode, log_fn=log_fn)


def load_checkpoint_weights(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = FMRIDynamics.convert_legacy_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"relevance_type_embedding.weight"}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"Unexpected checkpoint mismatch. missing={missing}, unexpected={unexpected}")
    return checkpoint
