"""Shared dynamics inference helpers.

This module holds the generic model-building, checkpoint-loading, batching,
sampling, and decoding code used by HCP/IBC evaluation.
"""

import math
import random

import numpy as np
import torch
from torch.amp import autocast

from util.dynamics_roi_utils import (
    decode_roi_tokens,
    is_raw_roi_encoder_config,
)
from util.dynamics_runtime import (
    build_dynamics as build_dynamics_runtime_from_tokenizer,
    build_tokenizer,
    load_checkpoint_weights,
    load_config,
)
from util.dynamics_training_utils import encode_batch_signal, move_batch_tensors


DECODE_STEPS = 1
DECODE_BATCH = 1024


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flatten_time_coordinates(x):
    bsz = x.shape[0]
    seq_len = x.shape[1]
    coord_shape = x.shape[2:]
    num_coords = int(np.prod(coord_shape))
    return bsz, seq_len, coord_shape, num_coords


def _ar1_noise_like(x):
    bsz, seq_len, coord_shape, num_coords = _flatten_time_coordinates(x)
    rho = 0.7
    white = torch.randn(bsz, seq_len, num_coords, device=x.device, dtype=torch.float32)
    noise = torch.empty_like(white)
    innovation_scale = math.sqrt(1.0 - rho * rho)
    noise[:, 0] = white[:, 0]
    for t in range(1, seq_len):
        noise[:, t] = rho * noise[:, t - 1] + innovation_scale * white[:, t]
    noise = noise.view(bsz, seq_len, *coord_shape)
    noise = noise - noise.mean(dim=1, keepdim=True)
    noise = noise / noise.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return noise.to(dtype=x.dtype)


def add_stochastic_eval_noise(x, eps):
    eps = float(eps)
    if eps <= 0.0:
        return x
    return x + eps * _ar1_noise_like(x)


_load_config = load_config
_build_tokenizer = build_tokenizer
_load_checkpoint_weights = load_checkpoint_weights


def _build_dynamics(cfg, tokenizer, device):
    return build_dynamics_runtime_from_tokenizer(cfg, tokenizer, device, eval_mode=True).model


def _batch_to_prep(batch, model, tokenizer, device):
    one_roi_one_token = bool(getattr(model, "one_roi_one_token", False))
    roi_encoder = getattr(model, "roi_encoder", None)
    z = encode_batch_signal(
        batch,
        "signal",
        tokenizer,
        device,
        one_roi_one_token=one_roi_one_token,
        roi_encoder=roi_encoder,
    )

    cond_disc_raw = batch.get("condition_disc", batch.get("condition"))
    if cond_disc_raw is None:
        raise KeyError("Batch is missing 'condition_disc' (or legacy 'condition')")
    cond_disc = cond_disc_raw.to(device, non_blocking=True)
    fields = move_batch_tensors(
        batch,
        (
            "condition_disc_weight",
            "condition_cont",
            "condition_mode",
            "relevance_scores",
            "relevance_embedding_type",
            "drop_mask",
            "drop_mask_relevance",
            "age",
            "sex",
            "motion",
            "field_strength",
        ),
        device,
    )
    cond_disc_weight = fields["condition_disc_weight"]
    cond_cont = fields["condition_cont"]
    cond_mode = fields["condition_mode"]
    relevance_scores = fields["relevance_scores"]
    relevance_embedding_type = fields["relevance_embedding_type"]
    drop_mask = fields["drop_mask"]
    drop_mask_relevance = fields["drop_mask_relevance"]

    t_ctx = int(model.context_frames)
    t_fut = int(model.generation_frames)
    out = {
        "z_context": z[:, :t_ctx],
        "z_future": z[:, t_ctx:t_ctx + t_fut],
        "cond_disc_ctx": cond_disc[:, :t_ctx],
        "cond_disc_fut": cond_disc[:, t_ctx:t_ctx + t_fut],
        "cond_disc_weight_ctx": cond_disc_weight[:, :t_ctx] if cond_disc_weight is not None else None,
        "cond_disc_weight_fut": cond_disc_weight[:, t_ctx:t_ctx + t_fut] if cond_disc_weight is not None else None,
        "cond_cont_ctx": cond_cont[:, :t_ctx] if cond_cont is not None else None,
        "cond_cont_fut": cond_cont[:, t_ctx:t_ctx + t_fut] if cond_cont is not None else None,
        "cond_mode_ctx": cond_mode[:, :t_ctx] if cond_mode is not None else None,
        "cond_mode_fut": cond_mode[:, t_ctx:t_ctx + t_fut] if cond_mode is not None else None,
        "drop_mask_ctx": drop_mask[:, :t_ctx] if drop_mask is not None else None,
        "drop_mask_fut": drop_mask[:, t_ctx:t_ctx + t_fut] if drop_mask is not None else None,
        "relevance_scores_ctx": relevance_scores[:, :t_ctx] if relevance_scores is not None else None,
        "relevance_scores_fut": relevance_scores[:, t_ctx:t_ctx + t_fut] if relevance_scores is not None else None,
        "relevance_embedding_type": relevance_embedding_type,
        "drop_mask_relevance_ctx": drop_mask_relevance[:, :t_ctx] if drop_mask_relevance is not None else None,
        "drop_mask_relevance_fut": drop_mask_relevance[:, t_ctx:t_ctx + t_fut] if drop_mask_relevance is not None else None,
        "age": fields["age"],
        "sex": fields["sex"],
        "motion": fields["motion"],
        "field_strength": fields["field_strength"],
        "subject_token": None,
    }

    if model.subject_token_enabled:
        z_subj = encode_batch_signal(
            batch,
            "distant_signal",
            tokenizer,
            device,
            one_roi_one_token=one_roi_one_token,
            roi_encoder=roi_encoder,
        )
        out["subject_token"] = model.build_subject_token(z_subj)

    return out


def _concat_prep_batches(prep_batches):
    out = {}
    for key in prep_batches[0].keys():
        value = prep_batches[0][key]
        if torch.is_tensor(value):
            out[key] = torch.cat([batch[key] for batch in prep_batches], dim=0)
        else:
            out[key] = value
    return out


def _slice_prep(prep, start_idx, end_idx):
    out = {}
    for key, value in prep.items():
        if torch.is_tensor(value):
            out[key] = value[start_idx:end_idx]
        else:
            out[key] = value
    return out


@torch.no_grad()
def _sample_parallel_with_disc_weight(
    model,
    z_context,
    cond_context,
    cond_future,
    cond_disc_weight_context,
    cond_disc_weight_future,
    num_steps,
    guidance_scale=1.0,
    cond_cont_context=None,
    cond_cont_future=None,
    cond_mode_context=None,
    cond_mode_future=None,
    drop_mask_context=None,
    drop_mask_future=None,
    relevance_scores_context=None,
    relevance_scores_future=None,
    relevance_embedding_type=None,
    drop_mask_relevance_context=None,
    drop_mask_relevance_future=None,
    subject_token=None,
    age=None,
    sex=None,
    motion=None,
    field_strength=None,
    *,
    condition_guidance_scope,
    context_guidance_scale=1.0,
    drop_latent_context=False,
    stochastic_euler_eps=0.0,
    stochastic_euler_noise_space="latent",
    euler_stop_sigma=1.0,
):
    if cond_disc_weight_context is None or cond_disc_weight_future is None:
        raise ValueError("condition_disc_weight is required for training-parity conditioning")
    assert str(stochastic_euler_noise_space).lower() in ("latent", "roi")

    bsz, ctx_len, num_latents, latent_dim = z_context.shape
    gen_len = cond_future.shape[1]
    total_len = ctx_len + gen_len
    device = z_context.device

    use_future_cfg = guidance_scale != 1.0
    use_context_cfg = context_guidance_scale != 1.0
    use_joint_cfg = (
        use_future_cfg
        and use_context_cfg
        and float(guidance_scale) == float(context_guidance_scale)
    )
    use_task_uncond = use_future_cfg or use_context_cfg
    if use_task_uncond and not hasattr(model, "condition_mask_token"):
        raise ValueError(
            "guidance needs condition_mask_token "
            "(condition-token dropout was not enabled during training)"
        )
    if use_context_cfg and not hasattr(model, "context_mask_token"):
        raise ValueError(
            "context_guidance_scale != 1.0 but model has no context_mask_token "
            "(p_drop_context=0 during training)"
        )
    if drop_latent_context and not hasattr(model, "context_mask_token"):
        raise ValueError("drop_latent_context requires context_mask_token")

    z_future = torch.randn(bsz, gen_len, num_latents, latent_dim, device=device)
    dt = 1.0 / num_steps
    euler_stop_sigma = float(euler_stop_sigma)
    assert 0.0 < euler_stop_sigma <= 1.0
    euler_steps = int(math.ceil(euler_stop_sigma * num_steps))

    block_mask = None
    task_block_masks = None
    if model.factorized_attention_enabled:
        task_block_masks = model.get_factorized_task_masks(total_len, device)
    else:
        block_mask = model.get_block_mask(
            total_len,
            device,
            num_prefix_tokens=model.num_prefix_tokens_task,
        )

    if use_task_uncond:
        if condition_guidance_scope == "future_timepoints":
            drop_mask_uncond = torch.zeros((bsz, total_len), device=device, dtype=torch.bool)
            drop_mask_uncond[:, ctx_len:] = True
        elif condition_guidance_scope == "all_timepoints":
            drop_mask_uncond = torch.ones((bsz, total_len), device=device, dtype=torch.bool)
        else:
            raise ValueError(f"Unknown condition_guidance_scope={condition_guidance_scope!r}")
    if use_context_cfg:
        drop_mask_context_uncond = torch.ones(bsz, device=device, dtype=torch.bool)
    drop_mask_context_base = torch.ones(bsz, device=device, dtype=torch.bool) if bool(drop_latent_context) else None

    for step in range(euler_steps):
        sigma_val = step / num_steps
        step_dt = min(dt, euler_stop_sigma - sigma_val)

        sigma_ctx = torch.ones(bsz, ctx_len, device=device)
        sigma_gen = torch.full((bsz, gen_len), sigma_val, device=device)
        sigma = torch.cat([sigma_ctx, sigma_gen], dim=1)

        z_full = torch.cat([z_context, z_future], dim=1)
        cond_full = torch.cat([cond_context, cond_future], dim=1)
        cond_weight_full = torch.cat([cond_disc_weight_context, cond_disc_weight_future], dim=1)
        cond_full_cont = (
            torch.cat([cond_cont_context, cond_cont_future], dim=1)
            if cond_cont_context is not None and cond_cont_future is not None
            else None
        )
        cond_full_mode = (
            torch.cat([cond_mode_context, cond_mode_future], dim=1)
            if cond_mode_context is not None and cond_mode_future is not None
            else None
        )
        cond_drop_full = (
            torch.cat([drop_mask_context, drop_mask_future], dim=1)
            if drop_mask_context is not None and drop_mask_future is not None
            else None
        )
        rel_full = (
            torch.cat([relevance_scores_context, relevance_scores_future], dim=1)
            if relevance_scores_context is not None and relevance_scores_future is not None
            else None
        )
        rel_drop_full = (
            torch.cat([drop_mask_relevance_context, drop_mask_relevance_future], dim=1)
            if drop_mask_relevance_context is not None and drop_mask_relevance_future is not None
            else None
        )

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            pred_cond = model.forward(
                z_full,
                sigma,
                condition_disc=cond_full,
                condition_disc_weight=cond_weight_full,
                condition_cont=cond_full_cont,
                condition_mode=cond_full_mode,
                drop_mask=cond_drop_full,
                relevance_scores=rel_full,
                relevance_embedding_type=relevance_embedding_type,
                drop_mask_relevance=rel_drop_full,
                subject_token=subject_token,
                age=age,
                sex=sex,
                motion=motion,
                field_strength=field_strength,
                drop_mask_context=drop_mask_context_base,
                block_mask=block_mask,
                task_block_masks=task_block_masks,
            )
            pred_cond_future = pred_cond[:, ctx_len:]

            if use_task_uncond and not use_joint_cfg:
                if cond_drop_full is None:
                    curr_drop_mask = drop_mask_uncond
                elif drop_mask_uncond.ndim == 1:
                    curr_drop_mask = cond_drop_full | drop_mask_uncond[:, None]
                else:
                    curr_drop_mask = cond_drop_full | drop_mask_uncond
                curr_drop_relevance = drop_mask_uncond if rel_drop_full is None else rel_drop_full | drop_mask_uncond
                pred_task_uncond = model.forward(
                    z_full,
                    sigma,
                    condition_disc=cond_full,
                    condition_disc_weight=cond_weight_full,
                    condition_cont=cond_full_cont,
                    condition_mode=cond_full_mode,
                    drop_mask=curr_drop_mask,
                    relevance_scores=rel_full,
                    relevance_embedding_type=relevance_embedding_type,
                    drop_mask_relevance=curr_drop_relevance,
                    subject_token=subject_token,
                    age=age,
                    sex=sex,
                    motion=motion,
                    field_strength=field_strength,
                    drop_mask_context=drop_mask_context_base,
                    block_mask=block_mask,
                    task_block_masks=task_block_masks,
                )
                pred_task_uncond = pred_task_uncond[:, ctx_len:]
            if use_context_cfg:
                if use_joint_cfg:
                    if cond_drop_full is None:
                        curr_drop_mask = drop_mask_uncond
                    elif drop_mask_uncond.ndim == 1:
                        curr_drop_mask = cond_drop_full | drop_mask_uncond[:, None]
                    else:
                        curr_drop_mask = cond_drop_full | drop_mask_uncond
                    curr_drop_relevance = drop_mask_uncond if rel_drop_full is None else rel_drop_full | drop_mask_uncond
                pred_all_uncond = model.forward(
                    z_full,
                    sigma,
                    condition_disc=cond_full,
                    condition_disc_weight=cond_weight_full,
                    condition_cont=cond_full_cont,
                    condition_mode=cond_full_mode,
                    relevance_scores=rel_full,
                    relevance_embedding_type=relevance_embedding_type,
                    drop_mask_relevance=curr_drop_relevance,
                    subject_token=subject_token,
                    age=age,
                    sex=sex,
                    motion=motion,
                    field_strength=field_strength,
                    drop_mask=curr_drop_mask,
                    drop_mask_context=drop_mask_context_uncond,
                    block_mask=block_mask,
                    task_block_masks=task_block_masks,
                )
                pred_all_uncond = pred_all_uncond[:, ctx_len:]

        if sigma_val < 1.0 - 1e-6:
            if model.prediction_type == "v":
                velocity_cond = pred_cond_future
                if use_task_uncond and not use_joint_cfg:
                    velocity_task_uncond = pred_task_uncond
                if use_context_cfg:
                    velocity_all_uncond = pred_all_uncond
            else:
                velocity_cond = (pred_cond_future - z_future) / (1 - sigma_val)
                if use_task_uncond and not use_joint_cfg:
                    velocity_task_uncond = (pred_task_uncond - z_future) / (1 - sigma_val)
                if use_context_cfg:
                    velocity_all_uncond = (pred_all_uncond - z_future) / (1 - sigma_val)

            velocity = velocity_cond
            if use_joint_cfg:
                velocity = velocity + (guidance_scale - 1.0) * (velocity_cond - velocity_all_uncond)
            elif use_context_cfg:
                velocity = velocity + (context_guidance_scale - 1.0) * (velocity_task_uncond - velocity_all_uncond)
            if use_future_cfg and not use_joint_cfg:
                velocity = velocity + (guidance_scale - 1.0) * (velocity_cond - velocity_task_uncond)

            z_future = z_future + velocity * step_dt
            if (
                stochastic_euler_eps > 0.0
                and str(stochastic_euler_noise_space).lower() == "latent"
                and step < euler_steps - 1
            ):
                noise_level = 1.0 - sigma_val
                z_future = z_future + (
                    stochastic_euler_eps
                    * math.sqrt(step_dt)
                    * noise_level
                    * _ar1_noise_like(z_future)
                )

    return z_future


def _sample_future(
    model,
    prep,
    num_steps=64,
    guidance_scale=2.0,
    seed=None,
    cond_disc_future=None,
    cond_disc_weight_future=None,
    cond_cont_future=None,
    cond_mode_future=None,
    drop_mask_future=None,
    relevance_scores_future=None,
    drop_mask_relevance_future=None,
    condition_guidance_scope="future_timepoints",
    context_guidance_scale=1.0,
    drop_latent_context=False,
    stochastic_euler_eps=0.0,
    stochastic_euler_noise_space="latent",
    euler_stop_sigma=1.0,
):
    if seed is not None:
        _set_seed(seed)

    return _sample_parallel_with_disc_weight(
        model=model,
        z_context=prep["z_context"],
        cond_context=prep["cond_disc_ctx"],
        cond_future=prep["cond_disc_fut"] if cond_disc_future is None else cond_disc_future,
        cond_disc_weight_context=prep["cond_disc_weight_ctx"],
        cond_disc_weight_future=prep["cond_disc_weight_fut"] if cond_disc_weight_future is None else cond_disc_weight_future,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        cond_cont_context=prep["cond_cont_ctx"],
        cond_cont_future=prep["cond_cont_fut"] if cond_cont_future is None else cond_cont_future,
        cond_mode_context=prep["cond_mode_ctx"],
        cond_mode_future=prep["cond_mode_fut"] if cond_mode_future is None else cond_mode_future,
        drop_mask_context=prep["drop_mask_ctx"],
        drop_mask_future=prep["drop_mask_fut"] if drop_mask_future is None else drop_mask_future,
        relevance_scores_context=prep["relevance_scores_ctx"],
        relevance_scores_future=prep["relevance_scores_fut"] if relevance_scores_future is None else relevance_scores_future,
        relevance_embedding_type=prep["relevance_embedding_type"],
        drop_mask_relevance_context=prep["drop_mask_relevance_ctx"],
        drop_mask_relevance_future=prep["drop_mask_relevance_fut"] if drop_mask_relevance_future is None else drop_mask_relevance_future,
        subject_token=prep["subject_token"],
        age=prep["age"],
        sex=prep["sex"],
        motion=prep["motion"],
        field_strength=prep["field_strength"],
        condition_guidance_scope=condition_guidance_scope,
        context_guidance_scale=context_guidance_scale,
        drop_latent_context=drop_latent_context,
        stochastic_euler_eps=stochastic_euler_eps,
        stochastic_euler_noise_space=stochastic_euler_noise_space,
        euler_stop_sigma=euler_stop_sigma,
    )


@torch.inference_mode()
def _decode_to_bold(tokenizer, z, model=None, num_steps=DECODE_STEPS, chunk_size=DECODE_BATCH):
    if tokenizer is None:
        if is_raw_roi_encoder_config(getattr(model, "roi_encoder", None)):
            return decode_roi_tokens(z, model.roi_encoder)
        if z.shape[-1] != 1:
            raise ValueError(f"Expected one_roi_one_token latents with shape [B,T,R,1], got {tuple(z.shape)}")
        return z.squeeze(-1)

    bsz, seq_len, num_latents, latent_dim = z.shape
    z_flat = z.reshape(bsz * seq_len, num_latents, latent_dim)
    if chunk_size is None or int(chunk_size) <= 0:
        chunk_size = z_flat.shape[0]
    chunk_size = int(chunk_size)

    rec_chunks = []
    for start_idx in range(0, z_flat.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, z_flat.shape[0])
        rec_chunks.append(tokenizer.generate(z_flat[start_idx:end_idx], num_steps=num_steps))

    rec = torch.cat(rec_chunks, dim=0).reshape(bsz, seq_len, -1, tokenizer.input_timesteps)
    return rec.mean(dim=-1)
