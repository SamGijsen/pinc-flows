import numpy as np
import torch

from util.dynamics_inference import _decode_to_bold, _sample_future

from .bank import _move_prep_to_device
from .report import compute_window_stats
from .stats import add_stats_value


def first_prep_rows_by_run(prep, run_ids):
    rows = []
    seen = set()
    for idx, run_id in enumerate(run_ids.tolist()):
        key = str(run_id)
        if key in seen:
            continue
        seen.add(key)
        rows.append(idx)

    row_idx = torch.as_tensor(rows, dtype=torch.long)
    out = {}
    for key, value in prep.items():
        if torch.is_tensor(value):
            out[key] = value[row_idx]
        elif value is None:
            out[key] = None
        else:
            raise TypeError(f"unexpected prep value for {key}: {type(value).__name__}")
    return out, np.asarray(rows, dtype=np.int64)


def _repeat_last_frame(value, length):
    if torch.is_tensor(value):
        shape = [-1] * value.ndim
        shape[1] = int(length)
        return value[:, -1:].expand(*shape).contiguous()
    if value is None:
        return None
    raise TypeError(f"unexpected prep value type: {type(value).__name__}")


def _fixation_conditions(prep, length, suffix):
    return {
        f"cond_disc_{suffix}": _repeat_last_frame(prep["cond_disc_ctx"], length),
        f"cond_disc_weight_{suffix}": _repeat_last_frame(prep["cond_disc_weight_ctx"], length),
        f"cond_cont_{suffix}": _repeat_last_frame(prep["cond_cont_ctx"], length),
        f"cond_mode_{suffix}": _repeat_last_frame(prep["cond_mode_ctx"], length),
        f"drop_mask_{suffix}": _repeat_last_frame(prep["drop_mask_ctx"], length),
        f"relevance_scores_{suffix}": _repeat_last_frame(prep["relevance_scores_ctx"], length),
        f"drop_mask_relevance_{suffix}": _repeat_last_frame(prep["drop_mask_relevance_ctx"], length),
    }


def generate_fixation_ar_windows(
    model,
    tokenizer,
    prep,
    device,
    cfg,
    progress_label,
):
    ctx_len = prep["z_context"].shape[1]
    gen_len = prep["cond_disc_fut"].shape[1]
    total = prep["z_context"].shape[0]
    halfway = max(1, total // 2)
    chunks = []
    printed_halfway = False

    for start in range(0, total, cfg["batch_size"]):
        end = min(start + cfg["batch_size"], total)
        chunk = {key: value[start:end] if torch.is_tensor(value) else value for key, value in prep.items()}
        chunk = _move_prep_to_device(chunk, device)
        fix_future = _fixation_conditions(chunk, gen_len, "fut")
        fix_context = _fixation_conditions(chunk, ctx_len, "ctx")
        step_windows = []

        for step in range(cfg["steps"]):
            z_future = _sample_future(
                model,
                chunk,
                num_steps=cfg["num_steps"],
                guidance_scale=cfg["guidance_scale"],
                context_guidance_scale=cfg["context_guidance_scale"],
                seed=cfg["seed"] + step * total + start,
                cond_disc_future=fix_future["cond_disc_fut"],
                cond_disc_weight_future=fix_future["cond_disc_weight_fut"],
                cond_cont_future=fix_future["cond_cont_fut"],
                cond_mode_future=fix_future["cond_mode_fut"],
                drop_mask_future=fix_future["drop_mask_fut"],
                relevance_scores_future=fix_future["relevance_scores_fut"],
                drop_mask_relevance_future=fix_future["drop_mask_relevance_fut"],
                condition_guidance_scope="all_timepoints",
                stochastic_euler_eps=cfg["stochastic_euler_eps"],
                stochastic_euler_noise_space=cfg["stochastic_euler_noise_space"],
                euler_stop_sigma=cfg["euler_stop_sigma"],
            )
            step_windows.append(_decode_to_bold(tokenizer, z_future, model=model).cpu())
            chunk["z_context"] = z_future[:, -ctx_len:].contiguous()
            for key, value in fix_context.items():
                chunk[key] = value

        chunks.append(torch.stack(step_windows, dim=0).cpu().numpy().astype(np.float32))
        if not printed_halfway and end >= halfway:
            print(f"[hcp-eval] {progress_label}: 50% ({end}/{total})", flush=True)
            printed_halfway = True

    print(f"[hcp-eval] {progress_label}: 100% ({total}/{total})", flush=True)
    return np.concatenate(chunks, axis=1)


def save_fixation_ar_stats(bundle, stats_prefix, windows, tr_seconds, row_indices):
    add_stats_value(bundle, stats_prefix, "row_indices", value=row_indices)
    add_stats_value(bundle, stats_prefix, "steps", value=np.arange(windows.shape[0], dtype=np.int64))

    rows = []
    for step_idx, step_windows in enumerate(windows):
        stats = compute_window_stats(step_windows, tr_seconds)
        step_key = f"step_{step_idx:02d}"
        for key, value in stats.items():
            add_stats_value(bundle, stats_prefix, step_key, key, value=value)
        rows.append({
            "step": step_idx,
            "roi_var_mean": float(stats["roi_var_mean"].mean()),
            "lag1_autocorr": float(stats["ac_mean"][0]),
            "psd_mean": float(stats["psd_mean"].mean()),
        })

    return rows
