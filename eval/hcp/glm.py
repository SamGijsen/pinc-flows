import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.stats import gamma, t as t_dist

from .common import base_subject_id, plot_line_with_band, sanitize_name
from .stats import add_stats_dataframe, add_stats_value

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HCP_TASKS = ("EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")
HCP_CONTRASTS = {
    "MOTOR": [
        {"name": "move_lh_gt_fixation", "event_names": ("move_lh", "fixation")},
        {"name": "move_rh_gt_fixation", "event_names": ("move_rh", "fixation")},
        {"name": "move_lf_gt_fixation", "event_names": ("move_lf", "fixation")},
        {"name": "move_rf_gt_fixation", "event_names": ("move_rf", "fixation")},
        {"name": "move_t_gt_fixation", "event_names": ("move_t", "fixation")},
    ],
    "EMOTION": [
        {"name": "stim_face_gt_stim_shape", "event_names": ("stim_face", "stim_shape")},
        {"name": "stim_shape_gt_fixation", "event_names": ("stim_shape", "fixation")},
    ],
    "GAMBLING": [
        {"name": "reward_feedback_gt_loss_feedback", "event_names": ("reward_feedback", "loss_feedback")},
        {"name": "loss_feedback_gt_reward_feedback", "event_names": ("loss_feedback", "reward_feedback")},
    ],
    "WM": [
        {"name": "block_2bk_gt_block_0bk", "event_names": ("block_2bk", "block_0bk")},
    ],
    "RELATIONAL": [
        {"name": "block_rel_gt_block_match", "event_names": ("block_rel", "block_match")},
    ],
    "LANGUAGE": [
        {"name": "story_listen_gt_math_listen", "event_names": ("story_listen", "math_listen")},
    ],
    "SOCIAL": [
        {"name": "movie_mental_gt_movie_random", "event_names": ("movie_mental", "movie_random")},
    ],
}

BASELINE_SUPPORT_MIN_TRS = 20
BASELINE_SUPPORT_MIN_FRAC = 0.05


def _spm_hrf(tr, time_length=32.0):
    t = np.arange(0, time_length, tr)
    hrf = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6
    return hrf / hrf.max()


def _convolve_hrf(x, hrf):
    return fftconvolve(x, hrf, mode="full")[: len(x)]


def _benjamini_hochberg(p):
    p = np.asarray(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def _corr_1d(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if den <= 0:
        return 0.0
    return float(np.sum(a * b) / den)


def _active_segments(x, threshold=1e-6):
    mask = np.asarray(x) > threshold
    padded = np.pad(mask.astype(int), 1)
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    return list(zip(starts, stops))


def _build_block_events_from_starts(starts_by_name, n_tp, stop_mask=None):
    stop_starts = [] if stop_mask is None else [start for start, _ in _active_segments(stop_mask)]
    all_starts = sorted((start, name) for name, starts in starts_by_name.items() for start in starts)
    out = {name: np.zeros(n_tp, dtype=float) for name in starts_by_name}
    for i, (start, name) in enumerate(all_starts):
        next_candidates = stop_starts.copy()
        if i + 1 < len(all_starts):
            next_candidates.append(all_starts[i + 1][0])
        next_candidates = [x for x in next_candidates if x > start]
        end = min(next_candidates) if next_candidates else n_tp
        out[name][start:end] = 1.0
    return out


def get_block_start_event_starts(f, task, run_idx):
    n_tp = int(f["valid_timepoints"][run_idx])
    if task == "LANGUAGE":
        return {
            "story_listen": [start for start, _ in _active_segments(f["events"]["story_listen"][run_idx, :n_tp])],
            "math_listen": [start for start, _ in _active_segments(f["events"]["math_listen"][run_idx, :n_tp])],
        }
    if task == "WM":
        return {
            "block_2bk": [start for start, _ in _active_segments(f["events"]["cue_2bk"][run_idx, :n_tp])],
            "block_0bk": [start for start, _ in _active_segments(f["events"]["cue_0bk"][run_idx, :n_tp])],
        }
    if task == "RELATIONAL":
        return {
            "block_rel": [start for start, _ in _active_segments(f["events"]["prompt_relation"][run_idx, :n_tp])],
            "block_match": [start for start, _ in _active_segments(f["events"]["prompt_match"][run_idx, :n_tp])],
        }
    if task == "GAMBLING":
        event_names = ["question_mark", "reward_feedback", "loss_feedback", "neutral_feedback", "fixation"]
        active = sum(np.asarray(f["events"][name][run_idx, :n_tp], dtype=float) for name in event_names)
        return {"block_start": [start for start, _ in _active_segments(active, threshold=0.05)]}
    if task == "SOCIAL":
        return {
            "movie_mental": [start for start, _ in _active_segments(f["events"]["movie_mental"][run_idx, :n_tp])],
            "movie_random": [start for start, _ in _active_segments(f["events"]["movie_random"][run_idx, :n_tp])],
        }
    if task == "EMOTION":
        return {
            "stim_face": [start for start, _ in _active_segments(f["events"]["cue_face"][run_idx, :n_tp])],
            "stim_shape": [start for start, _ in _active_segments(f["events"]["cue_shape"][run_idx, :n_tp])],
        }
    if task == "MOTOR":
        cue_map = {
            "move_lf": "cue_lf",
            "move_lh": "cue_lh",
            "move_rf": "cue_rf",
            "move_rh": "cue_rh",
            "move_t": "cue_t",
        }
        return {
            event_name: [start for start, _ in _active_segments(f["events"][cue_name][run_idx, :n_tp])]
            for event_name, cue_name in cue_map.items()
        }
    raise KeyError(f"no block-start rule for task {task}")


def _get_task_event(f, task, run_idx, event_name):
    n_tp = int(f["valid_timepoints"][run_idx])
    if event_name in f["events"]:
        return np.asarray(f["events"][event_name][run_idx, :n_tp], dtype=float)

    if task == "WM" and event_name in {"block_2bk", "block_0bk"}:
        starts = {
            "block_2bk": [start for start, _ in _active_segments(f["events"]["cue_2bk"][run_idx, :n_tp])],
            "block_0bk": [start for start, _ in _active_segments(f["events"]["cue_0bk"][run_idx, :n_tp])],
        }
        return _build_block_events_from_starts(starts, n_tp, f["events"]["fixation"][run_idx, :n_tp])[event_name]

    if task == "RELATIONAL" and event_name in {"block_rel", "block_match"}:
        starts = {
            "block_rel": [start for start, _ in _active_segments(f["events"]["prompt_relation"][run_idx, :n_tp])],
            "block_match": [start for start, _ in _active_segments(f["events"]["prompt_match"][run_idx, :n_tp])],
        }
        return _build_block_events_from_starts(starts, n_tp, f["events"]["fixation"][run_idx, :n_tp])[event_name]

    if task == "GAMBLING" and event_name in {"block_mostly_reward", "block_mostly_loss"}:
        question = np.asarray(f["events"]["question_mark"][run_idx, :n_tp], dtype=float)
        reward = np.asarray(f["events"]["reward_feedback"][run_idx, :n_tp], dtype=float)
        loss = np.asarray(f["events"]["loss_feedback"][run_idx, :n_tp], dtype=float)
        neutral = np.asarray(f["events"]["neutral_feedback"][run_idx, :n_tp], dtype=float)
        fixation = np.asarray(f["events"]["fixation"][run_idx, :n_tp], dtype=float)
        active = question + reward + loss + neutral + fixation
        blocks = _active_segments(active)
        out = {
            "block_mostly_reward": np.zeros(n_tp, dtype=float),
            "block_mostly_loss": np.zeros(n_tp, dtype=float),
        }
        for start, end in blocks:
            reward_sum = reward[start:end].sum()
            loss_sum = loss[start:end].sum()
            if reward_sum > loss_sum:
                out["block_mostly_reward"][start:end] = 1.0
            elif loss_sum > reward_sum:
                out["block_mostly_loss"][start:end] = 1.0
            else:
                raise ValueError(f"ambiguous gambling block in run {run_idx}")
        return out[event_name]

    raise KeyError(f"unknown event {event_name!r} for task {task}")


def _make_block_start_event(starts, n_tp, block_start_trs):
    x = np.zeros(n_tp, dtype=float)
    for start in starts:
        x[start:min(start + block_start_trs, n_tp)] = 1.0
    return x


def _get_design_event(f, task, run_idx, event_name, block_start_trs):
    n_tp = int(f["valid_timepoints"][run_idx])
    if task == "GAMBLING":
        return _get_task_event(f, task, run_idx, event_name)
    starts_by_name = get_block_start_event_starts(f, task, run_idx)
    if event_name in starts_by_name:
        return _make_block_start_event(starts_by_name[event_name], n_tp, block_start_trs)
    if event_name == "fixation" and "fixation" in f["events"]:
        fixation_starts = [start for start, _ in _active_segments(f["events"]["fixation"][run_idx, :n_tp])]
        return _make_block_start_event(fixation_starts, n_tp, block_start_trs)
    raise KeyError(f"no block-start event rule for {event_name!r} in task {task}")


def _plain_term_name(name):
    return str(name).split(":", 1)[-1]


def _is_baseline_name(name):
    name = _plain_term_name(name).lower()
    return name in {"fixation", "rest"} or "baseline" in name


def _baseline_name(contrast_name, names):
    for name in names:
        if _is_baseline_name(name):
            return _plain_term_name(name)
    if str(contrast_name).endswith("_gt_fixation"):
        return "fixation"
    if str(contrast_name).endswith("_gt_rest"):
        return "rest"
    return "none"


def _contrast_kind(contrast_name, names):
    if _baseline_name(contrast_name, names) != "none":
        return "task_vs_baseline"
    return "task_vs_task"


def _baseline_support(values):
    if len(values) == 0:
        return -1, -1.0, False
    values = np.concatenate(values, axis=0)
    support_trs = int((values > 1e-6).sum())
    support_frac = float(support_trs / len(values))
    sparse = support_trs < BASELINE_SUPPORT_MIN_TRS or support_frac < BASELINE_SUPPORT_MIN_FRAC
    return support_trs, support_frac, sparse


def _cropped_design_event(f, task, run_idx, event_name, block_start_trs, crop_starts):
    full = _get_design_event(f, task, run_idx, event_name, block_start_trs)
    pieces = [full[int(start):int(start) + int(block_start_trs)] for start in crop_starts]
    return np.concatenate(pieces, axis=0)


def _build_cropped_design(f, task, run_idx, event_names, tr, block_start_trs, crop_starts):
    hrf = _spm_hrf(tr)
    cols = []
    for name in event_names:
        full = _convolve_hrf(_get_design_event(f, task, run_idx, name, block_start_trs), hrf)
        pieces = [full[int(start):int(start) + int(block_start_trs)] for start in crop_starts]
        cols.append(np.concatenate(pieces, axis=0))
    cols.append(np.ones(len(crop_starts) * int(block_start_trs), dtype=np.float32))
    design_names = list(event_names) + ["intercept"]
    return np.column_stack(cols), design_names


def _get_ibc_design_term(f, run_idx, source, name, n_tp):
    if source == "events":
        return np.asarray(f["events"][name][run_idx, :n_tp], dtype=float)
    if source == "responses":
        return np.asarray(f["responses"][name][run_idx, :n_tp], dtype=float)
    raise AssertionError(f"unknown IBC contrast source {source!r}")


def _build_full_run_design(f, run_idx, terms, tr, n_tp):
    hrf = _spm_hrf(tr)
    cols = []
    weights = []
    design_names = []
    for term in terms:
        source = str(term["source"])
        name = str(term["name"])
        weights.append(float(term["weight"]))
        design_names.append(f"{source}:{name}")
        cols.append(_convolve_hrf(_get_ibc_design_term(f, run_idx, source, name, n_tp), hrf)[:n_tp])
    cols.append(np.ones(n_tp, dtype=np.float32))
    design_names.append("intercept")
    return np.column_stack(cols), design_names, np.asarray(weights + [0.0], dtype=float)


def _make_contrast(design_names, weights):
    return np.array([weights.get(name, 0.0) for name in design_names], dtype=float)


def _fit_run_contrast(y, x, c):
    xtx_inv = np.linalg.pinv(x.T @ x)
    betas = xtx_inv @ x.T @ y
    resid = y - x @ betas
    dof = x.shape[0] - x.shape[1]
    sigma2 = (resid ** 2).sum(axis=0) / dof
    effect = c @ betas
    effect_var = sigma2 * (c @ xtx_inv @ c)
    t_vals = effect / np.sqrt(effect_var)
    return effect, t_vals, betas


def _group_stats(effects, roi_labels):
    effects = np.asarray(effects, dtype=np.float32)
    group_mean = effects.mean(axis=0)
    group_std = effects.std(axis=0, ddof=1)
    group_se = group_std / np.sqrt(len(effects))
    group_t = group_mean / np.maximum(group_se, 1e-8)
    group_p = 2 * t_dist.sf(np.abs(group_t), df=len(effects) - 1)
    group_q = _benjamini_hochberg(group_p)
    return pd.DataFrame({
        "roi": np.arange(effects.shape[1]),
        "label": roi_labels,
        "mean_beta": group_mean,
        "std_beta": group_std,
        "t": group_t,
        "p": group_p,
        "q": group_q,
    })


def _group_stats_no_label(effects, roi_labels):
    return _group_stats(effects, roi_labels).drop(columns=["label"])


def _subject_effects(run_ids, effects):
    grouped = {}
    for run_id, effect in zip(run_ids, effects):
        grouped.setdefault(base_subject_id(run_id), []).append(effect)
    subject_ids = sorted(grouped.keys())
    values = np.stack([np.mean(grouped[sid], axis=0) for sid in subject_ids], axis=0)
    return subject_ids, values


def _icc31(real_vals, synth_vals):
    x = np.stack([real_vals, synth_vals], axis=1)
    n, k = x.shape
    mean_row = x.mean(axis=1, keepdims=True)
    mean_col = x.mean(axis=0, keepdims=True)
    mean_all = x.mean()
    msr = k * np.sum((mean_row - mean_all) ** 2) / max(n - 1, 1)
    mse = np.sum((x - mean_row - mean_col + mean_all) ** 2) / max((n - 1) * (k - 1), 1)
    return float((msr - mse) / max(msr + (k - 1) * mse, 1e-8))


def _dice_topk(real_t, synth_t, k):
    real_idx = set(np.argsort(np.abs(real_t))[-k:].tolist())
    synth_idx = set(np.argsort(np.abs(synth_t))[-k:].tolist())
    return float(2 * len(real_idx & synth_idx) / max(len(real_idx) + len(synth_idx), 1))


def run_glm_for_task(task, bank, roi_labels, tr_seconds, generation_frames, synth_future):
    dataset = bank["dataset"]
    run_to_synth = {
        run_id: synth_future[np.asarray(indices, dtype=np.int64)]
        for run_id, indices in bank["runs"].items()
    }
    outputs = []
    for contrast in HCP_CONTRASTS[task]:
        contrast_kind = _contrast_kind(contrast["name"], contrast["event_names"])
        baseline_name = _baseline_name(contrast["name"], contrast["event_names"])
        baseline_values = []
        real_effects = []
        synth_effects = []
        real_betas_by_name = {name: [] for name in contrast["event_names"]}
        synth_betas_by_name = {name: [] for name in contrast["event_names"]}
        selected_runs = []
        design_example = None
        for run_id, indices in bank["runs"].items():
            indices = np.asarray(indices, dtype=np.int64)
            run_idx = int(bank["run_indices"][indices[0]])
            crop_starts = bank["block_starts"][indices].astype(np.int64).tolist()
            x, design_names = _build_cropped_design(
                dataset.file,
                task,
                run_idx,
                contrast["event_names"],
                tr=float(tr_seconds),
                block_start_trs=int(generation_frames),
                crop_starts=crop_starts,
            )
            if baseline_name != "none":
                baseline_values.append(
                    _cropped_design_event(
                        dataset.file,
                        task,
                        run_idx,
                        baseline_name,
                        int(generation_frames),
                        crop_starts,
                    )
                )
            c = _make_contrast(
                design_names,
                {contrast["event_names"][0]: 1.0, contrast["event_names"][1]: -1.0},
            )
            ts = bank["future_signal"][indices][:, :, :, 0]
            y_real = np.concatenate(ts, axis=0)
            y_synth = np.concatenate(run_to_synth[run_id], axis=0)
            assert y_real.shape == y_synth.shape
            real_effect, _, real_betas = _fit_run_contrast(y_real, x, c)
            synth_effect, _, synth_betas = _fit_run_contrast(y_synth, x, c)
            real_effects.append(real_effect)
            synth_effects.append(synth_effect)
            for beta_idx, event_name in enumerate(contrast["event_names"]):
                real_betas_by_name[event_name].append(real_betas[beta_idx])
                synth_betas_by_name[event_name].append(synth_betas[beta_idx])
            selected_runs.append(run_id)
            if design_example is None:
                design_example = pd.DataFrame(x[:, :len(contrast["event_names"])], columns=list(contrast["event_names"]))

        real_effects = np.stack(real_effects, axis=0)
        synth_effects = np.stack(synth_effects, axis=0)
        baseline_support_trs, baseline_support_frac, baseline_sparse = _baseline_support(baseline_values)
        real_df = _group_stats(real_effects, roi_labels)
        synth_df = _group_stats(synth_effects, roi_labels)
        real_subject_ids, real_subject_vals = _subject_effects(selected_runs, real_effects)
        synth_subject_ids, synth_subject_vals = _subject_effects(selected_runs, synth_effects)
        assert real_subject_ids == synth_subject_ids
        condition_betas = {}
        for event_name in contrast["event_names"]:
            real_event_betas = np.stack(real_betas_by_name[event_name], axis=0)
            synth_event_betas = np.stack(synth_betas_by_name[event_name], axis=0)
            real_event_subject_ids, real_event_subject_vals = _subject_effects(selected_runs, real_event_betas)
            synth_event_subject_ids, synth_event_subject_vals = _subject_effects(selected_runs, synth_event_betas)
            assert real_event_subject_ids == synth_event_subject_ids
            condition_betas[event_name] = {
                "real_df": _group_stats_no_label(real_event_betas, roi_labels),
                "synth_df": _group_stats_no_label(synth_event_betas, roi_labels),
                "real_subject_vals": real_event_subject_vals,
                "synth_subject_vals": synth_event_subject_vals,
            }

        real_t = real_df["t"].to_numpy()
        synth_t = synth_df["t"].to_numpy()
        sig_count = int(max((real_df["q"] < 0.05).sum(), 25))
        key_pos = np.argsort(real_t)[-25:]
        key_neg = np.argsort(real_t)[:25]
        icc_per_roi = np.asarray([
            _icc31(real_subject_vals[:, roi_idx], synth_subject_vals[:, roi_idx])
            for roi_idx in range(real_subject_vals.shape[1])
        ], dtype=np.float32)

        outputs.append({
            "contrast": contrast,
            "contrast_kind": contrast_kind,
            "baseline_name": baseline_name,
            "baseline_support_trs": baseline_support_trs,
            "baseline_support_frac": baseline_support_frac,
            "baseline_sparse": baseline_sparse,
            "selected_runs": selected_runs,
            "design_example": design_example,
            "real_df": real_df,
            "synth_df": synth_df,
            "spatial_corr": _corr_1d(real_t, synth_t),
            "dice_topk": _dice_topk(real_t, synth_t, sig_count),
            "topk": sig_count,
            "real_subject_vals": real_subject_vals,
            "synth_subject_vals": synth_subject_vals,
            "icc_per_roi": icc_per_roi,
            "key_pos": key_pos,
            "key_neg": key_neg,
            "condition_betas": condition_betas,
        })
    return outputs


def run_glm_for_ibc_task(task, bank, roi_labels, tr_seconds, contrasts, synth_future):
    dataset = bank["dataset"]
    run_to_synth = {
        run_id: np.concatenate(synth_future[np.asarray(indices, dtype=np.int64)], axis=0)
        for run_id, indices in bank["runs"].items()
    }
    outputs = []
    for contrast in contrasts:
        terms = list(contrast["terms"])
        term_names = [term["name"] for term in terms]
        contrast_kind = _contrast_kind(contrast["name"], term_names)
        baseline_name = _baseline_name(contrast["name"], term_names)
        baseline_values = []
        real_effects = []
        synth_effects = []
        real_betas_by_name = {}
        synth_betas_by_name = {}
        selected_runs = []
        design_example = None
        for run_id, indices in bank["runs"].items():
            indices = np.asarray(indices, dtype=np.int64)
            run_idx = int(bank["run_indices"][indices[0]])
            y_real = np.concatenate(bank["future_signal"][indices][:, :, :, 0], axis=0)
            y_synth = run_to_synth[run_id]
            assert y_real.shape == y_synth.shape
            x, design_names, c = _build_full_run_design(
                dataset.file,
                run_idx,
                terms,
                tr=float(tr_seconds),
                n_tp=int(y_real.shape[0]),
            )
            if baseline_name != "none":
                baseline_values.append(_get_ibc_design_term(dataset.file, run_idx, "events", baseline_name, int(y_real.shape[0])))
            real_effect, _, real_betas = _fit_run_contrast(y_real, x, c)
            synth_effect, _, synth_betas = _fit_run_contrast(y_synth, x, c)
            real_effects.append(real_effect)
            synth_effects.append(synth_effect)
            for beta_idx, event_name in enumerate(design_names[:len(terms)]):
                real_betas_by_name.setdefault(event_name, []).append(real_betas[beta_idx])
                synth_betas_by_name.setdefault(event_name, []).append(synth_betas[beta_idx])
            selected_runs.append(run_id)
            if design_example is None:
                design_example = pd.DataFrame(x[:, :len(terms)], columns=design_names[:len(terms)])

        real_effects = np.stack(real_effects, axis=0)
        synth_effects = np.stack(synth_effects, axis=0)
        baseline_support_trs, baseline_support_frac, baseline_sparse = _baseline_support(baseline_values)
        real_df = _group_stats(real_effects, roi_labels)
        synth_df = _group_stats(synth_effects, roi_labels)
        real_subject_ids, real_subject_vals = _subject_effects(selected_runs, real_effects)
        synth_subject_ids, synth_subject_vals = _subject_effects(selected_runs, synth_effects)
        assert real_subject_ids == synth_subject_ids
        condition_betas = {}
        for event_name in design_names[:len(terms)]:
            real_event_betas = np.stack(real_betas_by_name[event_name], axis=0)
            synth_event_betas = np.stack(synth_betas_by_name[event_name], axis=0)
            real_event_subject_ids, real_event_subject_vals = _subject_effects(selected_runs, real_event_betas)
            synth_event_subject_ids, synth_event_subject_vals = _subject_effects(selected_runs, synth_event_betas)
            assert real_event_subject_ids == synth_event_subject_ids
            condition_betas[event_name] = {
                "real_df": _group_stats_no_label(real_event_betas, roi_labels),
                "synth_df": _group_stats_no_label(synth_event_betas, roi_labels),
                "real_subject_vals": real_event_subject_vals,
                "synth_subject_vals": synth_event_subject_vals,
            }

        real_t = real_df["t"].to_numpy()
        synth_t = synth_df["t"].to_numpy()
        sig_count = int(max((real_df["q"] < 0.05).sum(), 25))
        key_pos = np.argsort(real_t)[-25:]
        key_neg = np.argsort(real_t)[:25]
        icc_per_roi = np.asarray([
            _icc31(real_subject_vals[:, roi_idx], synth_subject_vals[:, roi_idx])
            for roi_idx in range(real_subject_vals.shape[1])
        ], dtype=np.float32)

        outputs.append({
            "contrast": {
                "name": str(contrast["name"]),
                "event_names": tuple(design_names[:len(terms)]),
                "terms": terms,
            },
            "contrast_kind": contrast_kind,
            "baseline_name": baseline_name,
            "baseline_support_trs": baseline_support_trs,
            "baseline_support_frac": baseline_support_frac,
            "baseline_sparse": baseline_sparse,
            "selected_runs": selected_runs,
            "design_example": design_example,
            "real_df": real_df,
            "synth_df": synth_df,
            "spatial_corr": _corr_1d(real_t, synth_t),
            "dice_topk": _dice_topk(real_t, synth_t, sig_count),
            "topk": sig_count,
            "real_subject_vals": real_subject_vals,
            "synth_subject_vals": synth_subject_vals,
            "icc_per_roi": icc_per_roi,
            "key_pos": key_pos,
            "key_neg": key_neg,
            "condition_betas": condition_betas,
        })
    return outputs


def save_glm(
    task_dir,
    outputs,
    guidance_scale,
    context_guidance_scale,
    bundle=None,
    stats_prefix=None,
    image_contrasts=None,
):
    image_contrasts = set(image_contrasts or [])
    task_dir.mkdir(parents=True, exist_ok=True)
    contrast_names = []
    summary_rows = []
    for output in outputs:
        name = output["contrast"]["name"]
        contrast_key = sanitize_name(name)
        contrast_names.append(name)

        real_t = output["real_df"]["t"].to_numpy()
        synth_t = output["synth_df"]["t"].to_numpy()
        real_subject_mean_beta = output["real_subject_vals"].mean(axis=0)
        synth_subject_mean_beta = output["synth_subject_vals"].mean(axis=0)
        real_subject_std_beta = output["real_subject_vals"].std(axis=0)
        synth_subject_std_beta = output["synth_subject_vals"].std(axis=0)
        subject_mean_beta_corr = _corr_1d(real_subject_mean_beta, synth_subject_mean_beta)
        key_pos = np.asarray(output["key_pos"], dtype=np.int64)
        key_neg = np.asarray(output["key_neg"], dtype=np.int64)
        summary = {
            "contrast": name,
            "guidance_scale": float(guidance_scale),
            "context_guidance_scale": float(context_guidance_scale),
            "spatial_corr": output["spatial_corr"],
            "subject_mean_beta_corr": subject_mean_beta_corr,
            "dice_topk": output["dice_topk"],
            "topk": int(output["topk"]),
            "real_top_pos_mean_beta": float(output["real_df"]["mean_beta"].to_numpy()[key_pos].mean()),
            "synth_top_pos_mean_beta": float(output["synth_df"]["mean_beta"].to_numpy()[key_pos].mean()),
            "real_top_neg_mean_beta": float(output["real_df"]["mean_beta"].to_numpy()[key_neg].mean()),
            "synth_top_neg_mean_beta": float(output["synth_df"]["mean_beta"].to_numpy()[key_neg].mean()),
            "icc_mean": float(output["icc_per_roi"].mean()),
        }
        summary_rows.append(summary)
        if bundle is not None and stats_prefix is not None:
            add_stats_value(bundle, stats_prefix, contrast_key, "name", value=name)
            add_stats_value(bundle, stats_prefix, contrast_key, "event_names", value=np.asarray(output["contrast"]["event_names"], dtype=str))
            if "terms" in output["contrast"]:
                add_stats_value(bundle, stats_prefix, contrast_key, "term_source", value=np.asarray([term["source"] for term in output["contrast"]["terms"]], dtype=str))
                add_stats_value(bundle, stats_prefix, contrast_key, "term_name", value=np.asarray([term["name"] for term in output["contrast"]["terms"]], dtype=str))
                add_stats_value(bundle, stats_prefix, contrast_key, "term_weight", value=np.asarray([term["weight"] for term in output["contrast"]["terms"]], dtype=np.float32))
            add_stats_value(bundle, stats_prefix, contrast_key, "selected_runs", value=np.asarray(output["selected_runs"], dtype=str))
            add_stats_dataframe(bundle, stats_prefix, contrast_key, "real_group", frame=output["real_df"])
            add_stats_dataframe(bundle, stats_prefix, contrast_key, "synth_group", frame=output["synth_df"])
            add_stats_dataframe(bundle, stats_prefix, contrast_key, "design_example", frame=output["design_example"])
            add_stats_value(bundle, stats_prefix, contrast_key, "subject_level", "real_subject_vals", value=output["real_subject_vals"])
            add_stats_value(bundle, stats_prefix, contrast_key, "subject_level", "synth_subject_vals", value=output["synth_subject_vals"])
            add_stats_value(bundle, stats_prefix, contrast_key, "subject_level", "icc_per_roi", value=output["icc_per_roi"])
            add_stats_value(bundle, stats_prefix, contrast_key, "subject_level", "key_pos", value=key_pos.astype(np.int64))
            add_stats_value(bundle, stats_prefix, contrast_key, "subject_level", "key_neg", value=key_neg.astype(np.int64))
            if "condition_betas" in output:
                for event_name, event_output in output["condition_betas"].items():
                    event_key = sanitize_name(event_name)
                    add_stats_value(bundle, stats_prefix, contrast_key, "condition_betas", event_key, "name", value=event_name)
                    add_stats_dataframe(bundle, stats_prefix, contrast_key, "condition_betas", event_key, "real_group", frame=event_output["real_df"])
                    add_stats_dataframe(bundle, stats_prefix, contrast_key, "condition_betas", event_key, "synth_group", frame=event_output["synth_df"])
                    add_stats_value(bundle, stats_prefix, contrast_key, "condition_betas", event_key, "subject_level", "real_subject_vals", value=event_output["real_subject_vals"])
                    add_stats_value(bundle, stats_prefix, contrast_key, "condition_betas", event_key, "subject_level", "synth_subject_vals", value=event_output["synth_subject_vals"])
            for summary_key, summary_value in summary.items():
                add_stats_value(bundle, stats_prefix, contrast_key, "summary", summary_key, value=summary_value)

        if name not in image_contrasts:
            continue

        contrast_dir = task_dir / name
        contrast_dir.mkdir(parents=True, exist_ok=True)

        x = np.arange(1, real_subject_mean_beta.shape[0] + 1, dtype=np.int64)
        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        plot_line_with_band(ax, x, real_subject_mean_beta, real_subject_std_beta, "real")
        plot_line_with_band(ax, x, synth_subject_mean_beta, synth_subject_std_beta, "synth")
        ax.set_title(f"{name} mean beta by ROI, r={subject_mean_beta_corr:.3f}")
        ax.set_xlabel("ROI index")
        ax.set_ylabel("mean beta")
        ax.set_xlim(1, int(x[-1]))
        ax.legend()
        fig.savefig(contrast_dir / "beta_profile.png")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        output["design_example"].plot(ax=axes[0, 0])
        axes[0, 0].set_title(name)
        axes[0, 1].scatter(real_t, synth_t, s=10)
        axes[0, 1].set_xlabel("real t")
        axes[0, 1].set_ylabel("synth t")
        axes[0, 1].set_title(
            f"t-map correlation, guidance={guidance_scale}, context={context_guidance_scale}"
        )
        axes[1, 0].bar(
            ["real_top_pos", "synth_top_pos", "real_top_neg", "synth_top_neg"],
            [
                summary["real_top_pos_mean_beta"],
                summary["synth_top_pos_mean_beta"],
                summary["real_top_neg_mean_beta"],
                summary["synth_top_neg_mean_beta"],
            ],
        )
        axes[1, 0].set_title("real top positive/negative contrast ROIs")
        axes[1, 1].hist(output["icc_per_roi"], bins=30)
        axes[1, 1].set_title("ICC per ROI")
        fig.savefig(contrast_dir / "overview.png")
        plt.close(fig)

    if bundle is not None and stats_prefix is not None:
        add_stats_value(bundle, stats_prefix, "guidance_scale", value=float(guidance_scale))
        add_stats_value(bundle, stats_prefix, "context_guidance_scale", value=float(context_guidance_scale))
        add_stats_value(bundle, stats_prefix, "contrast_names", value=np.asarray(contrast_names, dtype=str))
    return pd.DataFrame(summary_rows)


def summarize_glm_outputs(outputs, guidance_scale, context_guidance_scale, phrase_variant_idx):
    rows = []
    for output in outputs:
        name = output["contrast"]["name"]
        real_subject_mean_beta = output["real_subject_vals"].mean(axis=0)
        synth_subject_mean_beta = output["synth_subject_vals"].mean(axis=0)
        key_pos = np.asarray(output["key_pos"], dtype=np.int64)
        key_neg = np.asarray(output["key_neg"], dtype=np.int64)
        row = {
            "phrase_variant_idx": int(phrase_variant_idx),
            "contrast": name,
            "contrast_kind": output["contrast_kind"],
            "baseline_name": output["baseline_name"],
            "baseline_support_trs": int(output["baseline_support_trs"]),
            "baseline_support_frac": float(output["baseline_support_frac"]),
            "baseline_sparse": bool(output["baseline_sparse"]),
            "guidance_scale": float(guidance_scale),
            "context_guidance_scale": float(context_guidance_scale),
            "spatial_corr": output["spatial_corr"],
            "subject_mean_beta_corr": _corr_1d(real_subject_mean_beta, synth_subject_mean_beta),
            "dice_topk": output["dice_topk"],
            "topk": int(output["topk"]),
            "real_top_pos_mean_beta": float(output["real_df"]["mean_beta"].to_numpy()[key_pos].mean()),
            "synth_top_pos_mean_beta": float(output["synth_df"]["mean_beta"].to_numpy()[key_pos].mean()),
            "real_top_neg_mean_beta": float(output["real_df"]["mean_beta"].to_numpy()[key_neg].mean()),
            "synth_top_neg_mean_beta": float(output["synth_df"]["mean_beta"].to_numpy()[key_neg].mean()),
            "icc_mean": float(output["icc_per_roi"].mean()),
        }
        if phrase_variant_idx == -1:
            row["condition_betas"] = {
                event_name: {
                    "mean_beta_corr": _corr_1d(
                        event_output["real_df"]["mean_beta"].to_numpy(),
                        event_output["synth_df"]["mean_beta"].to_numpy(),
                    ),
                }
                for event_name, event_output in output["condition_betas"].items()
            }
        rows.append(row)
    return pd.DataFrame(rows)


def save_glm_group_only(outputs, guidance_scale, context_guidance_scale, phrase_variant_idx, bundle, stats_prefix):
    contrast_names = []
    summary_frame = summarize_glm_outputs(outputs, guidance_scale, context_guidance_scale, phrase_variant_idx)
    add_stats_value(bundle, stats_prefix, "phrase_variant_idx", value=int(phrase_variant_idx))
    add_stats_value(bundle, stats_prefix, "guidance_scale", value=float(guidance_scale))
    add_stats_value(bundle, stats_prefix, "context_guidance_scale", value=float(context_guidance_scale))
    for output, summary in zip(outputs, summary_frame.to_dict(orient="records")):
        name = output["contrast"]["name"]
        contrast_key = sanitize_name(name)
        contrast_names.append(name)

        add_stats_value(bundle, stats_prefix, contrast_key, "name", value=name)
        add_stats_value(bundle, stats_prefix, contrast_key, "event_names", value=np.asarray(output["contrast"]["event_names"], dtype=str))
        if "terms" in output["contrast"]:
            add_stats_value(bundle, stats_prefix, contrast_key, "term_source", value=np.asarray([term["source"] for term in output["contrast"]["terms"]], dtype=str))
            add_stats_value(bundle, stats_prefix, contrast_key, "term_name", value=np.asarray([term["name"] for term in output["contrast"]["terms"]], dtype=str))
            add_stats_value(bundle, stats_prefix, contrast_key, "term_weight", value=np.asarray([term["weight"] for term in output["contrast"]["terms"]], dtype=np.float32))
        add_stats_value(bundle, stats_prefix, contrast_key, "subject_level", "real_subject_vals", value=np.asarray(output["real_subject_vals"], dtype=np.float16))
        add_stats_value(bundle, stats_prefix, contrast_key, "subject_level", "synth_subject_vals", value=np.asarray(output["synth_subject_vals"], dtype=np.float16))
        for event_name, event_output in output["condition_betas"].items():
            event_key = sanitize_name(event_name)
            add_stats_value(bundle, stats_prefix, contrast_key, "condition_betas", event_key, "name", value=event_name)
            if phrase_variant_idx == -1:
                real_mean_beta = event_output["real_df"]["mean_beta"].to_numpy()
                synth_mean_beta = event_output["synth_df"]["mean_beta"].to_numpy()
                add_stats_value(
                    bundle,
                    stats_prefix,
                    contrast_key,
                    "condition_betas",
                    event_key,
                    "real_group",
                    "mean_beta",
                    value=real_mean_beta.astype(np.float32),
                )
                add_stats_value(
                    bundle,
                    stats_prefix,
                    contrast_key,
                    "condition_betas",
                    event_key,
                    "synth_group",
                    "mean_beta",
                    value=synth_mean_beta.astype(np.float32),
                )
                add_stats_value(
                    bundle,
                    stats_prefix,
                    contrast_key,
                    "condition_betas",
                    event_key,
                    "summary",
                    "mean_beta_corr",
                    value=_corr_1d(real_mean_beta, synth_mean_beta),
                )
        for summary_key, summary_value in summary.items():
            if isinstance(summary_value, dict):
                continue
            add_stats_value(bundle, stats_prefix, contrast_key, "summary", summary_key, value=summary_value)

    add_stats_value(bundle, stats_prefix, "contrast_names", value=np.asarray(contrast_names, dtype=str))
    return summary_frame


def average_glm_phrase_variant_outputs(outputs_by_variant, roi_labels):
    variant_indices = sorted(outputs_by_variant)
    outputs = []
    for contrast_idx, first_output in enumerate(outputs_by_variant[variant_indices[0]]):
        name = first_output["contrast"]["name"]
        for variant_idx in variant_indices:
            assert outputs_by_variant[variant_idx][contrast_idx]["contrast"]["name"] == name
            assert outputs_by_variant[variant_idx][contrast_idx]["selected_runs"] == first_output["selected_runs"]
        real_subject_vals = np.stack([outputs_by_variant[idx][contrast_idx]["real_subject_vals"] for idx in variant_indices], axis=0).mean(axis=0).astype(np.float32)
        synth_subject_vals = np.stack([outputs_by_variant[idx][contrast_idx]["synth_subject_vals"] for idx in variant_indices], axis=0).mean(axis=0).astype(np.float32)
        real_df = _group_stats(real_subject_vals, roi_labels)
        synth_df = _group_stats(synth_subject_vals, roi_labels)
        real_t = real_df["t"].to_numpy()
        synth_t = synth_df["t"].to_numpy()
        topk = int(max((real_df["q"] < 0.05).sum(), 25))
        key_pos = np.argsort(real_t)[-25:]
        key_neg = np.argsort(real_t)[:25]
        icc_per_roi = np.asarray([
            _icc31(real_subject_vals[:, roi_idx], synth_subject_vals[:, roi_idx])
            for roi_idx in range(real_subject_vals.shape[1])
        ], dtype=np.float32)
        condition_betas = {}
        for event_name in first_output["condition_betas"]:
            for variant_idx in variant_indices:
                assert event_name in outputs_by_variant[variant_idx][contrast_idx]["condition_betas"]
            event_outputs = [
                outputs_by_variant[idx][contrast_idx]["condition_betas"][event_name]
                for idx in variant_indices
            ]
            real_event_subject_vals = np.stack([
                event_output["real_subject_vals"]
                for event_output in event_outputs
            ], axis=0).mean(axis=0).astype(np.float32)
            synth_event_subject_vals = np.stack([
                event_output["synth_subject_vals"]
                for event_output in event_outputs
            ], axis=0).mean(axis=0).astype(np.float32)
            condition_betas[event_name] = {
                "real_df": _group_stats_no_label(real_event_subject_vals, roi_labels),
                "synth_df": _group_stats_no_label(synth_event_subject_vals, roi_labels),
                "real_subject_vals": real_event_subject_vals,
                "synth_subject_vals": synth_event_subject_vals,
            }
        output = dict(first_output)
        output.update(
            real_df=real_df,
            synth_df=synth_df,
            spatial_corr=_corr_1d(real_t, synth_t),
            dice_topk=_dice_topk(real_t, synth_t, topk),
            topk=topk,
            real_subject_vals=real_subject_vals,
            synth_subject_vals=synth_subject_vals,
            icc_per_roi=icc_per_roi,
            key_pos=key_pos,
            key_neg=key_neg,
            condition_betas=condition_betas,
        )
        outputs.append(output)
    return outputs

def save_glm_guidance_compare(outputs_by_scale, bundle=None, stats_prefix=None):
    if bundle is None or stats_prefix is None:
        return

    scales = sorted(outputs_by_scale.keys())
    default_outputs = outputs_by_scale[scales[0]]
    add_stats_value(bundle, stats_prefix, "scales", value=np.asarray(scales, dtype=np.float32))
    contrast_names = []
    for contrast_idx, output in enumerate(default_outputs):
        contrast_name = output["contrast"]["name"]
        contrast_key = sanitize_name(contrast_name)
        contrast_names.append(contrast_name)
        roi_sets = []
        roi_indices = []
        for roi_set, key_indices in (
            ("real_top_pos", np.asarray(output["key_pos"], dtype=np.int64)),
            ("real_top_neg", np.asarray(output["key_neg"], dtype=np.int64)),
        ):
            for roi_idx in key_indices:
                roi_sets.append(roi_set)
                roi_indices.append(int(roi_idx))

        labels = np.asarray(
            [str(default_outputs[contrast_idx]["real_df"].iloc[idx]["label"]) for idx in roi_indices],
            dtype=str,
        )
        real_mean_beta = np.empty((len(scales), len(roi_indices)), dtype=np.float32)
        real_t = np.empty((len(scales), len(roi_indices)), dtype=np.float32)
        synth_mean_beta = np.empty((len(scales), len(roi_indices)), dtype=np.float32)
        synth_t = np.empty((len(scales), len(roi_indices)), dtype=np.float32)
        for scale_idx, guidance_scale in enumerate(scales):
            curr = outputs_by_scale[guidance_scale][contrast_idx]
            curr_real = curr["real_df"]
            curr_synth = curr["synth_df"]
            for roi_col, roi_idx in enumerate(roi_indices):
                real_mean_beta[scale_idx, roi_col] = float(curr_real.iloc[roi_idx]["mean_beta"])
                real_t[scale_idx, roi_col] = float(curr_real.iloc[roi_idx]["t"])
                synth_mean_beta[scale_idx, roi_col] = float(curr_synth.iloc[roi_idx]["mean_beta"])
                synth_t[scale_idx, roi_col] = float(curr_synth.iloc[roi_idx]["t"])

        add_stats_value(bundle, stats_prefix, contrast_key, "name", value=contrast_name)
        add_stats_value(bundle, stats_prefix, contrast_key, "roi_set", value=np.asarray(roi_sets, dtype=str))
        add_stats_value(bundle, stats_prefix, contrast_key, "roi", value=np.asarray(roi_indices, dtype=np.int64))
        add_stats_value(bundle, stats_prefix, contrast_key, "label", value=labels)
        add_stats_value(bundle, stats_prefix, contrast_key, "real_mean_beta", value=real_mean_beta)
        add_stats_value(bundle, stats_prefix, contrast_key, "real_t", value=real_t)
        add_stats_value(bundle, stats_prefix, contrast_key, "synth_mean_beta", value=synth_mean_beta)
        add_stats_value(bundle, stats_prefix, contrast_key, "synth_t", value=synth_t)
    add_stats_value(bundle, stats_prefix, "contrast_names", value=np.asarray(contrast_names, dtype=str))
