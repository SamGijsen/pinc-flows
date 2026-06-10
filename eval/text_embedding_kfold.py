import glob
import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.stats import gamma
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.yeo17 import load_schaefer_yeo17_spec_for_atlas


HCP_DIR = "/TODO_SET_PATH/h5"
NAKAI_DIR = "/TODO_SET_PATH/nakai"
IBC_DIR = "/TODO_SET_PATH/task_h5"
TRAIN_TARGET_LEVEL = "group"  # "group" or "subject"
EVAL_TARGET_LEVEL = "group"  # "group" or "subject"
ATLAS = "schaefer1000"
TARGET_BETA_SPACE = ATLAS.replace("schaefer", "parcel")  # e.g. "parcel400" or "yeo17"
TR = 1.0
HCP_MAX_SUBJECTS = 100

CV_N_SPLITS = 7
EVENT_PAIRING_MODE = "pair_by_variant_idx"  # "pair_by_variant_idx" or "cartesian_product"
USE_PCA_TARGETS = False
ESTIMATOR_TYPES = ["ridge"] #nn, Ridge
RUN_MODE = "cv"  # "cv" or "fit_save_event_ridge"

FIT_SAVE_EVENT_HOLDOUT_PATHS = [
    f"{HCP_DIR}/HCP_YA_EMOTION_lp0.25_tr1.00_zscore_20260402.h5",
    f"{HCP_DIR}/HCP_YA_MOTOR_lp0.25_tr1.00_zscore_20260402.h5",
]
FIT_SAVE_EVENT_RIDGE_ALPHA = 10000.0
FIT_SAVE_EVENT_AGGREGATE_MODE = "mean"
FIT_SAVE_EVENT_OUTPUT_DIR = "/TODO_SET_PATH/ridge"
FIT_SAVE_EVENT_OUTPUT_NAME = "event_ridge_yeo17_alpha10k_mean.pkl"

EVENT_RIDGE_ALPHAS = [1e2,1e3, 1e4, 1e5, 1e6]
EVENT_PCA_COMPONENTS = [100]
RESPONSE_RIDGE_ALPHAS = [1e1]
RESPONSE_PCA_COMPONENTS = [100]
EVENT_NN_K_VALUES = [1]
RESPONSE_NN_K_VALUES = [1]
R2_STD_FLOOR = 1e-6
R2_ABS_DIAGNOSTIC_THRESHOLD = 1e6
DIAGNOSTIC_TOP_K = 10


def spm_hrf(tr, time_length=32.0):
    t = np.arange(0, time_length, tr)
    hrf = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6
    return (hrf / hrf.max()).astype(np.float32)


def convolve_event_matrix(event_matrix, hrf):
    return np.stack(
        [
            fftconvolve(event_row, hrf, mode="full")[: event_row.shape[0]]
            for event_row in event_matrix
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def subject_key(run_id):
    return run_id.split("_")[0]


def decode_text_rows(dataset):
    return [raw.decode() if isinstance(raw, bytes) else str(raw) for raw in dataset[:]]


def text_at(rows, idx, default=""):
    if not rows:
        return default
    if idx < len(rows):
        return rows[idx]
    return rows[0]


_condition_beta_cache = {}
_yeo17_spec_cache = None


def yeo17_spec():
    global _yeo17_spec_cache
    if _yeo17_spec_cache is not None:
        return _yeo17_spec_cache

    _yeo17_spec_cache = load_schaefer_yeo17_spec_for_atlas(ATLAS)
    return _yeo17_spec_cache


def collapse_betas_to_target_space(beta):
    beta = np.asarray(beta, dtype=np.float32)
    if TARGET_BETA_SPACE == ATLAS.replace("schaefer", "parcel"):
        return beta

    spec = yeo17_spec()
    out_shape = beta.shape[:-1] + (spec["n_networks"],)
    collapsed = np.empty(out_shape, dtype=np.float32)
    for network_idx in range(spec["n_networks"]):
        mask = spec["parcel_to_network_idx"] == network_idx
        collapsed[..., network_idx] = beta[..., mask].mean(axis=-1)
    return collapsed


def expand_betas_for_eval(beta):
    beta = np.asarray(beta, dtype=np.float32)
    if TARGET_BETA_SPACE == ATLAS.replace("schaefer", "parcel"):
        return beta

    spec = yeo17_spec()
    return np.asarray(beta[..., spec["parcel_to_network_idx"]], dtype=np.float32)


def compute_condition_betas(h5_path, condition_group):
    cache_key = (h5_path, condition_group, ATLAS, TR, HCP_MAX_SUBJECTS)
    if cache_key in _condition_beta_cache:
        return _condition_beta_cache[cache_key]

    print(f"Computing {condition_group} betas: {os.path.basename(h5_path)}")
    hrf = spm_hrf(TR)

    with h5py.File(h5_path, "r") as f:
        condition_names = sorted(f[condition_group].keys())
        n_conditions = len(condition_names)
        run_ids = [x.decode() for x in f["long_subject_id"][:]]

        selected_subjects = None
        if "HCP_YA_" in os.path.basename(h5_path) and HCP_MAX_SUBJECTS is not None:
            selected_subjects = []
            for run_id in run_ids:
                subj = subject_key(run_id)
                if subj not in selected_subjects:
                    selected_subjects.append(subj)
                if len(selected_subjects) == HCP_MAX_SUBJECTS:
                    break
            selected_subjects = set(selected_subjects)
            print(f"Using first {len(selected_subjects)} HCP subjects")

        subj_sums = {}
        subj_counts = {}
        for run_idx, run_id in enumerate(run_ids):
            subj = subject_key(run_id)
            if selected_subjects is not None and subj not in selected_subjects:
                continue

            n_tp = int(f["valid_timepoints"][run_idx])
            y = np.asarray(f["timeseries"][ATLAS][run_idx, :, :n_tp], dtype=np.float32).T
            y_centered = y - y.mean(axis=0, keepdims=True)

            x = np.stack(
                [
                    np.asarray(f[condition_group][condition_name][run_idx, :n_tp], dtype=np.float32)
                    for condition_name in condition_names
                ],
                axis=0,
            )
            x = convolve_event_matrix(x, hrf)
            x = x - x.mean(axis=1, keepdims=True)

            denom = np.sum(x * x, axis=1)
            betas = np.zeros((n_conditions, y.shape[1]), dtype=np.float32)
            valid = denom > 0
            if valid.any():
                betas[valid] = x[valid] @ y_centered / denom[valid, None]

            if subj not in subj_sums:
                subj_sums[subj] = betas.copy()
                subj_counts[subj] = 1
            else:
                subj_sums[subj] += betas
                subj_counts[subj] += 1

    subjects = sorted(subj_sums)
    subject_betas = np.stack([subj_sums[s] / subj_counts[s] for s in subjects], axis=0)
    group_betas = subject_betas.mean(axis=0)
    result = {
        "condition_names": condition_names,
        "subjects": subjects,
        "subject_betas": subject_betas,
        "group_betas": group_betas,
    }
    _condition_beta_cache[cache_key] = result
    return result


def h5_specs():
    specs = []

    for path in sorted(glob.glob(f"{HCP_DIR}/*.h5")):
        task = path.split("HCP_YA_")[1].split("_lp")[0]
        specs.append({"path": path, "source": f"HCP/{task}"})

    for path in sorted(glob.glob(f"{NAKAI_DIR}/*.h5")):
        split = os.path.basename(path).replace(".h5", "").replace("nakai_", "")
        specs.append({"path": path, "source": f"Nakai/{split}"})

    for path in sorted(glob.glob(f"{IBC_DIR}/*.h5")):
        task = os.path.basename(path).replace(".h5", "")
        specs.append({"path": path, "source": f"IBC/{task}"})

    return specs


def load_event_rows_for_file(h5_path, source_name):
    assert EVENT_PAIRING_MODE in {"pair_by_variant_idx", "cartesian_product"}
    rows = []
    with h5py.File(h5_path, "r") as f:
        event_emb = f["embeddings"]["events"]
        desc = f.get("events_desc", {})

        event_names = sorted(set(event_emb["instruction"].keys()) & set(event_emb["sensory"].keys()))
        for event_name in event_names:
            instruction_vecs = np.asarray(event_emb["instruction"][event_name][:], dtype=np.float32)
            sensory_vecs = np.asarray(event_emb["sensory"][event_name][:], dtype=np.float32)

            instruction_phrases = []
            sensory_phrases = []
            if event_name in desc and "instruction" in desc[event_name]:
                instruction_phrases = decode_text_rows(desc[event_name]["instruction"])
            if event_name in desc and "sensory" in desc[event_name]:
                sensory_phrases = decode_text_rows(desc[event_name]["sensory"])

            if EVENT_PAIRING_MODE == "pair_by_variant_idx":
                n_rows = min(len(instruction_vecs), len(sensory_vecs))
                for variant_idx in range(n_rows):
                    rows.append(
                        {
                            "source": source_name,
                            "condition_name": event_name,
                            "condition_group": "events",
                            "feature": np.concatenate(
                                [sensory_vecs[variant_idx], instruction_vecs[variant_idx]],
                                axis=0,
                            ).astype(np.float32),
                            "sensory_variant_idx": variant_idx,
                            "instruction_variant_idx": variant_idx,
                            "sensory_phrase": text_at(sensory_phrases, variant_idx),
                            "instruction_phrase": text_at(instruction_phrases, variant_idx),
                        }
                    )
            else:
                for sensory_idx, sensory_vec in enumerate(sensory_vecs):
                    for instruction_idx, instruction_vec in enumerate(instruction_vecs):
                        rows.append(
                            {
                                "source": source_name,
                                "condition_name": event_name,
                                "condition_group": "events",
                                "feature": np.concatenate([sensory_vec, instruction_vec], axis=0).astype(np.float32),
                                "sensory_variant_idx": sensory_idx,
                                "instruction_variant_idx": instruction_idx,
                                "sensory_phrase": text_at(sensory_phrases, sensory_idx),
                                "instruction_phrase": text_at(instruction_phrases, instruction_idx),
                            }
                        )

    return rows


def load_response_rows_for_file(h5_path, source_name):
    rows = []
    with h5py.File(h5_path, "r") as f:
        response_emb = f["embeddings"].get("responses")
        if response_emb is None:
            return rows

        response_desc = f.get("response_desc", {})
        for response_name in sorted(response_emb.keys()):
            if not response_name.startswith("response_"):
                continue
            if response_name == "response_unknown":
                continue

            response_vecs = np.asarray(response_emb[response_name][:], dtype=np.float32)
            response_phrases = decode_text_rows(response_desc[response_name]) if response_name in response_desc else []
            for variant_idx, response_vec in enumerate(response_vecs):
                rows.append(
                    {
                        "source": source_name,
                        "condition_name": response_name,
                        "condition_group": "responses",
                        "feature": response_vec.astype(np.float32),
                        "response_variant_idx": variant_idx,
                        "response_phrase": text_at(response_phrases, variant_idx, default=response_name),
                    }
                )

    return rows


def build_condition_targets(beta_bundle):
    condition_targets = {}
    for condition_idx, condition_name in enumerate(beta_bundle["condition_names"]):
        raw_subject_betas = np.asarray(beta_bundle["subject_betas"][:, condition_idx], dtype=np.float32)
        raw_group_beta = np.asarray(beta_bundle["group_betas"][condition_idx], dtype=np.float32)
        target_subject_betas = collapse_betas_to_target_space(raw_subject_betas)
        target_group_beta = collapse_betas_to_target_space(raw_group_beta)
        condition_targets[condition_name] = {
            "subjects": list(beta_bundle["subjects"]),
            "target_subject_betas": np.asarray(target_subject_betas, dtype=np.float32),
            "target_group_beta": np.asarray(target_group_beta, dtype=np.float32),
            "eval_subject_betas": expand_betas_for_eval(target_subject_betas),
            "eval_group_beta": expand_betas_for_eval(target_group_beta),
        }
    return condition_targets


def condition_targets_at(dataset, row):
    return dataset["targets"][condition_group(row)]


def targets_for_level(targets, target_level):
    assert target_level in {"group", "subject"}, target_level
    if target_level == "group":
        return [("group", targets["target_group_beta"])]
    return list(zip(targets["subjects"], targets["target_subject_betas"]))


def eval_targets_for_level(targets, target_level):
    assert target_level in {"group", "subject"}, target_level
    if target_level == "group":
        return [("group", targets["eval_group_beta"])]
    return list(zip(targets["subjects"], targets["eval_subject_betas"]))


def build_training_view(dataset, target_level):
    x_rows = []
    y_rows = []
    meta_rows = []

    for row in dataset["rows"]:
        for subject_name, beta_vec in targets_for_level(condition_targets_at(dataset, row), target_level):
            x_rows.append(row["feature"])
            y_rows.append(np.asarray(beta_vec, dtype=np.float32))
            meta = dict(row)
            meta["subject"] = subject_name
            meta_rows.append(meta)

    return {
        "X": np.stack(x_rows).astype(np.float32),
        "Y": np.stack(y_rows).astype(np.float32),
        "meta": meta_rows,
    }


def build_eval_entries(dataset, target_level):
    grouped_rows = {}

    for row in dataset["rows"]:
        group = condition_group(row)
        for subject_name, beta_vec in eval_targets_for_level(condition_targets_at(dataset, row), target_level):
            eval_key = group if target_level == "group" else f"{group}::{subject_name}"
            entry = grouped_rows.setdefault(
                eval_key,
                {
                    "group": group,
                    "source": row["source"],
                    "condition_name": row["condition_name"],
                    "subject": subject_name,
                    "rows": [],
                    "y_true": np.asarray(beta_vec, dtype=np.float32),
                },
            )
            entry["rows"].append(row["feature"])

    return grouped_rows


def build_event_dataset(specs):
    dataset = {"rows": [], "targets": {}}
    for spec in specs:
        event_rows = load_event_rows_for_file(spec["path"], spec["source"])
        event_betas = compute_condition_betas(spec["path"], "events")
        dataset["rows"].extend(event_rows)
        for condition_name, targets in build_condition_targets(event_betas).items():
            dataset["targets"][f"{spec['source']}::{condition_name}"] = targets
    return dataset


def build_response_dataset(specs):
    dataset = {"rows": [], "targets": {}}
    for spec in specs:
        response_rows = load_response_rows_for_file(spec["path"], spec["source"])
        response_betas = compute_condition_betas(spec["path"], "responses")
        dataset["rows"].extend(response_rows)
        for condition_name, targets in build_condition_targets(response_betas).items():
            dataset["targets"][f"{spec['source']}::{condition_name}"] = targets
    return dataset


def metrics_from_vectors(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    y_true_norm = float(np.linalg.norm(y_true))
    y_pred_norm = float(np.linalg.norm(y_pred))
    cosine = np.nan if y_true_norm == 0.0 or y_pred_norm == 0.0 else float(np.dot(y_true, y_pred) / (y_true_norm * y_pred_norm))

    y_true_std = float(np.std(y_true))
    y_pred_std = float(np.std(y_pred))
    corr = np.nan if y_true_std == 0.0 or y_pred_std == 0.0 else float(np.corrcoef(y_true, y_pred)[0, 1])

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = np.nan if y_true_std <= R2_STD_FLOOR or ss_tot <= 1e-8 else 1.0 - ss_res / ss_tot
    return {
        "cosine": cosine,
        "corr": corr,
        "r2": r2,
        "y_true_std": y_true_std,
        "y_pred_std": y_pred_std,
        "y_true_norm": y_true_norm,
        "y_pred_norm": y_pred_norm,
        "max_abs_true": float(np.max(np.abs(y_true))),
        "max_abs_pred": float(np.max(np.abs(y_pred))),
        "ss_res": ss_res,
        "ss_tot": ss_tot,
    }


def condition_group(meta):
    return f"{meta['source']}::{meta['condition_name']}"


def condition_subject_key(meta):
    return f"{condition_group(meta)}::{meta['subject']}"


def l2_normalize_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def fit_model(x_train, y_train, alpha, pca_components):
    if USE_PCA_TARGETS:
        pca = PCA(n_components=pca_components)
        y_train_targets = pca.fit_transform(y_train)
        pca_component_count = int(pca_components)
    else:
        pca = None
        y_train_targets = y_train
        pca_component_count = 0
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x_train, y_train_targets)
    return {
        "estimator": "ridge",
        "model": model,
        "pca": pca,
        "pca_components": pca_component_count,
        "alpha": float(alpha),
        "nn_k": 0,
    }


def fit_nn_model(x_train, y_train, nn_k):
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    return {
        "estimator": "nn",
        "x_train": x_train,
        "x_train_norm": l2_normalize_rows(x_train.astype(np.float64)).astype(np.float32),
        "y_train": y_train,
        "pca": None,
        "pca_components": 0,
        "alpha": 0.0,
        "nn_k": int(nn_k),
    }


def predict_model(bundle, x):
    if bundle["estimator"] == "nn":
        x = np.asarray(x, dtype=np.float32)
        x_norm = x / max(float(np.linalg.norm(x)), 1e-12)
        sims = bundle["x_train_norm"] @ x_norm.astype(np.float32)
        k_eff = min(bundle["nn_k"], len(sims))
        topk_idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
        return np.asarray(bundle["y_train"][topk_idx].mean(axis=0), dtype=np.float32)

    y_pred_targets = bundle["model"].predict(x[None].astype(np.float32))
    if bundle["pca"] is None:
        return np.asarray(y_pred_targets[0], dtype=np.float32)
    y_pred = bundle["pca"].inverse_transform(y_pred_targets)[0]
    return np.asarray(y_pred, dtype=np.float32)


def expand_prediction_for_eval(y_pred):
    return expand_betas_for_eval(y_pred)


def aggregate_predictions(preds, mode):
    stacked = np.stack(preds, axis=0).astype(np.float32)
    if mode == "mean":
        return np.mean(stacked, axis=0).astype(np.float32)
    assert mode == "median", mode
    return np.median(stacked, axis=0).astype(np.float32)


def rounded_df(df, decimals=3):
    return df.round(decimals)


def normalize_paths(paths):
    return {os.path.normpath(path) for path in paths}


def print_r2_diagnostics(results_df):
    tiny_target_df = results_df[results_df["y_true_std"] <= R2_STD_FLOOR]
    huge_r2_df = results_df[np.isfinite(results_df["r2"]) & (np.abs(results_df["r2"]) >= R2_ABS_DIAGNOSTIC_THRESHOLD)]

    # print("\nR2 Diagnostics")
    # print(
    #     f"  tiny_target_rows={len(tiny_target_df)}"
    #     f"  huge_abs_r2_rows={len(huge_r2_df)}"
    #     f"  total_rows={len(results_df)}"
    # )

    if len(huge_r2_df) == 0:
        return

    cols = [
        "model_name",
        "estimator",
        "fold",
        "aggregate_mode",
        "alpha",
        "nn_k",
        "pca_components",
        "source",
        "condition_name",
        "subject",
        "r2",
        "corr",
        "cosine",
        "y_true_std",
        "y_pred_std",
        "y_true_norm",
        "y_pred_norm",
        "max_abs_true",
        "max_abs_pred",
        "ss_tot",
        "ss_res",
        "n_rows",
    ]
    worst_df = huge_r2_df.reindex(
        huge_r2_df["r2"].abs().sort_values(ascending=False).index
    ).head(DIAGNOSTIC_TOP_K)
    # print("\nWorst R2 Rows")
    # print(rounded_df(worst_df[cols]).to_string(index=False))


def evaluate_dataset(name, dataset, alphas, pca_components_grid, nn_k_values):
    train_view = build_training_view(dataset, TRAIN_TARGET_LEVEL)
    x_all = train_view["X"]
    y_all = train_view["Y"]
    groups = np.array([condition_group(meta) for meta in train_view["meta"]])
    grouped_rows = build_eval_entries(dataset, EVAL_TARGET_LEVEL)

    unique_groups = np.array(sorted(dataset["targets"]))
    assert len(unique_groups) >= CV_N_SPLITS, (name, len(unique_groups), CV_N_SPLITS)

    print(
        f"\n{name.upper()}",
        f"\n  samples={len(x_all)}",
        f"\n  eval_units={len(grouped_rows)}",
        f"\n  unique_conditions={len(unique_groups)}",
        f"\n  x_dim={x_all.shape[1]}",
        f"\n  target_y_dim={y_all.shape[1]}",
        f"\n  eval_y_dim={len(next(iter(grouped_rows.values()))['y_true'])}",
        f"\n  train_target_level={TRAIN_TARGET_LEVEL}",
        f"\n  eval_target_level={EVAL_TARGET_LEVEL}",
        f"\n  target_beta_space={TARGET_BETA_SPACE}",
        f"\n  use_pca_targets={USE_PCA_TARGETS}",
        f"\n  estimators={','.join(ESTIMATOR_TYPES)}",
        f"\n  pairing_mode={EVENT_PAIRING_MODE if name == 'event' else 'response_only'}",
    )

    splitter = GroupKFold(n_splits=CV_N_SPLITS)
    rows = []
    effective_pca_components_grid = pca_components_grid if USE_PCA_TARGETS else [None]
    for fold_idx, (train_group_idx, test_group_idx) in enumerate(
        splitter.split(unique_groups, groups=unique_groups),
        start=1,
    ):
        train_groups = set(unique_groups[train_group_idx])
        test_groups = set(unique_groups[test_group_idx])
        train_mask = np.isin(groups, list(train_groups))
        print(
            f"  fold {fold_idx}/{CV_N_SPLITS}: "
            f"train_samples={int(train_mask.sum())} "
            f"test_conditions={len(test_groups)}"
        )

        test_eval_keys = [eval_key for eval_key, entry in grouped_rows.items() if entry["group"] in test_groups]
        model_bundles = []
        if "ridge" in ESTIMATOR_TYPES:
            for pca_components in effective_pca_components_grid:
                for alpha in alphas:
                    model_bundles.append(
                        fit_model(
                            x_all[train_mask],
                            y_all[train_mask],
                            alpha=alpha,
                            pca_components=pca_components,
                        )
                    )
        if "nn" in ESTIMATOR_TYPES:
            for nn_k in nn_k_values:
                model_bundles.append(
                    fit_nn_model(
                        x_all[train_mask],
                        y_all[train_mask],
                        nn_k=nn_k,
                    )
                )

        for model in model_bundles:
            for eval_key in test_eval_keys:
                entry = grouped_rows[eval_key]
                preds = [predict_model(model, row) for row in entry["rows"]]
                for aggregate_mode in ("mean", "median"):
                    y_pred = expand_prediction_for_eval(aggregate_predictions(preds, aggregate_mode))
                    rows.append(
                        {
                            "model_name": name,
                            "estimator": model["estimator"],
                            "fold": fold_idx,
                            "alpha": model["alpha"],
                            "nn_k": model["nn_k"],
                            "pca_components": model["pca_components"],
                            "aggregate_mode": aggregate_mode,
                            "source": entry["source"],
                            "condition_name": entry["condition_name"],
                            "subject": entry["subject"],
                            "n_rows": len(entry["rows"]),
                            **metrics_from_vectors(entry["y_true"], y_pred),
                        }
                    )

    results_df = pd.DataFrame(rows)
    print_r2_diagnostics(results_df)
    fold_summary_df = (
        results_df.groupby(
            ["model_name", "estimator", "aggregate_mode", "alpha", "nn_k", "pca_components", "fold"],
            as_index=False,
        )[["corr", "cosine", "r2"]]
        .mean()
    )
    summary_df = (
        fold_summary_df.groupby(
            ["model_name", "estimator", "aggregate_mode", "alpha", "nn_k", "pca_components"],
            as_index=False,
        )
        .agg(
            corr_mean=("corr", "mean"),
            corr_std=("corr", "std"),
            cosine_mean=("cosine", "mean"),
            cosine_std=("cosine", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .sort_values(
            ["aggregate_mode", "corr_mean", "cosine_mean", "r2_mean"],
            ascending=[True, False, False, False],
        )
        .reset_index(drop=True)
    )

    best_rows = []
    for metric in ("corr", "cosine", "r2"):
        for aggregate_mode in ("mean", "median"):
            best = (
                summary_df[summary_df["aggregate_mode"] == aggregate_mode]
                .sort_values(f"{metric}_mean", ascending=False)
                .head(1)
                .copy()
            )
            best["selected_by"] = metric
            best_rows.append(best)
    best_df = pd.concat(best_rows, ignore_index=True)

    print("\nSummary")
    print(rounded_df(summary_df).to_string(index=False))
    print("\nBest By Metric")
    print(
        rounded_df(
            best_df[
                [
                    "selected_by",
                    "estimator",
                    "aggregate_mode",
                    "model_name",
                    "alpha",
                    "nn_k",
                    "pca_components",
                    "corr_mean",
                    "corr_std",
                    "cosine_mean",
                    "cosine_std",
                    "r2_mean",
                    "r2_std",
                ]
            ]
        ).to_string(index=False)
    )

    return results_df, summary_df, best_df


def fit_and_save_event_ridge(specs):
    holdout_paths = normalize_paths(FIT_SAVE_EVENT_HOLDOUT_PATHS)
    train_specs = [spec for spec in specs if os.path.normpath(spec["path"]) not in holdout_paths]
    print(
        "Fitting saved event ridge",
        f"\n  total_files={len(specs)}",
        f"\n  holdout_files={len(specs) - len(train_specs)}",
        f"\n  train_files={len(train_specs)}",
        f"\n  alpha={FIT_SAVE_EVENT_RIDGE_ALPHA}",
        f"\n  aggregate_mode={FIT_SAVE_EVENT_AGGREGATE_MODE}",
        f"\n  train_target_level={TRAIN_TARGET_LEVEL}",
        f"\n  use_pca_targets={USE_PCA_TARGETS}",
    )

    dataset = build_event_dataset(train_specs)
    train_view = build_training_view(dataset, TRAIN_TARGET_LEVEL)
    pca_components = EVENT_PCA_COMPONENTS[0] if USE_PCA_TARGETS else None
    bundle = fit_model(
        train_view["X"],
        train_view["Y"],
        alpha=FIT_SAVE_EVENT_RIDGE_ALPHA,
        pca_components=pca_components,
    )

    os.makedirs(FIT_SAVE_EVENT_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(FIT_SAVE_EVENT_OUTPUT_DIR, FIT_SAVE_EVENT_OUTPUT_NAME)
    artifact = {
        "model_name": "event",
        "estimator": "ridge",
        "aggregate_mode": FIT_SAVE_EVENT_AGGREGATE_MODE,
        "alpha": float(FIT_SAVE_EVENT_RIDGE_ALPHA),
        "nn_k": 0,
        "pca_components": bundle["pca_components"],
        "train_target_level": TRAIN_TARGET_LEVEL,
        "pairing_mode": EVENT_PAIRING_MODE,
        "target_beta_space": TARGET_BETA_SPACE,
        "use_pca_targets": USE_PCA_TARGETS,
        "x_dim": int(train_view["X"].shape[1]),
        "y_dim": int(train_view["Y"].shape[1]),
        "eval_y_dim": int(expand_betas_for_eval(train_view["Y"][0]).shape[0]),
        "train_files": [spec["path"] for spec in train_specs],
        "holdout_files": sorted(holdout_paths),
        "bundle": bundle,
    }
    if TARGET_BETA_SPACE == "yeo17":
        spec = yeo17_spec()
        artifact["yeo17_network_names"] = spec["network_names"]
        artifact["parcel_to_network_idx"] = spec["parcel_to_network_idx"]
    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved event ridge to {output_path}")


def main():
    specs = h5_specs()
    if RUN_MODE == "fit_save_event_ridge":
        fit_and_save_event_ridge(specs)
        return

    assert RUN_MODE == "cv", RUN_MODE
    print("Building datasets from", len(specs), "files")

    event_dataset = build_event_dataset(specs)
    response_dataset = build_response_dataset(specs)

    event_results, event_summary, event_best = evaluate_dataset(
        name="event",
        dataset=event_dataset,
        alphas=EVENT_RIDGE_ALPHAS,
        pca_components_grid=EVENT_PCA_COMPONENTS,
        nn_k_values=EVENT_NN_K_VALUES,
    )
    response_results, response_summary, response_best = evaluate_dataset(
        name="response",
        dataset=response_dataset,
        alphas=RESPONSE_RIDGE_ALPHAS,
        pca_components_grid=RESPONSE_PCA_COMPONENTS,
        nn_k_values=RESPONSE_NN_K_VALUES,
    )

    print("\nFinal Overview")
    print("Event best:")
    print(
        rounded_df(
            event_best[
                [
                    "selected_by",
                    "estimator",
                    "aggregate_mode",
                    "alpha",
                    "nn_k",
                    "pca_components",
                    "corr_mean",
                    "corr_std",
                    "cosine_mean",
                    "cosine_std",
                    "r2_mean",
                    "r2_std",
                ]
            ]
        ).to_string(index=False)
    )
    print("\nResponse best:")
    print(
        rounded_df(
            response_best[
                [
                    "selected_by",
                    "estimator",
                    "aggregate_mode",
                    "alpha",
                    "nn_k",
                    "pca_components",
                    "corr_mean",
                    "corr_std",
                    "cosine_mean",
                    "cosine_std",
                    "r2_mean",
                    "r2_std",
                ]
            ]
        ).to_string(index=False)
    )

    print("\nEvent results rows:", len(event_results))
    print("Response results rows:", len(response_results))


if __name__ == "__main__":
    main()
