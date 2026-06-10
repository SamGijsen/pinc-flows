from pathlib import Path

import numpy as np
import pickle


def load_relevance_input_config(eval_cfg):
    cfg = eval_cfg.get("relevance_input")
    if cfg is None:
        return {"default": "true_roi", "grid": [], "predict": None, "ood_roi_h5_group": None}

    assert isinstance(cfg, dict)
    default = str(cfg.get("default", "true_roi")).lower()
    grid = [str(mode).lower() for mode in cfg.get("grid", [])]
    for mode in [default] + grid:
        assert mode in {"h5", "true_roi", "mask", "predict", "ood", "ood_roi"}, mode

    unique_grid = []
    for mode in grid:
        if mode != default and mode not in unique_grid:
            unique_grid.append(mode)

    requested_modes = {normalize_relevance_input_mode(mode) for mode in [default] + unique_grid}
    ood_roi_h5_group = cfg.get("ood_roi_h5_group")
    if "ood_roi" in requested_modes:
        assert isinstance(ood_roi_h5_group, str) and ood_roi_h5_group.strip(), (
            "evaluation.hcp.relevance_input.ood_roi_h5_group is required when requesting ood_roi"
        )

    predict = None
    if "predict" in [default] + unique_grid:
        predict_cfg = cfg["predict"]
        assert isinstance(predict_cfg, dict)
        with open(Path(predict_cfg["event_beta_ridge_model_path"]), "rb") as f:
            artifact = pickle.load(f)
        bundle = artifact["bundle"]
        assert bundle["estimator"] == "ridge"
        assert bundle["pca"] is None, "HCP relevance predict requires an event ridge artifact with USE_PCA_TARGETS=False"
        event_model = bundle["model"]
        event_target_beta_space = str(artifact.get("target_beta_space", "parcel400"))
        assert event_target_beta_space in {"parcel400", "yeo17"}
        event_parcel_to_network_idx = artifact.get("parcel_to_network_idx")
        if event_target_beta_space == "yeo17":
            assert event_parcel_to_network_idx is not None
            event_parcel_to_network_idx = np.asarray(event_parcel_to_network_idx, dtype=np.int64)
        nn_k = int(predict_cfg.get("response_beta_nn_k", 10))
        assert nn_k > 0
        predict = {
            "event_model": event_model,
            "event_target_beta_space": event_target_beta_space,
            "event_parcel_to_network_idx": event_parcel_to_network_idx,
            "response_nn_k": nn_k,
            **_build_response_nn_bank(),
        }

    return {
        "default": default,
        "grid": unique_grid,
        "predict": predict,
        "ood_roi_h5_group": ood_roi_h5_group,
    }


def normalize_relevance_input_mode(mode):
    mode = str(mode).lower()
    if mode == "h5":
        return "true_roi"
    if mode == "ood":
        return "ood_roi"
    return mode


def get_requested_relevance_modes(relevance_input_cfg):
    modes = {normalize_relevance_input_mode(relevance_input_cfg["default"])}
    for mode in relevance_input_cfg["grid"]:
        modes.add(normalize_relevance_input_mode(mode))
    return modes


def resolve_relevance_input(
    mode,
    task,
    condition_cont,
    condition_mode,
    relevance_scores,
    predict_cfg,
    num_rois,
    relevance_scores_ood_roi=None,
):
    mode = normalize_relevance_input_mode(mode)
    if mode == "true_roi":
        return relevance_scores, None, 0

    batch, steps = condition_cont.shape[:2]
    if mode == "mask":
        scores = np.zeros((batch, steps, int(num_rois)), dtype=np.float32)
        drop_mask = np.ones((batch, steps), dtype=bool)
        return scores, drop_mask, 2

    if mode == "ood_roi":
        assert relevance_scores_ood_roi is not None, (
            "relevance_input mode='ood_roi' requires evaluation.hcp.relevance_input.ood_roi_h5_group"
        )
        return relevance_scores_ood_roi, None, 0

    assert mode == "predict", mode
    assert predict_cfg is not None
    return _predict_relevance_scores(task, condition_cont, condition_mode, predict_cfg, num_rois), None, 0


def _build_response_nn_bank():
    from eval.text_embedding_kfold import (
        build_condition_targets,
        compute_condition_betas,
        h5_specs,
        l2_normalize_rows,
        load_response_rows_for_file,
    )

    features = []
    targets = []
    sources = []
    for spec in h5_specs():
        response_rows = load_response_rows_for_file(spec["path"], spec["source"])
        condition_targets = build_condition_targets(compute_condition_betas(spec["path"], "responses"))
        for row in response_rows:
            features.append(np.asarray(row["feature"], dtype=np.float32))
            targets.append(np.asarray(condition_targets[row["condition_name"]]["eval_group_beta"], dtype=np.float32))
            sources.append(str(row["source"]))

    x = np.stack(features).astype(np.float32)
    y = np.stack(targets).astype(np.float32)
    return {
        "response_features_norm": l2_normalize_rows(x.astype(np.float64)).astype(np.float32),
        "response_targets": y,
        "response_sources": np.asarray(sources, dtype=object),
    }


def _predict_relevance_scores(task, condition_cont, condition_mode, predict_cfg, num_rois):
    assert condition_cont.ndim == 3
    cond_dim = int(condition_cont.shape[-1])
    assert (cond_dim - 2) % 3 == 0
    piece_dim = (cond_dim - 2) // 3
    batch, steps = condition_cont.shape[:2]

    instruction = condition_cont[..., :piece_dim]
    sensory = condition_cont[..., piece_dim:2 * piece_dim]
    response = condition_cont[..., 2 * piece_dim:3 * piece_dim]
    response_special = condition_cont[..., 3 * piece_dim:]
    assert response_special.shape[-1] == 2

    event_x = np.concatenate([sensory, instruction], axis=-1).reshape(batch * steps, 2 * piece_dim)
    event_beta = np.asarray(predict_cfg["event_model"].predict(event_x), dtype=np.float32)
    event_beta = _expand_event_beta(event_beta, predict_cfg, int(num_rois))
    event_beta = event_beta.reshape(batch, steps, num_rois)

    response_beta = _predict_response_beta(
        task,
        response.reshape(batch * steps, piece_dim),
        response_special.reshape(batch * steps, 2),
        predict_cfg,
        num_rois,
    ).reshape(batch, steps, num_rois)

    total = event_beta + response_beta
    if condition_mode is not None:
        total = total * condition_mode[..., None].astype(np.float32)
    return total.astype(np.float32)


def _expand_event_beta(event_beta, predict_cfg, num_rois):
    target_beta_space = predict_cfg["event_target_beta_space"]
    if target_beta_space == "parcel400":
        assert event_beta.shape == (event_beta.shape[0], num_rois)
        return event_beta

    assert target_beta_space == "yeo17"
    parcel_to_network_idx = predict_cfg["event_parcel_to_network_idx"]
    assert parcel_to_network_idx.shape == (num_rois,)
    assert event_beta.shape[1] == int(parcel_to_network_idx.max()) + 1
    return np.asarray(event_beta[:, parcel_to_network_idx], dtype=np.float32)


def _predict_response_beta(task, response_x, response_special, predict_cfg, num_rois):
    out = np.zeros((response_x.shape[0], int(num_rois)), dtype=np.float32)
    active = np.logical_and(response_special[:, 0] <= 0.0, response_special[:, 1] <= 0.0)
    if not np.any(active):
        return out

    source_keep = np.logical_and(
        predict_cfg["response_sources"] != f"HCP/{task}",
        predict_cfg["response_sources"] != f"IBC/{task}",
    )
    assert np.any(source_keep)
    bank_x = predict_cfg["response_features_norm"][source_keep]
    bank_y = predict_cfg["response_targets"][source_keep]

    query_x = response_x[active].astype(np.float64)
    query_x = query_x / np.maximum(np.linalg.norm(query_x, axis=1, keepdims=True), 1e-12)
    sims = query_x.astype(np.float32) @ bank_x.T
    k = min(int(predict_cfg["response_nn_k"]), bank_x.shape[0])
    topk = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    out[active] = bank_y[topk].mean(axis=1)
    return out
