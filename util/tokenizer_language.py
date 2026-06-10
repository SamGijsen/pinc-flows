import numpy as np


def _l2_normalize_rows_np(x, eps=1e-8):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def _as_language_embedding_rows_np(x, embedding_dim=None):
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embedding array [N, D], got shape {arr.shape}")
    if embedding_dim is not None:
        if arr.shape[1] < int(embedding_dim):
            raise ValueError(
                f"Embedding dim {arr.shape[1]} is smaller than requested truncate dim {embedding_dim}"
            )
        arr = arr[:, :int(embedding_dim)]
    return arr.astype(np.float32, copy=True)


def compute_language_embedding_mean_np(embedding_sets):
    parts = [_as_language_embedding_rows_np(x) for x in embedding_sets]
    if len(parts) == 0:
        raise ValueError("embedding_sets must be non-empty")
    dim = parts[0].shape[1]
    for part in parts[1:]:
        if part.shape[1] != dim:
            raise ValueError(f"Embedding dims must match, got {dim} and {part.shape[1]}")
    return np.concatenate(parts, axis=0).mean(axis=0, keepdims=True).astype(np.float32, copy=False)


def mean_center_l2_normalize_rows_np(x, mean):
    x_rows = _as_language_embedding_rows_np(x)
    mean_arr = np.asarray(mean, dtype=np.float32)
    if mean_arr.ndim == 1:
        mean_arr = mean_arr[None, :]
    if mean_arr.shape != (1, x_rows.shape[1]):
        raise ValueError(
            f"Mean must have shape (1, {x_rows.shape[1]}) or ({x_rows.shape[1]},), got {mean_arr.shape}"
        )
    return _l2_normalize_rows_np(x_rows - mean_arr)


def collect_condition_cont_language_event_cfgs(data_cfg):
    cfgs = []
    for key in (
        'condition_cont_language_events',
        'train_condition_cont_language_events',
        'val_condition_cont_language_events',
    ):
        cfg = data_cfg.get(key)
        if cfg is not None:
            cfgs.append(cfg)

    for split in ('train', 'val'):
        specs = data_cfg.get(f'{split}_datasets')
        if not isinstance(specs, (list, tuple)):
            continue
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            cfg = spec.get('condition_cont_language_events')
            if cfg is not None:
                cfgs.append(cfg)
    return cfgs


def _to_path_list(value, context_name):
    if isinstance(value, str):
        if len(value.strip()) == 0:
            raise ValueError(f"{context_name} cannot be empty")
        return [value]
    if isinstance(value, (list, tuple)):
        paths = []
        for i, path in enumerate(value):
            if not isinstance(path, str) or len(path.strip()) == 0:
                raise TypeError(f"{context_name}[{i}] must be a non-empty string")
            paths.append(path)
        if len(paths) == 0:
            raise ValueError(f"{context_name} cannot be empty")
        return paths
    raise TypeError(f"{context_name} must be a path string or list/tuple of path strings")


def _flatten_embedding_rows(obj, embedding_dim):
    rows = []
    if isinstance(obj, dict):
        for value in obj.values():
            rows.extend(_flatten_embedding_rows(value, embedding_dim))
        return rows
    rows.append(_as_language_embedding_rows_np(obj, embedding_dim))
    return rows


def normalize_roi_embeddings_shared_mean_from_condition_cfgs(
    roi_embeddings,
    model_embedding_dim,
    condition_cont_language_event_cfgs,
):
    """
    Normalize ROI embeddings with the same shared_mean_l2 formulation used by language-event conditioning.

    Shapes:
      roi_embeddings: [R, D_raw] -> returns [R, model_embedding_dim]
    """
    roi_rows = _as_language_embedding_rows_np(roi_embeddings, model_embedding_dim)
    mean_inputs = []
    path_cache = {}
    included_source_paths = set()
    included_task_pool_paths = set()

    def _load_rows_from_path(path):
        if path not in path_cache:
            path_cache[path] = _as_language_embedding_rows_np(
                np.load(path, allow_pickle=True), model_embedding_dim
            )
        return path_cache[path]

    for cfg in condition_cont_language_event_cfgs:
        if not isinstance(cfg, dict):
            raise TypeError("condition_cont_language_events must be a dict")
        norm_cfg = cfg.get('normalization')
        if not isinstance(norm_cfg, dict):
            raise TypeError("condition_cont_language_events.normalization must be a dict")
        if norm_cfg.get('mode') != 'shared_mean_l2':
            raise ValueError(
                "one_roi_one_token requires condition_cont_language_events.normalization.mode='shared_mean_l2'"
            )

        roi_paths = _to_path_list(norm_cfg['roi_embeddings_path'], "normalization.roi_embeddings_path")
        term_paths = _to_path_list(norm_cfg['term_embeddings_path'], "normalization.term_embeddings_path")
        for path in roi_paths:
            if path not in included_source_paths:
                mean_inputs.append(_load_rows_from_path(path))
                included_source_paths.add(path)
        for path in term_paths:
            if path not in included_source_paths:
                mean_inputs.append(_load_rows_from_path(path))
                included_source_paths.add(path)

        if bool(norm_cfg.get('include_task_embeddings_in_mean', True)):
            task_path = cfg['embeddings_path']
            if task_path not in included_task_pool_paths:
                task_obj = np.load(task_path, allow_pickle=True)
                if isinstance(task_obj, np.ndarray) and task_obj.shape == ():
                    task_obj = task_obj.item()
                mean_inputs.extend(_flatten_embedding_rows(task_obj, model_embedding_dim))
                included_task_pool_paths.add(task_path)

    if len(mean_inputs) == 0:
        raise ValueError(
            "one_roi_one_token requires at least one shared_mean_l2 condition_cont_language_events config"
        )
    mean = compute_language_embedding_mean_np(mean_inputs)
    return mean_center_l2_normalize_rows_np(roi_rows, mean)
