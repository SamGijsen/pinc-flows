from pathlib import Path

import numpy as np


def make_stats_key(*parts):
    return "__".join(str(part) for part in parts if part not in (None, ""))


def _coerce_stats_value(value):
    if isinstance(value, Path):
        return np.asarray(str(value), dtype=str)
    if isinstance(value, str):
        return np.asarray(value, dtype=str)

    arr = np.asarray(value)
    if arr.dtype != object:
        return arr

    flat = arr.reshape(-1).tolist()
    if all(isinstance(item, (str, Path)) for item in flat):
        return np.asarray([str(item) for item in flat], dtype=str).reshape(arr.shape)

    raise TypeError(f"stats bundle value for type {type(value)!r} produced an object array")


def add_stats_value(bundle, *parts, value):
    key = make_stats_key(*parts)
    assert key not in bundle, f"duplicate stats bundle key: {key}"
    bundle[key] = _coerce_stats_value(value)


def add_stats_dataframe(bundle, *parts, frame):
    add_stats_value(bundle, *parts, "columns", value=np.asarray(frame.columns, dtype=str))
    for column in frame.columns:
        add_stats_value(bundle, *parts, str(column), value=frame[column].to_numpy())


def _decode_loaded_value(value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def load_hcp_stats(path):
    nested = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            cursor = nested
            parts = key.split("__")
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = _decode_loaded_value(data[key])
    return nested
