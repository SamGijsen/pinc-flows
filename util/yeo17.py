from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


SCHAEFER_ORDER_DIR = Path(
    "/TODO_SET_PATH/order"
)


def _load_schaefer_network_spec(tsv_path, column):
    df = pd.read_csv(tsv_path, sep="\t").sort_values("index")

    network_names = []
    network_name_to_idx = {}
    parcel_to_network_idx = []
    for network_name in df[column]:
        if network_name not in network_name_to_idx:
            network_name_to_idx[network_name] = len(network_names)
            network_names.append(network_name)
        parcel_to_network_idx.append(network_name_to_idx[network_name])

    parcel_to_network_idx = np.asarray(parcel_to_network_idx, dtype=np.int64)
    return {
        "network_names": network_names,
        "parcel_to_network_idx": parcel_to_network_idx,
        "n_networks": len(network_names),
        "n_parcels": len(parcel_to_network_idx),
    }


@lru_cache(maxsize=None)
def load_schaefer_yeo7_spec(tsv_path):
    spec = _load_schaefer_network_spec(tsv_path, "network_label")
    assert spec["n_networks"] == 7
    return spec


@lru_cache(maxsize=None)
def load_schaefer_yeo17_spec(tsv_path):
    spec = _load_schaefer_network_spec(tsv_path, "network_label_17network")
    assert spec["n_networks"] == 17
    return spec


def _schaefer_order_path(atlas_name):
    atlas_name = str(atlas_name)
    assert atlas_name.startswith("schaefer"), atlas_name
    parcel_count = int(atlas_name[len("schaefer"):])
    return SCHAEFER_ORDER_DIR / (
        f"atlas-Schaefer2018v0143_desc-{parcel_count}ParcelsAllNetworks_dseg.tsv"
    )


@lru_cache(maxsize=None)
def load_schaefer_yeo7_spec_for_atlas(atlas_name):
    return load_schaefer_yeo7_spec(str(_schaefer_order_path(atlas_name)))


@lru_cache(maxsize=None)
def load_schaefer_yeo17_spec_for_atlas(atlas_name):
    return load_schaefer_yeo17_spec(str(_schaefer_order_path(atlas_name)))


def load_schaefer400_yeo7_spec(tsv_path):
    return load_schaefer_yeo7_spec(tsv_path)


def load_schaefer400_yeo17_spec(tsv_path):
    return load_schaefer_yeo17_spec(tsv_path)
