import h5py
import numpy as np

H5_PATH = "/TODO_SET_PATH/ukbb_1hz_braincontrol_rest.h5"
ATLAS_NAME = "schaefer400"
EVENT_NAME = "rest"
EMB_DIM = 1024
BASE_RELEVANCE_GROUPS = (
    f"relevance_v2/{ATLAS_NAME}",
    f"relevance_p99_clip/{ATLAS_NAME}",
)
OOD_ROI_RELEVANCE_GROUPS = (
    f"relevance_ood_rescaled_roi/{ATLAS_NAME}",
    f"relevance_ood_7fold_roi/{ATLAS_NAME}",
)
OOD_NETWORK_RELEVANCE_GROUPS = (
    f"relevance_ood_rescaled_yeo17/{ATLAS_NAME}",
    f"relevance_ood_7fold_yeo17/{ATLAS_NAME}",
)

def upsert_dataset(f, path, data):
    if path in f:
        del f[path]
    parent, name = path.rsplit("/", 1)
    f.require_group(parent)
    f[parent].create_dataset(name, data=data)

def upsert_relevance_group(f, group, event_names, response_names, data):
    f.require_group(group)
    f.require_group(f"{group}/events")
    f.require_group(f"{group}/responses")
    for event_name in event_names:
        upsert_dataset(f, f"{group}/events/{event_name}", data)
    for response_name in response_names:
        upsert_dataset(f, f"{group}/responses/{response_name}", data)

with h5py.File(H5_PATH, "r+") as f:
    n_runs = int(f["long_subject_id"].shape[0])
    t_len = int(f[f"timeseries/{ATLAS_NAME}"].shape[-1])
    num_rois = int(f[f"timeseries/{ATLAS_NAME}"].shape[1])

    zeros_rt = np.zeros((n_runs, t_len), dtype=np.float32)
    zeros_emb = np.zeros((1, EMB_DIM), dtype=np.float32)
    zeros_rel = np.zeros((n_runs, num_rois), dtype=np.float32)
    zeros_ood_rel = np.zeros((1, num_rois), dtype=np.float32)

    upsert_dataset(f, f"events/{EVENT_NAME}", zeros_rt)
    upsert_dataset(f, "responses/no_response", zeros_rt)
    upsert_dataset(f, "responses/response_unknown", zeros_rt)

    upsert_dataset(f, f"embeddings/events/instruction/{EVENT_NAME}", zeros_emb)
    upsert_dataset(f, f"embeddings/events/sensory/{EVENT_NAME}", zeros_emb)

    f.require_group("embeddings/responses")
    event_names = sorted(f["events"].keys())
    response_names = sorted(f["responses"].keys())
    for relevance_group in BASE_RELEVANCE_GROUPS:
        upsert_relevance_group(f, relevance_group, event_names, response_names, zeros_rel)
    for relevance_group in OOD_ROI_RELEVANCE_GROUPS:
        upsert_relevance_group(f, relevance_group, event_names, response_names, zeros_ood_rel)
    for relevance_group in OOD_NETWORK_RELEVANCE_GROUPS:
        upsert_relevance_group(f, relevance_group, event_names, response_names, zeros_ood_rel)
