"""Add the released caption embeddings (and caption text) to a user-built H5.

Download the captions once:

    hf download SamGijsen/pinc-flows captions/captions.h5 --local-dir .

then, for each H5 you built:

    python util/add_captions_to_h5.py --captions captions/captions.h5 --task ibc/Discount --h5 /path/to/Discount.h5

Tasks are named `<dataset>/<task>`: HCP and IBC tasks as in
fold_to_holdout_mapping.yaml (e.g. `hcp/EMOTION`, `ibc/Discount`); all Nakai
files use `nakai/all`. Existing embeddings/ and *_desc groups in the H5 are
replaced.
"""
import argparse
from pathlib import Path

import h5py

GROUPS = ("embeddings", "events_desc", "events_desc_long", "response_desc")
UNCAPTIONED_RESPONSES = {"no_response", "response_unknown"}

parser = argparse.ArgumentParser()
parser.add_argument("--captions", type=Path, required=True)
parser.add_argument("--task", required=True, help="e.g. hcp/EMOTION, ibc/Discount, nakai/all")
parser.add_argument("--h5", type=Path, required=True)
args = parser.parse_args()

with h5py.File(args.captions, "r") as captions, h5py.File(args.h5, "r+") as f:
    if args.task not in captions:
        available = [f"{d}/{t}" for d in captions for t in captions[d]]
        raise KeyError(f"No captions for '{args.task}'. Available: {available}")
    source = captions[args.task]

    events = set(f["events"])
    responses = set(f["responses"]) - UNCAPTIONED_RESPONSES
    caption_events = set(source["embeddings/events/sensory"])
    caption_responses = set(source["embeddings/responses"])
    if events - caption_events or responses - caption_responses:
        raise ValueError(
            f"{args.h5} has events/responses without captions: "
            f"events={sorted(events - caption_events)}, responses={sorted(responses - caption_responses)}"
        )

    for name in GROUPS:
        if name in f:
            del f[name]
        f.copy(source[name], f, name=name)

print(f"Added captions for {len(events)} events and {len(responses)} responses from '{args.task}' to {args.h5}")
