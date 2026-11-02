"""Build HCP task event and response timecourses for the pinc-flows H5 schema.

Timecourses are on a 1 s grid: each value is the fraction of that second
occupied by the event. Inputs are the HCP EV files, movement regressors (for the
run length) and E-Prime TAB files:

    <hcp_root>/<subject>/MNINonLinear/Results/tfMRI_<TASK>_<LR|RL>/EVs/
    <hcp_root>/<subject>/MNINonLinear/Results/tfMRI_<TASK>_<LR|RL>/Movement_Regressors.txt
    <hcp_root>/<subject>/unprocessed/3T/tfMRI_<TASK>_<LR|RL>/LINKED_DATA/EPRIME/*TAB.txt

Fill events/ and responses/ in an H5 that already has long_subject_id
(e.g. "100206_LR") and valid_timepoints:

    python util/hcp_events.py --hcp-root /path/to/HCP_1200 --task GAMBLING --h5 /path/to/gambling.h5
"""
import argparse
import glob
import os

import h5py
import numpy as np
import pandas as pd

HCP_TR = 0.72
TASKS = ("EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")


def occupancy(intervals, n):
    out = np.zeros(n, dtype=np.float64)
    for start, end in intervals:
        start, end = max(start, 0.0), min(end, n)
        for i in range(int(np.floor(start)), int(np.ceil(end))):
            out[i] += max(0.0, min(i + 1, end) - max(i, start))
    return np.clip(out, 0.0, 1.0)


def impulses(times, n):
    out = np.zeros(n, dtype=np.float64)
    for t in times:
        if not np.isfinite(t) or t < 0 or t >= n:
            continue
        lo = int(t)
        out[lo] += 1.0 - (t - lo)
        if lo + 1 < n:
            out[lo + 1] += t - lo
    return np.clip(out, 0.0, 1.0)


def read_tab(hcp_root, subject, run, pattern="*TAB.txt"):
    eprime_dir = os.path.join(hcp_root, subject, "unprocessed/3T", run, "LINKED_DATA/EPRIME")
    (path,) = glob.glob(os.path.join(eprime_dir, pattern))
    return pd.read_csv(path, sep="\t")


def ev_onsets(evs_dir, name):
    return np.atleast_2d(np.loadtxt(os.path.join(evs_dir, name)))[:, 0]


def seconds(df, column, sync_ms):
    return (df[column].astype(float).to_numpy() - sync_ms) / 1000.0


def intervals(df, start, end, sync_ms):
    return list(zip(seconds(df, start, sync_ms), seconds(df, end, sync_ms)))


def next_after(t, starts, run_end):
    later = [s for s in starts if s > t]
    return min(later) if later else run_end


def emotion(evs_dir, tab, n, run_end):
    sync = tab["SyncSlide.OnsetTime"].dropna().iloc[0]
    shape_cues = seconds(tab[tab["Procedure"] == "ShapePromptPROC"], "shape.OnsetTime", sync)
    face_cues = seconds(tab[tab["Procedure"] == "FacePromptPROC"], "face.OnsetTime", sync)
    trials = tab[tab["Procedure"] == "TrialsPROC"]
    stim = seconds(trials, "StimSlide.OnsetTime", sync)
    fix = seconds(trials, "Fixation.OnsetTime", sync)
    rt = trials["StimSlide.RT"].astype(float).to_numpy()

    cues = sorted([(t, "shape") for t in shape_cues] + [(t, "face") for t in face_cues])
    cue_starts = [t for t, _ in cues]
    block = np.searchsorted(cue_starts, stim, side="right") - 1

    events = {"cue_face": [], "cue_shape": [], "stim_face": [], "stim_shape": [], "fixation": []}
    for b, (cue_start, kind) in enumerate(cues):
        idx = np.flatnonzero(block == b)
        events[f"cue_{kind}"].append((cue_start, stim[idx[0]]))
        block_end = cue_starts[b + 1] if b + 1 < len(cues) else run_end
        for j, i in enumerate(idx):
            events[f"stim_{kind}"].append((stim[i], fix[i]))
            events["fixation"].append((fix[i], stim[idx[j + 1]] if j + 1 < len(idx) else block_end))
    responses = {"response_1": impulses(stim + rt / 1000.0, n)}
    return events, responses


def gambling(evs_dir, tab, n, run_end):
    outcomes = {"win_event.txt": "reward_feedback", "loss_event.txt": "loss_feedback", "neut_event.txt": "neutral_feedback"}
    events = {"question_mark": [], **{name: [] for name in outcomes.values()}, "fixation": []}
    for ev, name in outcomes.items():
        for t in ev_onsets(evs_dir, ev):
            events["question_mark"].append((t, t + 1.5))
            events[name].append((t + 1.5, t + 2.5))
            events["fixation"].append((t + 2.5, t + 3.5))

    sync = tab["SyncSlide.OnsetTime"].dropna().iloc[0]
    answered = tab[tab["QuestionMark.RT"].notna() & tab["QuestionMark.RESP"].notna()]
    press = seconds(answered, "QuestionMark.OnsetTime", sync) + answered["QuestionMark.RT"] / 1000.0
    return events, {"response_1": impulses(press, n)}


def language(evs_dir, tab, n, run_end):
    sync_ms = float(np.loadtxt(os.path.join(evs_dir, "Sync.txt"))) * 1000.0
    cues = tab[tab["PresentBlockChange.OnsetTime"].notna()]
    story = tab[tab["StoryFile"].notna()]
    math = tab[tab["MathFile"].notna()]
    trials = tab[tab["StoryFile"].notna() | tab["MathFile"].notna()]
    pressed = trials[trials["FilteredTrialStats.RTTIME"] > 0]

    events = {
        "block_cue": intervals(cues, "PresentBlockChange.OnsetTime", "Wait500.OnsetTime[Block]", sync_ms),
        "story_listen": intervals(story, "PresentStoryFile.OnsetTime", "ResponsePeriod.OffsetTime", sync_ms),
        "math_listen": intervals(math, "PresentMathFile.OnsetTime", "ResponsePeriod.OffsetTime", sync_ms),
        "fixation": [],
    }
    return events, {"response_1": impulses(seconds(pressed, "FilteredTrialStats.RTTIME", sync_ms), n)}


MOTOR_CUES = {
    "LeftHandCueProcedure": ("lh", "LeftHandCue.OnsetTime"),
    "LeftFootCuePROC": ("lf", "LeftFootCue.OnsetTime"),
    "RightHandCuePROC": ("rh", "RightHandCue.OnsetTime"),
    "RightFoottCuePROC": ("rf", "RightFootCue.OnsetTime"),
    "TongueCuePROC": ("t", "TongueCue.OnsetTime"),
}
MOTOR_CROSSES = {
    "CrossLeftPROC": "CrossLeft.OnsetTime",
    "CrossRightPROC": "CrossRight.OnsetTime",
    "CrossCenterPROC": "CrossCenter.OnsetTime",
}


def motor(evs_dir, tab, n, run_end):
    sync = tab["CountDownSlide.OnsetTime"].dropna().iloc[0]
    blocks, fixations = [], []
    for _, row in tab[tab["Procedure[Trial]"].notna()].iterrows():
        proc = row["Procedure[Trial]"]
        if proc in MOTOR_CUES:
            name, column = MOTOR_CUES[proc]
            blocks.append({"name": name, "cue": (row[column] - sync) / 1000.0, "cross": [], "blank": []})
        elif proc == "FixPROC":
            fixations.append((row["Fixdot.OnsetTime"] - sync) / 1000.0)
        elif proc in MOTOR_CROSSES:
            blocks[-1]["cross"].append((row[MOTOR_CROSSES[proc]] - sync) / 1000.0)
            blocks[-1]["blank"].append((row["BLANK.OnsetTime"] - sync) / 1000.0)

    cue_starts = [b["cue"] for b in blocks]
    boundaries = cue_starts + fixations
    events = {f"{kind}_{m}": [] for kind in ("cue", "move") for m in ("lh", "lf", "rh", "rf", "t")}
    events["fixation"] = [(t, next_after(t, cue_starts, run_end)) for t in fixations]
    for b in blocks:
        events[f"cue_{b['name']}"].append((b["cue"], b["cross"][0]))
        events[f"move_{b['name']}"].append((b["cross"][0], next_after(b["blank"][-1], boundaries, run_end)))

    movements = ("lh", "lf", "rh", "rf", "t")
    responses = {f"response_{i}": occupancy(events[f"move_{m}"], n) for i, m in enumerate(movements, 1)}
    return events, responses


def relational(evs_dir, tab, n, run_end):
    sync = tab["SyncSlide.OnsetTime"].dropna().iloc[0]
    blocks, fixations, presses = [], [], []
    for _, row in tab[tab["Procedure"].notna()].iterrows():
        sec = lambda column: (row[column] - sync) / 1000.0
        proc = row["Procedure"]
        if proc == "RelationalPromptPROC":
            blocks.append({"name": "relation", "prompt": sec("RelationalPrompt.OnsetTime"), "trials": []})
        elif proc == "ControlPromptPROC":
            blocks.append({"name": "match", "prompt": sec("ControlPrompt.OnsetTime"), "trials": []})
        elif proc == "FixationPROC":
            fixations.append(sec("FixationBlock.OnsetTime"))
        elif proc in ("RelationalPROC", "ControlPROC"):
            slide, blank = ("RelationalSlide", "RelationalBlank") if proc == "RelationalPROC" else ("ControlSlide", "ControlBlank")
            blocks[-1]["trials"].append((sec(f"{slide}.OnsetTime"), sec(f"{blank}.OnsetTime")))
            if row[f"{slide}.RTTime"] > 0:
                presses.append(sec(f"{slide}.RTTime"))

    prompts = [b["prompt"] for b in blocks]
    events = {"prompt_relation": [], "prompt_match": [], "trial_relation": [], "trial_match": []}
    events["fixation"] = [(t, next_after(t, prompts, run_end)) for t in fixations]
    for b in blocks:
        events[f"prompt_{b['name']}"].append((b["prompt"], b["trials"][0][0]))
        events[f"trial_{b['name']}"].extend(b["trials"])
    return events, {"response_1": impulses(presses, n)}


def social(evs_dir, tab, n, run_end):
    sync = tab["CountDownSlide.OnsetTime"].dropna().iloc[0]
    blocks, presses = [], []
    for _, row in tab.iterrows():
        sec = lambda column: (row[column] - sync) / 1000.0
        if pd.notna(row["MovieSlide.OnsetTime"]):
            name = "movie_mental" if str(row["Type"]).strip() == "Mental" else "movie_random"
            blocks.append({"name": name, "movie": sec("MovieSlide.OnsetTime"), "response": sec("ResponseSlide.OnsetTime")})
            if row["ResponseSlide.RTTime"] > 0:
                presses.append(sec("ResponseSlide.RTTime"))
        if pd.notna(row["FixationBlock.OnsetTime"]):
            blocks[-1]["fixation"] = sec("FixationBlock.OnsetTime")

    movies = [b["movie"] for b in blocks]
    events = {"movie_mental": [], "movie_random": [], "response_prompt": [], "fixation": []}
    for b in blocks:
        events[b["name"]].append((b["movie"], b["fixation"]))
        events["response_prompt"].append((b["response"], b["fixation"]))
        events["fixation"].append((b["fixation"], next_after(b["fixation"], movies, run_end)))
    return events, {"response_1": impulses(presses, n)}


WM_CATEGORIES = {"Body": "stim_body", "Face": "stim_faces", "Place": "stim_places", "Tools": "stim_tools"}


def wm(evs_dir, tab, n, run_end):
    sync = tab["SyncSlide.OnsetTime"].dropna().iloc[0]
    cue0 = tab[tab["Procedure[Block]"] == "Cue0BackPROC"]
    cue2 = tab[tab["Procedure[Block]"] == "Cue2BackPROC"]
    cues = sorted(
        [(t, "cue_0bk") for t in seconds(cue0, "CueTarget.OnsetTime", sync)]
        + [(t, "cue_2bk") for t in seconds(cue2, "Cue2Back.OnsetTime", sync)]
    )
    fixations = seconds(tab[tab["Procedure[Block]"] == "Fix15secPROC"], "Fix15sec.OnsetTime", sync)
    trials = tab[tab["Procedure[Block]"] == "TrialsPROC"]
    stim = seconds(trials, "Stim.OnsetTime", sync)
    fix = seconds(trials, "Fix.OnsetTime", sync)

    events = {"cue_0bk": [], "cue_2bk": [], **{name: [] for name in WM_CATEGORIES.values()}}
    for b, (cue_start, name) in enumerate(cues):
        events[name].append((cue_start, stim[10 * b]))
    for category, start, end in zip(trials["StimType"], stim, fix):
        events[WM_CATEGORIES[category]].append((start, end))
    cue_starts = [t for t, _ in cues]
    events["fixation"] = [(t, next_after(t, cue_starts, run_end)) for t in fixations]

    answered = trials["Stim.RT"].notna() & trials["Stim.RESP"].notna()
    press = stim[answered.to_numpy()] + trials.loc[answered, "Stim.RT"].to_numpy() / 1000.0
    return events, {"response_1": impulses(press, n)}


BUILDERS = {
    "EMOTION": emotion,
    "GAMBLING": gambling,
    "LANGUAGE": language,
    "MOTOR": motor,
    "RELATIONAL": relational,
    "SOCIAL": social,
    "WM": wm,
}


def build_run(hcp_root, task, run_id, n):
    """Events and responses for one run ("100206_LR") over n seconds."""
    subject, direction = run_id.split("_")
    run = f"tfMRI_{task}_{direction}"
    results_dir = os.path.join(hcp_root, subject, "MNINonLinear/Results", run)
    evs_dir = os.path.join(results_dir, "EVs")
    run_end = len(np.loadtxt(os.path.join(results_dir, "Movement_Regressors.txt"))) * HCP_TR
    tab = read_tab(hcp_root, subject, run, "*WM*TAB.txt" if task == "WM" else "*TAB.txt")

    events, responses = BUILDERS[task](evs_dir, tab, n, run_end)
    events = {name: occupancy(intervals, n) for name, intervals in events.items()}
    any_response = np.clip(sum(responses.values()), 0.0, 1.0)
    responses["no_response"] = 1.0 - any_response
    responses["response_unknown"] = np.zeros(n)
    return events, responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hcp-root", required=True)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--h5", required=True)
    args = parser.parse_args()

    with h5py.File(args.h5, "r+") as f:
        run_ids = [r.decode() if isinstance(r, bytes) else str(r) for r in f["long_subject_id"][:]]
        lengths = f["valid_timepoints"][:]
        max_len = int(lengths.max())

        groups = {"events": {}, "responses": {}}
        for row, (run_id, n) in enumerate(zip(run_ids, lengths)):
            for group, values in zip(groups, build_run(args.hcp_root, args.task, run_id, int(n))):
                for name, x in values.items():
                    if name not in groups[group]:
                        groups[group][name] = np.full((len(run_ids), max_len), np.nan, dtype=np.float32)
                    groups[group][name][row, : len(x)] = x

        for group, arrays in groups.items():
            if group in f:
                del f[group]
            for name, data in arrays.items():
                f.create_dataset(f"{group}/{name}", data=data)
    print(f"Wrote events/ and responses/ for {len(run_ids)} {args.task} runs to {args.h5}")


if __name__ == "__main__":
    main()
