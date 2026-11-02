# [NeurIPS 2026] Flow Matching with In-Context Priors for Out-of-Distribution Brain Dynamics

https://github.com/user-attachments/assets/81f7e298-7dee-45e2-817a-4168bf3ad13b

Paper: https://arxiv.org/abs/2606.11833

[I'll be adding a small, clearer tutorial soon!]

The main workflow is:

1. Create the environment below.
2. Prepare H5 files with the schema below. For HCP, build the event and
   response timecourses with `util/hcp_events.py` (see [Events](#events)), then
   add the caption embeddings with `util/add_captions_to_h5.py` (see [Captions](#captions)).
3. Copy a dynamics config and set local data/output paths.
4. Train with `train_dynamics.py`, or evaluate a pretrained checkpoint with
   `eval.hcp.run_hcp_eval`.

## Environment


Python 3.12 / PyTorch 2.6.0+cu124, CUDA 12.4, cuDNN 9.1
Recommended install:

```bash
conda env create -f environment.yml
conda activate pinc-flows
```

Or inside an existing Python 3.12 environment:

```bash
python -m pip install -r requirements-cu124.txt
```

## Configs

Reference pretrained config:

```bash
config/pretrained/pinc-flows_fold0_epoch0999.yaml
```

Pretrained fold configs live under:

```bash
config/pretrained/
```

Use `config/train_template.yaml` for from-scratch training.


## Pretrained Models

Pretrained fold-specific checkpoints and configs are available
on HuggingFace (https://huggingface.co/SamGijsen/pinc-flows/). Each fold holds out one HCP task plus a corresponding set of
IBC tasks; use the `fold_to_holdout_mapping.yaml` included with the checkpoints
to choose the fold. Update placeholder paths such as `/TODO_SET_PATH/...` in
the released configs.

Expected release files:

```text
pinc-flows_fold0_epoch0999.pt
pinc-flows_fold0_epoch0999.yaml
...
pinc-flows_fold6_epoch0999.pt
pinc-flows_fold6_epoch0999.yaml
fold_to_holdout_mapping.yaml
```

## H5 Dataset Schema

For the dynamics model train/eval path, each H5 file must contain:

```text
long_subject_id                         [N]
timeseries/<atlas>                      [N, R, T]
valid_timepoints                        [N]
events/<event_name>                     [N, T]
responses/<response_name>               [N, T]
embeddings/events/instruction/<event>   [V, 1024]
embeddings/events/sensory/<event>       [V, 1024]
embeddings/responses/<response>         [V, 1024]
relevance_v2/<atlas>/events/<event>     [N, R]
relevance_v2/<atlas>/responses/<resp>   [N, R]
```

Our configs use `atlas=schaefer400`, `R=400`, and language-event
conditioning with `condition_cont_dim=3074`: instruction 1024, sensory 1024,
response 1024, plus `responses/no_response` and `responses/response_unknown`.
Those two response timecourses are required but do not need embedding rows.

## Events

`events/<event_name>` and `responses/<response_name>` are timecourses on a 1 s
grid: each value is the fraction of that second occupied by the event (0–1),
padded with NaN after `valid_timepoints`. Button presses are split across the
two seconds around the press time. `responses/no_response` is 1 minus the sum
of the response channels, and `responses/response_unknown` is all zeros for HCP.
Event and response names must match the captions (e.g. `question_mark`,
`loss_feedback`, `response_1`).

For HCP, `util/hcp_events.py` builds them from the EV files, movement
regressors and E-Prime TAB files (from the unprocessed task packages) and
writes them into an H5 that already has `long_subject_id` (e.g. `100206_LR`)
and `valid_timepoints`:

```bash
python util/hcp_events.py --hcp-root /path/to/HCP_1200 --task GAMBLING --h5 /path/to/gambling.h5
```

## Captions

The text captions and their embeddings can be found (together with the checkpoints) on HuggingFace, in `captions/`:

- `captions.h5`: one group per task, named `<dataset>/<task>`: HCP and IBC
  tasks as in `fold_to_holdout_mapping.yaml` (e.g. `hcp/EMOTION`,
  `ibc/Discount`); all Nakai files use `nakai/all`. Each group holds the
  `embeddings/...` datasets required by the schema above, plus the caption
  text (`events_desc/<event>/{sensory,instruction}`,
  `events_desc_long/<event>`, `response_desc/<response>`).
- `captions.csv`: the same captions as a flat table
  (`dataset, task, kind, name, variant_idx, text`).

Each event and response has several phrasings (`V` rows). During training, we sample a phrase at random, while evaluation by default uses `variant_idx` 0. The `BAAI/bge-large-en-v1.5` model was used for embedding.

To add them to your H5 files:

```bash
hf download SamGijsen/pinc-flows captions/captions.h5 --local-dir .
python util/add_captions_to_h5.py --captions captions/captions.h5 --task ibc/Discount --h5 /path/to/Discount.h5
# all IBC files at once (file names equal task names):
for f in /path/to/ibc/*.h5; do python util/add_captions_to_h5.py --captions captions/captions.h5 --task ibc/$(basename $f .h5) --h5 $f; done
```

The script fails if any `events/<name>` or `responses/<name>` in your file
has no caption. For the resting-state (UKBB) file, use `util/add_h5_fields_to_rest.py` instead (`rest` event
has an all-zero embedding). For other datasets, write captions in the same
format and embed them with the same model.

## Train

Copy a config, edit paths locally, then run:

```bash
conda run -n braincontrol python train_dynamics.py --config config/train_template.yaml
```

For training from scratch, leave `resume_checkpoint` and
`weights_only_checkpoint` unset. To initialize from released weights without
resuming optimizer state, set `weights_only_checkpoint` to the downloaded
checkpoint. The config writes checkpoints under `output_dir/run_name`.

## Inference And Evaluation

The reference config has `evaluation.hcp.enabled: true`, so `train_dynamics.py`
runs HCP/IBC evaluation at the final checkpoint.

Manual HCP/IBC evaluation is exposed as:

```bash
python -c "from eval.hcp import run_hcp_eval; run_hcp_eval('path/to/config.yaml', 'path/to/checkpoint.pt', 'path/to/eval_out')"
```

Minimal pretrained evaluation:

```bash
conda activate braincontrol
python -c "from eval.hcp import run_hcp_eval; run_hcp_eval('pinc-flows_fold0_epoch0999.yaml', 'pinc-flows_fold0_epoch0999.pt', 'eval_fold0')"
```

The eval config controls tasks, guidance scales, phrase variants, relevance
input modes, rollout counts, and whether IBC GLM is run under `evaluation.hcp`.

At batch size 256, inference/eval uses <10 GB GPU memory. The paper
training runs used one 32 GB GPU; half the batch size if you're using a ~20GB
GPU..

## Data Availability

This repository does not redistribute HCP, IBC, or derived H5 data files.
Users must obtain the source datasets according to their licenses and construct
H5 files matching the schema above.

Captions and caption embeddings for the datasets we trained on (HCP, IBC, Nakai) are released with the checkpoints (see [Captions](#captions)).
