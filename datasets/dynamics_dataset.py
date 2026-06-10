import math
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from util.tokenizer_language import mean_center_l2_normalize_rows_np


class SequenceDataset(Dataset):
    """
    Dataset that yields sequences of crops for dynamics training.

    Each sample is a sequence of T consecutive crops from the same subject/run,
    along with categorical condition indices for each timestep.

    Loads directly from HDF5 files.
    """

    def __init__(
        self,
        data_path,
        sequence_length,
        crop_length=3,
        condition_label_name=None,
        num_conditions=3,
        atlas_names=None,
        subject_ids_path=None,
        input_stride=1,
        unlabeled_condition=0,
        condition_mode='discrete',
        condition_cont_label_name=None,
        condition_cont_idx_label_name=None,
        condition_cont_scale_label_name=None,
        condition_cont_embeddings_name=None,
        condition_cont_language_events=None,
        pretrain_dynamics=None,
        unconditioned_pretraining=False,
        condition_relevance_path=None,
        condition_cont_dim=None,
        event_mapping=None,
        tr_seconds=None,
        sample_mode='random_safe',
        ood_tag_label_name=None,
        allowed_ood_tags=None,
        anchor_tag_label_name=None,
        anchor_tags=None,
        anchor_crop_index=None,
        subject_token_enabled=False,
        subject_context_length=None,
        subject_min_gap=None,
        age_label_name=None,
        sex_label_name=None,
        motion_label_name=None,
        field_strength_t=None,
        p_splice=0.0,
        splice_context_frames=None,
        splice_generation_frames=None,
        splice_tr_seconds=0.72,
        splice_recovery_gamma_shape=6.0,
        splice_onset_only=False,
        splice_onset_event_paths=None,
        splice_onset_threshold=1e-6,
        force_zero_relevance_scores=False,
    ):
        """
        Args:
            data_path: path to HDF5 data file
            sequence_length: number of crops per sequence (T)
            crop_length: timesteps per crop (matches tokenizer input_timesteps)
            condition_label_name: HDF5 path to categorical condition labels, e.g. 'labels/condition'
            num_conditions: number of condition categories (default 3: fixation/shape/face)
            atlas_names: list of atlas names to concatenate, e.g. ['schaefer400', 'tian3']
            subject_ids_path: optional path to .npy file with allowed subject IDs
            input_stride: temporal stride for volume selection (default 1 = consecutive volumes,
                          2 = every other volume, etc.). Effective TR = original_TR * input_stride
            condition_mode: 'discrete' or 'cont'
            condition_cont_label_name: HDF5 path to dense continuous condition vectors [N, T, D]
            condition_cont_idx_label_name: HDF5 path to per-volume embedding indices [N, T]
            condition_cont_scale_label_name: HDF5 path to per-volume embedding scales [N, T]
            condition_cont_embeddings_name: HDF5 path to embedding table [N_embed, D]
            condition_cont_language_events: config for H5 event-weighted language embedding pools
            pretrain_dynamics: config for dataset-derived language conditioning from ROI activity
            condition_relevance_path: optional .npy path with precomputed ROI relevance pools
            condition_cont_dim: optional expected dim for continuous condition vectors
            event_mapping: optional list of HDF5 event-to-dimension mapping specs for continuous conditions
            tr_seconds: base TR in seconds for pretrain_dynamics temporal filtering
            sample_mode: 'random_safe' or 'anchored_eval'
            ood_tag_label_name: HDF5 path to OOD tag labels [N, T] used for safe filtering
            allowed_ood_tags: optional allowed OOD tags for all sampled TRs (e.g. [0])
            anchor_tag_label_name: HDF5 path for anchor tags (default fallback: ood_tag_label_name)
            anchor_tags: tags used as eval anchors in anchored_eval mode (onset detected on this tag set)
            anchor_crop_index: crop index where anchor onset must occur
            subject_token_enabled: whether to emit distant subject-context crops
            subject_context_length: number of crops in distant context
            subject_min_gap: minimum raw-volume gap from main sequence window
            age_label_name: optional HDF5 path to per-run age labels [N]
            sex_label_name: optional HDF5 path to per-run sex labels [N]
            motion_label_name: optional HDF5 path to per-run motion labels [N]
            field_strength_t: optional dataset-level field strength (3 or 7 Tesla)
            p_splice: probability of replacing late generation frames with a second sampled sequence
            splice_context_frames: number of context frames (prefix from first sequence)
            splice_generation_frames: number of generation frames (suffix eligible for splice)
            splice_tr_seconds: base TR (seconds) used to build HRF-informed recovery weights
            splice_recovery_gamma_shape: rising gamma shape used for splice recovery weighting
            splice_onset_only: require splice donors to place an event onset at the splice boundary
            splice_onset_event_paths: HDF5 event arrays used to detect donor onsets, e.g. ['events/lh']
            splice_onset_threshold: onset threshold; prev<=thr and curr>thr counts as a raw onset
        """
        self.sequence_length = sequence_length
        self.crop_length = crop_length
        self.condition_label_name = condition_label_name
        self.num_conditions = num_conditions
        self.input_stride = input_stride
        self.unlabeled_condition = unlabeled_condition
        self.condition_mode = condition_mode
        self.condition_cont_label_name = condition_cont_label_name
        self.condition_cont_idx_label_name = condition_cont_idx_label_name
        self.condition_cont_scale_label_name = condition_cont_scale_label_name
        self.condition_cont_embeddings_name = condition_cont_embeddings_name
        self.condition_cont_language_events = condition_cont_language_events
        self.pretrain_dynamics = pretrain_dynamics
        self.unconditioned_pretraining = bool(unconditioned_pretraining)
        self.condition_relevance_path = condition_relevance_path
        self.event_mapping = event_mapping
        self.tr_seconds = None if tr_seconds is None else float(tr_seconds)
        self.sample_mode = sample_mode
        if self.sample_mode not in ('random_safe', 'anchored_eval'):
            raise ValueError(f"Unknown sample_mode: {self.sample_mode}")
        self.ood_tag_label_name = ood_tag_label_name
        self.anchor_tag_label_name = anchor_tag_label_name
        self.allowed_ood_tags = self._normalize_tag_values(allowed_ood_tags)
        self.anchor_tags = self._normalize_tag_values(anchor_tags)
        self.subject_token_enabled = bool(subject_token_enabled)
        self.subject_context_length = None if subject_context_length is None else int(subject_context_length)
        self.subject_min_gap = None if subject_min_gap is None else int(subject_min_gap)
        self.age_label_name = age_label_name
        self.sex_label_name = sex_label_name
        self.motion_label_name = motion_label_name
        self.field_strength_t = field_strength_t
        self.p_splice = float(p_splice)
        self.splice_context_frames = None if splice_context_frames is None else int(splice_context_frames)
        self.splice_generation_frames = None if splice_generation_frames is None else int(splice_generation_frames)
        self.splice_tr_seconds = float(splice_tr_seconds)
        self.splice_recovery_gamma_shape = float(splice_recovery_gamma_shape)
        self.splice_onset_only = bool(splice_onset_only)
        if splice_onset_event_paths is None:
            self.splice_onset_event_paths = None
        elif isinstance(splice_onset_event_paths, str):
            self.splice_onset_event_paths = [splice_onset_event_paths]
        else:
            self.splice_onset_event_paths = [str(path) for path in splice_onset_event_paths]
        self.splice_onset_threshold = float(splice_onset_threshold)
        self.force_zero_relevance_scores = bool(force_zero_relevance_scores)
        self.splice_recovery_weights = None
        self.splice_onset_arrays = None
        self.splice_onset_donor_pools = None
        if not (0.0 <= self.p_splice <= 1.0):
            raise ValueError(f"p_splice must be in [0, 1], got {self.p_splice}")
        if self.p_splice > 0.0:
            if self.sample_mode != 'random_safe':
                raise ValueError("p_splice > 0 requires sample_mode='random_safe'")
            if self.splice_context_frames is None or self.splice_generation_frames is None:
                raise ValueError(
                    "p_splice > 0 requires splice_context_frames and splice_generation_frames"
                )
            if self.splice_generation_frames < 2:
                raise ValueError("p_splice > 0 requires splice_generation_frames >= 2")
            if self.splice_context_frames + self.splice_generation_frames != self.sequence_length:
                raise ValueError(
                    f"splice frame split must match sequence_length={self.sequence_length}, got "
                    f"context={self.splice_context_frames}, generation={self.splice_generation_frames}"
                )
            if self.splice_tr_seconds <= 0.0:
                raise ValueError(f"splice_tr_seconds must be > 0, got {self.splice_tr_seconds}")
            if self.splice_recovery_gamma_shape <= 0.0:
                raise ValueError(
                    f"splice_recovery_gamma_shape must be > 0, got {self.splice_recovery_gamma_shape}"
                )
            if self.splice_onset_only and not self.splice_onset_event_paths:
                raise ValueError(
                    "splice_onset_only=True requires splice_onset_event_paths to be configured"
                )
            self.splice_recovery_weights = self._build_splice_recovery_weights(self.splice_generation_frames)
        if self.subject_token_enabled:
            assert self.subject_context_length is not None and self.subject_context_length > 0
            assert self.subject_min_gap is not None and self.subject_min_gap >= 0

        # Total raw volumes needed per sequence (accounting for stride)
        # With stride, we need: (crop_length * sequence_length - 1) * stride + 1 volumes
        self.total_seq_length = (crop_length * sequence_length - 1) * input_stride + 1
        self.n_values = self.crop_length * self.sequence_length
        self.sample_offsets = np.arange(self.n_values, dtype=np.int64) * self.input_stride
        if self.subject_token_enabled:
            self.subject_total_seq_length = (
                (crop_length * self.subject_context_length - 1) * input_stride + 1
            )
            self.subject_n_values = self.crop_length * self.subject_context_length
            self.subject_sample_offsets = np.arange(self.subject_n_values, dtype=np.int64) * self.input_stride
        if anchor_crop_index is None:
            self.anchor_crop_index = 0
        else:
            self.anchor_crop_index = int(anchor_crop_index)

        # Load data
        self.file = h5py.File(data_path, 'r')

        # Get atlas names from file if not specified
        if atlas_names is None:
            atlas_names = list(self.file['timeseries'].keys())
        self.atlas_names = atlas_names
        self.num_rois = sum(int(self.file[f'timeseries/{atlas}'].shape[1]) for atlas in self.atlas_names)

        # Load subject IDs
        def _normalize_subject_id(x):
            if isinstance(x, (bytes, bytearray, np.bytes_)):
                return x.decode('utf-8')
            return str(x)

        all_subject_ids = [_normalize_subject_id(s) for s in self.file['long_subject_id'][:]]

        # Filter to target subjects if specified
        if subject_ids_path and os.path.exists(subject_ids_path):
            loaded_subjects = np.asarray(np.load(subject_ids_path, allow_pickle=True)).reshape(-1)
            target_subjects = {_normalize_subject_id(s) for s in loaded_subjects}
            self.subject_indices = [i for i, sid in enumerate(all_subject_ids)
                                    if sid in target_subjects]
        else:
            self.subject_indices = list(range(len(all_subject_ids)))

        self.subject_ids = [all_subject_ids[i] for i in self.subject_indices]
        self.subject_id_by_index = {i: all_subject_ids[i] for i in self.subject_indices}

        # Load condition labels if specified
        self.condition_labels = None
        if self.condition_label_name and self.condition_label_name in self.file:
            self.condition_labels = self.file[self.condition_label_name][:]
        self.condition_cont_labels = None
        self.condition_cont_idx_labels = None
        self.condition_cont_scale_labels = None
        self.condition_cont_embeddings = None
        self.event_arrays = {}
        self.event_mappings = []
        self.language_event_arrays = {}
        self.language_event_specs = []
        self.language_event_pools = {}
        self.language_event_pool_means = {}
        self.language_event_relevance_pools = {}
        self.language_event_relevance_pool_means = {}
        self.language_event_subject_relevance_pools = {}
        self.language_event_subject_relevance_pool_means = {}
        self.language_event_h5_relevance_events = {}
        self.language_event_h5_relevance_responses = {}
        self.language_event_h5_relevance_ood_events = {}
        self.language_event_h5_relevance_ood_responses = {}
        self.condition_relevance_source = self.condition_relevance_path
        self.language_event_variant_sampling = None
        self.language_event_fixed_variant_idx = None
        self.language_event_dim = None
        self.language_event_piece_dim = None
        self.language_event_no_response_key = None
        self.language_event_unknown_response_key = None
        self.pretrain_roi_embeddings = None
        self.pretrain_condition_dim = None
        self.pretrain_membership_center = None
        self.pretrain_membership_sharpness = None
        self.pretrain_mass_threshold = None
        self.pretrain_mass_min = None
        self.pretrain_hrf_conditioning_offset = None
        self.pretrain_lowpass_hz = None
        self.condition_cont_source = 'none'

        if self.unconditioned_pretraining:
            if self.condition_mode != 'cont':
                raise ValueError("unconditioned_pretraining requires condition_mode='cont'")
            if self.pretrain_dynamics is not None:
                raise ValueError("unconditioned_pretraining does not support pretrain_dynamics")
            if self.event_mapping is not None:
                raise ValueError("unconditioned_pretraining does not support event_mapping")
            if condition_cont_dim is None:
                raise ValueError("unconditioned_pretraining requires condition_cont_dim")
        else:
            # Dense per-volume vectors (legacy/current default path)
            if self.condition_cont_label_name and self.condition_cont_label_name in self.file:
                self.condition_cont_labels = self.file[self.condition_cont_label_name][:]
                self.condition_cont_source = 'dense'

            # Symbolic index+scale+table representation (preferred for large repeated events)
            has_idx = (
                self.condition_cont_idx_label_name
                and self.condition_cont_idx_label_name in self.file
            )
            has_scale = (
                self.condition_cont_scale_label_name
                and self.condition_cont_scale_label_name in self.file
            )
            has_embed = (
                self.condition_cont_embeddings_name
                and self.condition_cont_embeddings_name in self.file
            )
            if self.condition_cont_labels is None and has_idx and has_scale and has_embed:
                self.condition_cont_idx_labels = self.file[self.condition_cont_idx_label_name][:]
                self.condition_cont_scale_labels = self.file[self.condition_cont_scale_label_name][:]
                self.condition_cont_embeddings = self.file[self.condition_cont_embeddings_name][:]
                self.condition_cont_source = 'indexed'

            if self.condition_cont_language_events is not None:
                if self.condition_mode != 'cont':
                    raise ValueError("condition_cont_language_events requires condition_mode='cont'")
                if self.condition_cont_source != 'none':
                    raise ValueError(
                        "Multiple continuous condition sources configured. "
                        "Use only one of dense/indexed/condition_cont_language_events/pretrain_dynamics/event_mapping."
                    )
                self._init_condition_cont_language_events(self.condition_cont_language_events)
                self.condition_cont_source = 'language_event_pool'

            if self.pretrain_dynamics is not None:
                if self.condition_mode != 'cont':
                    raise ValueError("pretrain_dynamics requires condition_mode='cont'")
                if self.crop_length != 1:
                    raise ValueError(
                        f"pretrain_dynamics requires crop_length=1, got {self.crop_length}"
                    )
                if self.condition_cont_source != 'none':
                    raise ValueError(
                        "Multiple continuous condition sources configured. "
                        "Use only one of dense/indexed/condition_cont_language_events/pretrain_dynamics/event_mapping."
                    )
                self._init_pretrain_dynamics(self.pretrain_dynamics, condition_cont_dim=condition_cont_dim)
                self.condition_cont_source = 'pretrain_dynamics'

            if self.event_mapping is not None:
                if self.condition_cont_source != 'none':
                    raise ValueError(
                        "Multiple continuous condition sources configured. "
                        "Use only one of dense/indexed/condition_cont_language_events/pretrain_dynamics/event_mapping."
                    )
                if not isinstance(self.event_mapping, (list, tuple)) or len(self.event_mapping) == 0:
                    raise ValueError("event_mapping must be a non-empty list when provided")
                for mapping in self.event_mapping:
                    if not isinstance(mapping, dict):
                        raise TypeError("event_mapping entries must be dicts")
                    h5_key = mapping.get('h5_key')
                    if h5_key is None:
                        raise ValueError("event_mapping entry missing required key 'h5_key'")
                    if h5_key not in self.file:
                        raise KeyError(f"Event mapping key not found in H5: {h5_key}")
                    if h5_key not in self.event_arrays:
                        self.event_arrays[h5_key] = self.file[h5_key][:]
                    transform = mapping.get('transform', 'identity')
                    if transform == 'identity':
                        dim = mapping.get('dim')
                        if dim is None:
                            raise ValueError("event_mapping identity transform requires 'dim'")
                        self.event_mappings.append({
                            'h5_key': h5_key,
                            'transform': 'identity',
                            'dim': int(dim),
                        })
                    elif transform == 'sign_split':
                        dims = mapping.get('dims')
                        if not isinstance(dims, (list, tuple)) or len(dims) != 2:
                            raise ValueError("event_mapping sign_split transform requires 'dims' with length 2")
                        self.event_mappings.append({
                            'h5_key': h5_key,
                            'transform': 'sign_split',
                            'dims': (int(dims[0]), int(dims[1])),
                        })
                    elif transform == 'onehot':
                        dims = mapping.get('dims')
                        if not isinstance(dims, (list, tuple)) or len(dims) == 0:
                            raise ValueError("event_mapping onehot transform requires non-empty 'dims'")
                        self.event_mappings.append({
                            'h5_key': h5_key,
                            'transform': 'onehot',
                            'dims': [int(d) for d in dims],
                        })
                    else:
                        raise ValueError(f"Unknown event_mapping transform: {transform}")
                self.condition_cont_source = 'event_mapping'

        if (
            self.condition_relevance_path is not None
            and self.condition_mode == 'cont'
            and self.condition_cont_source != 'language_event_pool'
        ):
            raise ValueError(
                "condition_relevance_path requires condition_cont_language_events when condition_mode='cont'"
            )

        if self.condition_cont_source == 'language_event_pool':
            if self.language_event_dim is None:
                raise ValueError("language_event_pool source did not set language_event_dim")
            if condition_cont_dim is not None and int(condition_cont_dim) != int(self.language_event_dim):
                raise ValueError(
                    f"condition_cont_dim={condition_cont_dim} does not match language event embedding dim "
                    f"{self.language_event_dim}"
                )
            self.condition_cont_dim = int(self.language_event_dim)
        elif self.condition_cont_source == 'pretrain_dynamics':
            if self.pretrain_condition_dim is None:
                raise ValueError("pretrain_dynamics source did not set pretrain_condition_dim")
            if condition_cont_dim is not None and int(condition_cont_dim) != int(self.pretrain_condition_dim):
                raise ValueError(
                    f"condition_cont_dim={condition_cont_dim} does not match pretrain_dynamics dim "
                    f"{self.pretrain_condition_dim}"
                )
            self.condition_cont_dim = int(self.pretrain_condition_dim)
        elif self.condition_cont_source == 'event_mapping':
            if condition_cont_dim is None:
                raise ValueError("condition_cont_dim must be set when using event_mapping")
            self.condition_cont_dim = int(condition_cont_dim)
        elif self.condition_cont_labels is not None:
            self.condition_cont_dim = int(self.condition_cont_labels.shape[-1])
        elif self.condition_cont_embeddings is not None:
            self.condition_cont_dim = int(self.condition_cont_embeddings.shape[-1])
        else:
            self.condition_cont_dim = int(condition_cont_dim) if condition_cont_dim is not None else 1

        self.ood_tags = None
        if self.ood_tag_label_name and self.ood_tag_label_name in self.file:
            self.ood_tags = self.file[self.ood_tag_label_name][:]

        self.anchor_tags_arr = None
        anchor_name = self.anchor_tag_label_name or self.ood_tag_label_name
        if anchor_name and anchor_name in self.file:
            self.anchor_tags_arr = self.file[anchor_name][:]

        n_rows = len(all_subject_ids)
        if self.splice_onset_event_paths is not None:
            self.splice_onset_arrays = self._load_splice_onset_arrays(n_rows)
        self.age_values = None
        if self.age_label_name is not None:
            if self.age_label_name not in self.file:
                raise KeyError(f"Age label path not found in H5: {self.age_label_name}")
            age_values = np.asarray(self.file[self.age_label_name][:], dtype=np.float32).reshape(-1)
            if age_values.shape[0] != n_rows:
                raise ValueError(
                    f"Age label length {age_values.shape[0]} does not match N_runs={n_rows}"
                )
            self.age_values = age_values

        self.sex_values = None
        if self.sex_label_name is not None:
            if self.sex_label_name not in self.file:
                raise KeyError(f"Sex label path not found in H5: {self.sex_label_name}")
            sex_values = np.asarray(self.file[self.sex_label_name][:], dtype=np.int64).reshape(-1)
            if sex_values.shape[0] != n_rows:
                raise ValueError(
                    f"Sex label length {sex_values.shape[0]} does not match N_runs={n_rows}"
                )
            self.sex_values = sex_values

        self.motion_values = None
        if self.motion_label_name is not None:
            if self.motion_label_name not in self.file:
                raise KeyError(f"Motion label path not found in H5: {self.motion_label_name}")
            motion_values = np.asarray(self.file[self.motion_label_name][:], dtype=np.float32).reshape(-1)
            if motion_values.shape[0] != n_rows:
                raise ValueError(
                    f"Motion label length {motion_values.shape[0]} does not match N_runs={n_rows}"
                )
            self.motion_values = motion_values

        self.field_strength_idx = None
        if self.field_strength_t is not None:
            fs = float(self.field_strength_t)
            if fs not in (3.0, 7.0):
                raise ValueError(f"field_strength_t must be 3 or 7, got {self.field_strength_t}")
            self.field_strength_idx = 0 if fs == 3.0 else 1

        self.subject_valid_volume_masks = self._compute_subject_valid_volume_masks()

        # Build sampling pools
        if self.splice_onset_only:
            self.splice_onset_donor_pools = self._compute_subject_splice_onset_pools()
        self.subject_start_pools = self._compute_subject_start_pools()
        if self.subject_token_enabled:
            self.subject_distant_start_pools = self._compute_subject_distant_start_pools()
        self.anchored_samples = self._compute_anchored_samples()

        if self.sample_mode == 'anchored_eval':
            print(f"SequenceDataset: {len(self.anchored_samples)} anchored eval samples")
        else:
            print(f"SequenceDataset: {len(self.subject_start_pools)} subjects, 1 random sequence per subject per epoch")
        print(f"  input_stride={input_stride}, {self.total_seq_length} raw volumes -> {sequence_length} crops of {crop_length}")
        print(f"  condition_mode={self.condition_mode}")
        print(f"  sample_mode={self.sample_mode}")
        if self.condition_labels is not None:
            print(f"  Condition labels loaded from '{self.condition_label_name}' with {self.num_conditions} categories")
        else:
            print(f"  No condition labels - using default condition {self.unlabeled_condition}")
        if self.condition_cont_source == 'dense':
            print(f"  Continuous condition labels loaded from '{self.condition_cont_label_name}' with dim={self.condition_cont_dim}")
        elif self.condition_cont_source == 'indexed':
            print(
                "  Continuous condition labels loaded from indexed representation: "
                f"idx='{self.condition_cont_idx_label_name}', "
                f"scale='{self.condition_cont_scale_label_name}', "
                f"embeddings='{self.condition_cont_embeddings_name}', dim={self.condition_cont_dim}"
            )
        elif self.condition_cont_source == 'event_mapping':
            print(f"  Continuous condition labels built from event_mapping with dim={self.condition_cont_dim}")
        elif self.condition_cont_source == 'pretrain_dynamics':
            print(
                "  Continuous condition labels built from pretrain_dynamics "
                f"with dim={self.condition_cont_dim} "
                f"(center={self.pretrain_membership_center}, sharpness={self.pretrain_membership_sharpness}, "
                f"mass_min={self.pretrain_mass_min}, mass_threshold={self.pretrain_mass_threshold}, "
                f"hrf_offset={self.pretrain_hrf_conditioning_offset}, lowpass_hz={self.pretrain_lowpass_hz}, "
                f"tr_seconds={self.tr_seconds})"
            )
        elif self.condition_cont_source == 'language_event_pool':
            print(
                "  Continuous condition labels built from language event pools "
                f"with dim={self.condition_cont_dim} "
                f"(events={len(self.language_event_specs)}, variant_sampling={self.language_event_variant_sampling})"
            )
        if self.condition_relevance_source is not None:
            print(f"  Relevance scores loaded from '{self.condition_relevance_source}'")
        if self.language_event_h5_relevance_ood_events:
            print(f"  OOD relevance loaded ({len(self.language_event_h5_relevance_ood_events)} events)")
        if self.ood_tags is not None:
            print(f"  OOD tags loaded from '{self.ood_tag_label_name}'")
        elif self.ood_tag_label_name:
            print(f"  OOD tag path '{self.ood_tag_label_name}' not found - sampling unfiltered")
        if self.allowed_ood_tags is not None:
            print(f"  allowed_ood_tags={self.allowed_ood_tags}")
        if self.sample_mode == 'anchored_eval':
            print(f"  anchor_tags={self.anchor_tags} anchor_crop_index={self.anchor_crop_index}")
        if self.subject_token_enabled:
            print(
                f"  subject_context_length={self.subject_context_length} "
                f"subject_min_gap={self.subject_min_gap}"
            )
        if self.age_values is not None:
            print(f"  age labels loaded from '{self.age_label_name}'")
        if self.sex_values is not None:
            print(f"  sex labels loaded from '{self.sex_label_name}'")
        if self.motion_values is not None:
            print(f"  motion labels loaded from '{self.motion_label_name}'")
        if self.field_strength_idx is not None:
            fs_text = 3 if self.field_strength_idx == 0 else 7
            print(f"  field_strength_t={fs_text}")
        if self.splice_context_frames is not None:
            print(
                "  splice: "
                f"p_splice={self.p_splice}, context_frames={self.splice_context_frames}, "
                f"generation_frames={self.splice_generation_frames}"
            )
            print(
                "  splice recovery: "
                f"tr_seconds={self.splice_tr_seconds}, gamma_shape={self.splice_recovery_gamma_shape}"
            )
            if self.splice_onset_only:
                donor_run_count = len(self.splice_onset_donor_pools)
                donor_window_count = sum(
                    starts.size
                    for per_run in self.splice_onset_donor_pools.values()
                    for starts in per_run.values()
                )
                print(
                    "  splice onset donors: "
                    f"paths={self.splice_onset_event_paths}, threshold={self.splice_onset_threshold}, "
                    f"runs={donor_run_count}, windows={donor_window_count}"
                )

    def _normalize_tag_values(self, values):
        def _parse_one(v):
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    return None
                if s.startswith('0b'):
                    return int(s, 2)
                # Support direct protocol bitstrings in config, e.g. "001", "010", "111".
                if all(ch in '01' for ch in s):
                    return int(s, 2)
                return int(s)
            return int(v)

        if values is None:
            return None
        if isinstance(values, str):
            items = [tok.strip() for tok in values.split(',') if tok.strip()]
            if not items:
                return None
            parsed = {_parse_one(tok) for tok in items}
            parsed.discard(None)
            return sorted(parsed) if parsed else None
        if isinstance(values, (list, tuple, set)):
            if len(values) == 0:
                return None
            parsed = {_parse_one(v) for v in values}
            parsed.discard(None)
            return sorted(parsed) if parsed else None
        parsed_one = _parse_one(values)
        return [parsed_one] if parsed_one is not None else None

    def has_direct_relevance_scores(self):
        return (
            bool(self.language_event_relevance_pools)
            or bool(self.language_event_h5_relevance_events)
            or bool(self.language_event_h5_relevance_responses)
            or self.condition_cont_source == 'pretrain_dynamics'
            or self.force_zero_relevance_scores
        )

    def install_language_event_relevance_pools(self, pools, source_name='in_memory', subject_pools=None):
        if self.condition_cont_source != 'language_event_pool':
            raise ValueError(
                "install_language_event_relevance_pools requires condition_cont_source='language_event_pool'"
            )

        used_pool_keys = sorted({
            spec['pool_key'] for spec in self.language_event_specs if spec['kind'] in ('pool', 'alias')
        })
        installed = {}
        for pool_key in used_pool_keys:
            pool = np.asarray(pools[pool_key], dtype=np.float32)
            if pool.ndim != 2:
                raise ValueError(
                    f"Expected relevance pool '{pool_key}' with shape [N, R], got {pool.shape}"
                )
            if pool.shape[1] != self.num_rois:
                raise ValueError(
                    f"Relevance pool '{pool_key}' width {pool.shape[1]} does not match num_rois={self.num_rois}"
                )
            installed[pool_key] = pool.copy()

        self.language_event_relevance_pools = installed
        self.language_event_relevance_pool_means = {
            pool_key: pool.mean(axis=0).astype(np.float32, copy=True)
            for pool_key, pool in installed.items()
        }
        self.language_event_h5_relevance_events = {}
        self.language_event_h5_relevance_responses = {}
        subject_installed = {}
        subject_installed_means = {}
        if subject_pools is not None:
            for subject_id, subject_pool_values in subject_pools.items():
                subject_pool_dict = {}
                subject_pool_means = {}
                for pool_key in used_pool_keys:
                    if pool_key not in subject_pool_values:
                        continue
                    pool = np.asarray(subject_pool_values[pool_key], dtype=np.float32)
                    if pool.ndim != 2:
                        raise ValueError(
                            f"Expected subject relevance pool '{pool_key}' with shape [N, R], got {pool.shape}"
                        )
                    if pool.shape[1] != self.num_rois:
                        raise ValueError(
                            f"Subject relevance pool '{pool_key}' width {pool.shape[1]} does not match num_rois={self.num_rois}"
                        )
                    subject_pool_dict[pool_key] = pool.copy()
                    subject_pool_means[pool_key] = pool.mean(axis=0).astype(np.float32, copy=True)
                if subject_pool_dict:
                    subject_installed[str(subject_id)] = subject_pool_dict
                    subject_installed_means[str(subject_id)] = subject_pool_means
        self.language_event_subject_relevance_pools = subject_installed
        self.language_event_subject_relevance_pool_means = subject_installed_means
        self.condition_relevance_source = str(source_name)

    def _start_is_allowed_with_offsets(self, subj_idx, start_pos, offsets):
        valid_mask = self.subject_valid_volume_masks.get(subj_idx)
        if valid_mask is not None:
            idx = start_pos + offsets
            if not bool(valid_mask[idx].all()):
                return False
        if self.allowed_ood_tags is None or self.ood_tags is None:
            return True
        idx = start_pos + offsets
        tags = self.ood_tags[subj_idx, idx]
        return bool(np.isin(tags, self.allowed_ood_tags).all())

    def _start_is_allowed(self, subj_idx, start_pos):
        return self._start_is_allowed_with_offsets(subj_idx, start_pos, self.sample_offsets)

    @staticmethod
    def _window_all_true(mask, offsets, num_starts):
        if num_starts <= 0:
            return np.zeros(0, dtype=bool)
        views = [mask[offset:offset + num_starts] for offset in offsets]
        return np.stack(views, axis=0).all(axis=0)

    def _compute_subject_start_pools(self):
        """Compute valid start positions per subject/run for random training."""
        pools = []
        for subj_idx in self.subject_indices:
            ts = self.file[f'timeseries/{self.atlas_names[0]}'][subj_idx]
            T = ts.shape[-1]
            max_start = T - self.total_seq_length
            if max_start < 0:
                continue

            starts = np.arange(max_start + 1, dtype=np.int32)
            valid_mask = self.subject_valid_volume_masks.get(subj_idx)
            use_ood_filter = self.allowed_ood_tags is not None and self.ood_tags is not None
            if valid_mask is not None or use_ood_filter:
                keep = np.ones(starts.shape[0], dtype=bool)
                if valid_mask is not None:
                    keep &= self._window_all_true(valid_mask, self.sample_offsets, starts.shape[0])
                if use_ood_filter:
                    tag_ok = np.isin(self.ood_tags[subj_idx], self.allowed_ood_tags)
                    keep &= self._window_all_true(tag_ok, self.sample_offsets, starts.shape[0])
                starts = starts[keep]
            if starts.size > 0:
                pools.append((subj_idx, starts))
        return pools

    def _load_subject_timeseries(self, subj_idx):
        ts_parts = [self.file[f'timeseries/{atlas}'][subj_idx] for atlas in self.atlas_names]
        return np.concatenate(ts_parts, axis=0).astype(np.float32)

    def _compute_subject_finite_volume_mask(self, subj_idx):
        valid = None
        for atlas in self.atlas_names:
            ts = self.file[f'timeseries/{atlas}'][subj_idx]
            atlas_valid = np.isfinite(ts).all(axis=0)
            valid = atlas_valid if valid is None else (valid & atlas_valid)
        return valid

    def _pretrain_soft_mass_is_valid(self, soft_mass):
        soft_mass = np.asarray(soft_mass, dtype=np.float64)
        return (soft_mass >= self.pretrain_mass_min) & (soft_mass < self.pretrain_mass_threshold)

    def _pretrain_condition_valid_mask(self, pretrain_condition_ts, num_volumes):
        _, soft_mass = self._compute_pretrain_weights_and_mass(pretrain_condition_ts)
        source_valid = self._pretrain_soft_mass_is_valid(soft_mass)
        cond_valid = np.zeros(num_volumes, dtype=bool)
        offset = int(self.pretrain_hrf_conditioning_offset)
        if offset == 0:
            cond_valid = source_valid
        elif offset < num_volumes:
            cond_valid[offset:] = source_valid[:-offset]
        return cond_valid

    def _compute_subject_valid_volume_masks(self):
        masks = {}
        for subj_idx in self.subject_indices:
            valid = self._compute_subject_finite_volume_mask(subj_idx)
            if self.condition_cont_source == 'pretrain_dynamics':
                ts = self._load_subject_timeseries(subj_idx)
                cond_ts = self._prepare_pretrain_condition_timeseries(ts)
                valid = valid & self._pretrain_condition_valid_mask(cond_ts, valid.shape[0])
            masks[subj_idx] = valid
        return masks

    def _compute_subject_distant_start_pools(self):
        pools = {}
        for subj_idx in self.subject_indices:
            ts = self.file[f'timeseries/{self.atlas_names[0]}'][subj_idx]
            T = ts.shape[-1]
            max_start = T - self.subject_total_seq_length
            if max_start < 0:
                continue
            starts = np.arange(max_start + 1, dtype=np.int32)
            valid_mask = self.subject_valid_volume_masks.get(subj_idx)
            use_ood_filter = self.allowed_ood_tags is not None and self.ood_tags is not None
            if valid_mask is not None or use_ood_filter:
                keep = np.ones(starts.shape[0], dtype=bool)
                if valid_mask is not None:
                    keep &= self._window_all_true(valid_mask, self.subject_sample_offsets, starts.shape[0])
                if use_ood_filter:
                    tag_ok = np.isin(self.ood_tags[subj_idx], self.allowed_ood_tags)
                    keep &= self._window_all_true(tag_ok, self.subject_sample_offsets, starts.shape[0])
                starts = starts[keep]
            if starts.size > 0:
                pools[subj_idx] = starts
        return pools

    @staticmethod
    def _window_gap(start_a, len_a, start_b, len_b):
        end_a = start_a + len_a - 1
        end_b = start_b + len_b - 1
        if end_a < start_b:
            return start_b - end_a - 1
        if end_b < start_a:
            return start_a - end_b - 1
        return -1

    def _sample_distant_start(self, subj_idx, main_start):
        assert self.subject_token_enabled
        assert subj_idx in self.subject_distant_start_pools, (
            f"No distant-start pool for subject index {subj_idx}"
        )
        starts = self.subject_distant_start_pools[subj_idx]
        main_len = self.total_seq_length
        distant_len = self.subject_total_seq_length
        gap_req = int(self.subject_min_gap)

        while True:
            gaps = np.asarray(
                [self._window_gap(main_start, main_len, int(s), distant_len) for s in starts],
                dtype=np.int32,
            )
            candidates = starts[gaps >= gap_req]
            if candidates.size > 0:
                pick = np.random.randint(0, candidates.size)
                return int(candidates[pick])
            if gap_req == 0:
                break
            gap_req = gap_req // 2

        raise RuntimeError(
            f"No distant window found for subj_idx={subj_idx}, main_start={main_start}, "
            f"main_len={main_len}, distant_len={distant_len}, min_gap={self.subject_min_gap}"
        )

    def _compute_anchored_samples(self):
        """Compute deterministic (subj_idx, start_pos) for anchored eval mode."""
        if self.sample_mode != 'anchored_eval':
            return []
        if self.anchor_tags is None:
            return []
        if self.anchor_tags_arr is None:
            return []
        if self.anchor_crop_index < 0 or self.anchor_crop_index >= self.sequence_length:
            return []

        samples = []
        seen = set()
        anchor_offset = self.anchor_crop_index * self.crop_length * self.input_stride

        for subj_idx in self.subject_indices:
            ts = self.file[f'timeseries/{self.atlas_names[0]}'][subj_idx]
            T = ts.shape[-1]
            max_start = T - self.total_seq_length
            if max_start < 0:
                continue

            run_tags = self.anchor_tags_arr[subj_idx]
            is_target = np.isin(run_tags, self.anchor_tags)
            if not np.any(is_target):
                continue
            prev = np.zeros_like(is_target, dtype=bool)
            prev[1:] = is_target[:-1]
            onset_idx = np.flatnonzero(is_target & (~prev))

            for anchor_tr in onset_idx.tolist():
                start_pos = int(anchor_tr - anchor_offset)
                if start_pos < 0 or start_pos > max_start:
                    continue
                if not self._start_is_allowed(subj_idx, start_pos):
                    continue
                key = (subj_idx, start_pos)
                if key in seen:
                    continue
                seen.add(key)
                samples.append(key)
        return samples

    def __len__(self):
        if self.sample_mode == 'anchored_eval':
            return len(self.anchored_samples)
        return len(self.subject_start_pools)

    def _zscore_roiwise(self, signal):
        """Z-score each ROI independently. Signal is [C, T]."""
        mean = np.nanmean(signal, axis=1, keepdims=True)
        std = np.nanstd(signal, axis=1, keepdims=True) + 1e-8
        return (signal - mean) / std

    def _get_crop_condition(self, subj_idx, crop_start, crop_end):
        """
        Determine the condition for a crop by argmax over volumes.
        """
        if self.condition_labels is None:
            return self.unlabeled_condition
        crop_labels = self.condition_labels[subj_idx, crop_start:crop_end]
        counts = np.bincount(crop_labels.astype(int), minlength=self.num_conditions)
        return int(np.argmax(counts))

    @staticmethod
    def _prepare_language_rows(arr, truncate_dim=None):
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected task embedding pool with shape [N, D], got {arr.shape}")
        if truncate_dim is not None:
            truncate_dim = int(truncate_dim)
            if arr.shape[1] < truncate_dim:
                raise ValueError(
                    f"Task embedding dim {arr.shape[1]} is smaller than requested truncate_dim={truncate_dim}"
                )
            arr = arr[:, :truncate_dim]
        return arr.astype(np.float32, copy=True)

    @staticmethod
    def _l2_normalize_rows_np(x, eps=1e-8):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norms, eps, None)

    @classmethod
    def _flatten_task_embedding_rows(cls, obj, truncate_dim=None):
        rows = []
        if isinstance(obj, dict):
            for value in obj.values():
                rows.extend(cls._flatten_task_embedding_rows(value, truncate_dim=truncate_dim))
            return rows
        rows.append(cls._prepare_language_rows(obj, truncate_dim=truncate_dim))
        return rows

    @classmethod
    def _load_language_pool_namespace(cls, path, embedding_namespace, truncate_dim=None):
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.shape == ():
            loaded = loaded.item()
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected dict-like task embeddings in {path}, got {type(loaded).__name__}")
        if embedding_namespace not in loaded:
            raise KeyError(f"Embedding namespace '{embedding_namespace}' not found in {path}")
        namespace_obj = loaded[embedding_namespace]
        if not isinstance(namespace_obj, dict):
            raise TypeError(
                f"Embedding namespace '{embedding_namespace}' must be a dict of pools, got {type(namespace_obj).__name__}"
            )

        raw_pools = {}
        for key, value in namespace_obj.items():
            raw_pools[str(key)] = cls._prepare_language_rows(value, truncate_dim=truncate_dim)
        if len(raw_pools) == 0:
            raise ValueError(f"Embedding namespace '{embedding_namespace}' has no pools")

        pool_dim = next(iter(raw_pools.values())).shape[1]
        for pool_name, pool in raw_pools.items():
            if pool.shape[1] != pool_dim:
                raise ValueError(
                    f"Inconsistent task embedding dims in namespace '{embedding_namespace}': "
                    f"{pool_name} has {pool.shape[1]}, expected {pool_dim}"
                )
            if pool.shape[0] == 0:
                raise ValueError(f"Task embedding pool '{pool_name}' is empty")

        return loaded, raw_pools

    @staticmethod
    def _as_path_list(value, context_name):
        if isinstance(value, str):
            if len(value.strip()) == 0:
                raise ValueError(f"{context_name} cannot be empty")
            return [value]
        if isinstance(value, (list, tuple)):
            paths = []
            for idx, path in enumerate(value):
                if not isinstance(path, str) or len(path.strip()) == 0:
                    raise TypeError(f"{context_name}[{idx}] must be a non-empty string")
                paths.append(path)
            if len(paths) == 0:
                raise ValueError(f"{context_name} cannot be empty")
            return paths
        raise TypeError(f"{context_name} must be a path string or list/tuple of path strings")

    def _init_condition_cont_language_events(self, cfg):
        if not isinstance(cfg, dict):
            raise TypeError("condition_cont_language_events must be a dict")

        h5_group = cfg.get('h5_group', 'events')
        embeddings_path = cfg.get('embeddings_path')
        embedding_namespace = cfg.get('embedding_namespace')
        event_map_cfg = cfg.get('event_map')
        relevance_h5_group = cfg.get('relevance_h5_group')
        relevance_ood_h5_group = cfg.get('relevance_ood_h5_group')
        truncate_dim = cfg.get('truncate_dim')

        variant_sampling = str(cfg.get('variant_sampling', 'random'))
        if variant_sampling not in ('random', 'fixed', 'average'):
            raise ValueError(
                "condition_cont_language_events.variant_sampling must be "
                "'random', 'fixed', or 'average'"
            )
        fixed_variant_idx = None
        if variant_sampling == 'fixed':
            fixed_variant_idx = int(cfg.get('fixed_variant_idx', 0))

        if embeddings_path is None and event_map_cfg is None:
            if embedding_namespace is None:
                embedding_namespace = 'h5_v2'
                cfg['embedding_namespace'] = embedding_namespace

            assert h5_group == 'events'
            assert embeddings_path is None
            assert event_map_cfg is None
            assert truncate_dim is None
            assert cfg.get('normalization') is None
            assert embedding_namespace == 'h5_v2'

            raw_pools = {}
            event_specs = []

            event_names = sorted(self.file['events'].keys())
            response_names = sorted(self.file['responses'].keys())
            for event_name in event_names:
                h5_key = f'events/{event_name}'
                self.language_event_arrays[h5_key] = self.file[h5_key][:]
                for piece in ('instruction', 'sensory'):
                    pool_key = f'{piece}:{event_name}'
                    raw_pools[pool_key] = self._prepare_language_rows(
                        self.file[f'embeddings/events/{piece}/{event_name}'][:]
                    )
                    event_specs.append({
                        'kind': 'pool',
                        'piece': piece,
                        'h5_key': h5_key,
                        'pool_key': pool_key,
                    })

            for response_name in response_names:
                h5_key = f'responses/{response_name}'
                self.language_event_arrays[h5_key] = self.file[h5_key][:]
                if response_name == 'no_response':
                    self.language_event_no_response_key = h5_key
                    continue
                if response_name == 'response_unknown':
                    self.language_event_unknown_response_key = h5_key
                    continue
                pool_key = f'response:{response_name}'
                raw_pools[pool_key] = self._prepare_language_rows(
                    self.file[f'embeddings/responses/{response_name}'][:]
                )
                event_specs.append({
                    'kind': 'pool',
                    'piece': 'response',
                    'h5_key': h5_key,
                    'pool_key': pool_key,
                })

            assert self.language_event_no_response_key is not None
            assert self.language_event_unknown_response_key is not None
            assert len(raw_pools) > 0

            pool_dim = next(iter(raw_pools.values())).shape[1]
            assert pool_dim == 1024
            for pool_key, pool in raw_pools.items():
                assert pool.shape[1] == pool_dim, pool_key
                assert pool.shape[0] > 0, pool_key
                if fixed_variant_idx is not None:
                    assert 0 <= fixed_variant_idx < pool.shape[0], pool_key

            self.language_event_pools = raw_pools
            self.language_event_pool_means = {
                pool_key: pool.mean(axis=0).astype(np.float32, copy=True)
                for pool_key, pool in raw_pools.items()
            }
            if relevance_h5_group is not None:
                assert self.condition_relevance_path is None
                assert isinstance(relevance_h5_group, str)
                assert relevance_h5_group in self.file
                relevance_group = self.file[relevance_h5_group]
                num_rows = int(self.file['long_subject_id'].shape[0])
                assert 'events' in relevance_group
                assert 'responses' in relevance_group
                for event_name in event_names:
                    rel = np.asarray(relevance_group[f'events/{event_name}'][:], dtype=np.float32)
                    assert rel.shape == (num_rows, self.num_rois), event_name
                    self.language_event_h5_relevance_events[f'events/{event_name}'] = rel
                for response_name in response_names:
                    if response_name in ('no_response', 'response_unknown'):
                        continue
                    rel = np.asarray(relevance_group[f'responses/{response_name}'][:], dtype=np.float32)
                    assert rel.shape == (num_rows, self.num_rois), response_name
                    self.language_event_h5_relevance_responses[f'responses/{response_name}'] = rel
                self.condition_relevance_source = str(relevance_h5_group)
            if relevance_ood_h5_group is not None and relevance_h5_group is not None:
                assert isinstance(relevance_ood_h5_group, str)
                assert relevance_ood_h5_group in self.file, (
                    f"OOD relevance group '{relevance_ood_h5_group}' not found in h5 file"
                )
                ood_group = self.file[relevance_ood_h5_group]
                assert 'events' in ood_group
                assert 'responses' in ood_group
                for event_name in event_names:
                    rel = np.asarray(ood_group[f'events/{event_name}'][:], dtype=np.float32)
                    assert rel.shape[1] == self.num_rois, event_name
                    self.language_event_h5_relevance_ood_events[f'events/{event_name}'] = rel
                for response_name in response_names:
                    if response_name in ('no_response', 'response_unknown'):
                        continue
                    rel = np.asarray(ood_group[f'responses/{response_name}'][:], dtype=np.float32)
                    assert rel.shape[1] == self.num_rois, response_name
                    self.language_event_h5_relevance_ood_responses[f'responses/{response_name}'] = rel
            if self.condition_relevance_path is not None:
                _, relevance_pools = self._load_language_pool_namespace(
                    self.condition_relevance_path,
                    embedding_namespace,
                    truncate_dim=None,
                )
                for pool_key, task_pool in raw_pools.items():
                    assert pool_key in relevance_pools, pool_key
                    relevance_pool = relevance_pools[pool_key]
                    assert relevance_pool.shape[0] == task_pool.shape[0], pool_key
                    assert relevance_pool.shape[1] == self.num_rois, pool_key
                self.language_event_relevance_pools = {
                    pool_key: relevance_pools[pool_key].astype(np.float32, copy=True)
                    for pool_key in raw_pools.keys()
                }
                self.language_event_relevance_pool_means = {
                    pool_key: pool.mean(axis=0).astype(np.float32, copy=True)
                    for pool_key, pool in self.language_event_relevance_pools.items()
                }

            self.language_event_specs = event_specs
            self.language_event_variant_sampling = variant_sampling
            self.language_event_fixed_variant_idx = fixed_variant_idx
            self.language_event_piece_dim = int(pool_dim)
            self.language_event_dim = int(3 * pool_dim + 2)
            return

        raise ValueError(
            "condition_cont_language_events supports only the current H5 events/responses layout"
        )

    def _init_pretrain_dynamics(self, cfg, condition_cont_dim=None):
        if not isinstance(cfg, dict):
            raise TypeError("pretrain_dynamics must be a dict")
        if self.tr_seconds is None or self.tr_seconds <= 0.0:
            raise ValueError("pretrain_dynamics requires explicit tr_seconds > 0")

        roi_paths = self._as_path_list(
            cfg.get('roi_embeddings_path'),
            "pretrain_dynamics.roi_embeddings_path",
        )
        truncate_dim = cfg.get('truncate_dim', condition_cont_dim)
        if truncate_dim is None:
            raise ValueError("pretrain_dynamics requires truncate_dim or condition_cont_dim")
        truncate_dim = int(truncate_dim)

        roi_rows = np.concatenate(
            [
                self._prepare_language_rows(np.load(path, allow_pickle=True), truncate_dim=truncate_dim)
                for path in roi_paths
            ],
            axis=0,
        )
        if roi_rows.shape[0] != self.num_rois:
            raise ValueError(
                f"pretrain_dynamics.roi_embeddings_path rows {roi_rows.shape[0]} "
                f"do not match num_rois={self.num_rois}"
            )

        norm_cfg = cfg.get('normalization')
        if not isinstance(norm_cfg, dict):
            raise TypeError("pretrain_dynamics.normalization must be a dict")
        if norm_cfg.get('mode') != 'roi_mean_l2':
            raise ValueError("pretrain_dynamics.normalization.mode must be 'roi_mean_l2'")

        roi_rows_l2 = self._l2_normalize_rows_np(roi_rows)
        mean = roi_rows_l2.mean(axis=0, keepdims=True)
        self.pretrain_roi_embeddings = mean_center_l2_normalize_rows_np(roi_rows_l2, mean)
        self.pretrain_condition_dim = int(self.pretrain_roi_embeddings.shape[1])
        self.pretrain_membership_center = np.float64(cfg.get('membership_center', 1.75))
        self.pretrain_membership_sharpness = np.float64(cfg.get('membership_sharpness', 7.0))
        self.pretrain_mass_threshold = np.float64(cfg['mass_threshold'])
        self.pretrain_mass_min = np.float64(cfg.get('mass_min', 1.0))
        self.pretrain_hrf_conditioning_offset = int(cfg['hrf_conditioning_offset'])
        self.pretrain_lowpass_hz = float(cfg.get('lowpass_hz', 0.1))
        if self.pretrain_lowpass_hz <= 0.0:
            raise ValueError("pretrain_dynamics.lowpass_hz must be > 0")

    @staticmethod
    def _fft_lowpass(signal, cutoff_hz, tr_seconds):
        freqs = np.fft.rfftfreq(signal.shape[1], d=float(tr_seconds))
        fft = np.fft.rfft(signal, axis=1)
        fft[:, freqs > float(cutoff_hz)] = 0.0
        return np.fft.irfft(fft, n=signal.shape[1], axis=1).astype(np.float32, copy=False)

    def _prepare_pretrain_condition_timeseries(self, ts):
        lowpassed = self._fft_lowpass(ts, cutoff_hz=self.pretrain_lowpass_hz, tr_seconds=self.tr_seconds)
        return self._zscore_roiwise(lowpassed)

    def _compute_pretrain_soft_weights(self, values):
        values = np.asarray(values, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-self.pretrain_membership_sharpness * (values - self.pretrain_membership_center)))

    def _compute_pretrain_weights_and_mass(self, values):
        weights = self._compute_pretrain_soft_weights(values)
        return weights, weights.sum(axis=0)

    def _pretrain_source_volume_idx(self, volume_idx):
        source_idx = int(volume_idx) - int(self.pretrain_hrf_conditioning_offset)
        if source_idx < 0:
            return None
        return source_idx

    def _pretrain_volume_is_valid(self, pretrain_condition_ts, volume_idx):
        source_idx = self._pretrain_source_volume_idx(volume_idx)
        if source_idx is None:
            return False
        _, soft_mass = self._compute_pretrain_weights_and_mass(pretrain_condition_ts[:, source_idx])
        return bool(self._pretrain_soft_mass_is_valid(soft_mass))

    def _get_pretrain_condition_parts(self, pretrain_condition_ts, volume_idx):
        source_idx = self._pretrain_source_volume_idx(volume_idx)
        if source_idx is None:
            raise ValueError(f"Invalid pretrain_dynamics source volume for volume_idx={volume_idx}")
        weights, soft_mass = self._compute_pretrain_weights_and_mass(pretrain_condition_ts[:, source_idx])
        if not bool(self._pretrain_soft_mass_is_valid(soft_mass)):
            raise ValueError(f"Invalid pretrain_dynamics soft mass {soft_mass} for volume_idx={volume_idx}")

        cond_cont = np.matmul(weights, self.pretrain_roi_embeddings.astype(np.float64, copy=False))
        cond_norm = np.linalg.norm(cond_cont)
        cond_cont = (cond_cont / max(cond_norm, 1e-8)).astype(np.float32, copy=False)
        relevance_scores = weights.astype(np.float32, copy=False)
        return cond_cont, relevance_scores

    def _sample_language_event_variant_idx(self, pool_key):
        pool = self.language_event_pools[pool_key]
        if self.language_event_variant_sampling == 'fixed':
            return self.language_event_fixed_variant_idx
        if self.language_event_variant_sampling == 'average':
            return None
        return int(np.random.randint(0, pool.shape[0]))

    def _get_language_event_pool_row(self, pool_key, variant_idx):
        if variant_idx is None:
            return self.language_event_pool_means[pool_key]
        return self.language_event_pools[pool_key][variant_idx]

    def _get_language_event_relevance_row(self, pool_key, variant_idx, subject_id=None):
        if subject_id is not None:
            subject_id = str(subject_id)
            subject_pools = self.language_event_subject_relevance_pools.get(subject_id)
            if subject_pools is not None and pool_key in subject_pools:
                if variant_idx is None:
                    return self.language_event_subject_relevance_pool_means[subject_id][pool_key]
                return subject_pools[pool_key][variant_idx]
        if variant_idx is None:
            return self.language_event_relevance_pool_means[pool_key]
        return self.language_event_relevance_pools[pool_key][variant_idx]

    def _sample_language_event_variants_for_sequence(self):
        # Sample once per sequence so phrasing is consistent across all volumes in the sample.
        return {
            pool_key: self._sample_language_event_variant_idx(pool_key)
            for pool_key in self.language_event_pools.keys()
        }

    @staticmethod
    def _gamma_pdf(x, shape):
        x = np.asarray(x, dtype=np.float64)
        out = np.zeros_like(x, dtype=np.float64)
        pos = x > 0.0
        if np.any(pos):
            xp = x[pos]
            out[pos] = np.exp((shape - 1.0) * np.log(xp) - xp - math.lgamma(shape))
        return out.astype(np.float32)

    def _build_splice_recovery_weights(self, generation_frames):
        # One generation step advances by crop_length * input_stride raw volumes.
        step_seconds = self.splice_tr_seconds * self.crop_length * self.input_stride
        t = np.arange(generation_frames, dtype=np.float32) * np.float32(step_seconds)
        hrf = self._gamma_pdf(t, shape=self.splice_recovery_gamma_shape) - np.float32(0.35) * self._gamma_pdf(
            t,
            shape=16.0,
        )
        rising = np.clip(hrf / np.max(hrf), 0.0, 1.0).astype(np.float32)
        rising = np.maximum.accumulate(rising)
        return rising

    def _load_splice_onset_arrays(self, n_rows):
        arrays = []
        for path in self.splice_onset_event_paths:
            if path not in self.file:
                raise KeyError(f"splice_onset_event_paths entry not found in H5: {path}")
            arr = np.asarray(self.file[path][:], dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"{path} must be 2D [N, T], got shape {arr.shape}")
            if arr.shape[0] != n_rows:
                raise ValueError(f"{path} first dimension {arr.shape[0]} does not match N_runs={n_rows}")
            arrays.append(arr)
        return arrays

    def _compute_subject_splice_onset_pools(self):
        assert self.splice_context_frames is not None
        assert self.splice_generation_frames is not None
        assert self.splice_onset_arrays is not None

        pools = {}
        boundary_offsets = {
            splice_point: (self.splice_context_frames + splice_point) * self.crop_length * self.input_stride
            for splice_point in range(1, self.splice_generation_frames)
        }

        for subj_idx in self.subject_indices:
            ts = self.file[f'timeseries/{self.atlas_names[0]}'][subj_idx]
            max_start = ts.shape[-1] - self.total_seq_length
            if max_start < 0:
                continue

            onset_mask = np.zeros(ts.shape[-1], dtype=bool)
            for arr in self.splice_onset_arrays:
                curr = np.asarray(arr[subj_idx], dtype=np.float32)
                prev = np.zeros_like(curr)
                prev[1:] = curr[:-1]
                onset_mask |= (prev <= self.splice_onset_threshold) & (curr > self.splice_onset_threshold)

            onset_idx = np.flatnonzero(onset_mask)
            if onset_idx.size == 0:
                continue

            per_run = {}
            for splice_point, boundary_offset in boundary_offsets.items():
                starts = onset_idx.astype(np.int64) - np.int64(boundary_offset)
                starts = starts[(starts >= 0) & (starts <= max_start)]
                if starts.size == 0:
                    continue
                starts = np.unique(starts.astype(np.int32, copy=False))
                keep = [self._start_is_allowed(subj_idx, int(start_pos)) for start_pos in starts]
                starts = starts[np.asarray(keep, dtype=bool)]
                if starts.size > 0:
                    per_run[splice_point] = starts
            if per_run:
                pools[subj_idx] = per_run

        return pools

    def _get_volume_language_event_condition_parts(self, subj_idx, volume_idx, language_event_sequence_variants=None):
        """Return mixed condition pieces for one volume from language-event specs.

        Returns:
            cond_disc_id: int discrete condition id placeholder (kept for compatibility; 0 when inactive)
            cond_disc_weight: [num_conditions] additive weights for discrete embeddings
            cond_cont: [Dc] continuous condition vector
        """
        instruction = np.zeros(self.language_event_piece_dim, dtype=np.float32)
        sensory = np.zeros(self.language_event_piece_dim, dtype=np.float32)
        response = np.zeros(self.language_event_piece_dim, dtype=np.float32)
        if language_event_sequence_variants is None:
            language_event_sequence_variants = self._sample_language_event_variants_for_sequence()
        for spec in self.language_event_specs:
            value = np.float32(self.language_event_arrays[spec['h5_key']][subj_idx, volume_idx])
            if value == 0.0:
                continue
            variant_idx = language_event_sequence_variants[spec['pool_key']]
            emb = self._get_language_event_pool_row(spec['pool_key'], variant_idx)
            piece = spec['piece']
            if piece == 'instruction':
                instruction += value * emb
                continue
            if piece == 'sensory':
                sensory += value * emb
                continue
            assert piece == 'response'
            response += value * emb
        response_special = np.asarray(
            [
                self.language_event_arrays[self.language_event_no_response_key][subj_idx, volume_idx],
                self.language_event_arrays[self.language_event_unknown_response_key][subj_idx, volume_idx],
            ],
            dtype=np.float32,
        )
        cond_cont = np.concatenate([instruction, sensory, response, response_special], axis=0)
        cond_disc_weight = np.zeros(self.num_conditions, dtype=np.float32)
        return -1, cond_disc_weight, cond_cont

    def _get_volume_relevance_scores(
        self,
        subj_idx,
        volume_idx,
        language_event_sequence_variants=None,
        relevance_subject_id='auto',
        relevance_source='train',
    ):
        def _sum_scores(event_scores, response_scores, row_idx):
            scores = np.zeros(self.num_rois, dtype=np.float32)
            for h5_key, rel_ds in event_scores.items():
                value = np.float32(self.language_event_arrays[h5_key][subj_idx, volume_idx])
                if value == 0.0:
                    continue
                scores += value * np.asarray(rel_ds[row_idx], dtype=np.float32)
            for h5_key, rel_ds in response_scores.items():
                value = np.float32(self.language_event_arrays[h5_key][subj_idx, volume_idx])
                if value == 0.0:
                    continue
                scores += value * np.asarray(rel_ds[row_idx], dtype=np.float32)
            return scores

        if self.language_event_h5_relevance_events or self.language_event_h5_relevance_responses:
            true_roi = _sum_scores(
                self.language_event_h5_relevance_events,
                self.language_event_h5_relevance_responses,
                subj_idx,
            )
            if relevance_source == 'train':
                return true_roi, 0
            if relevance_source == 'true_roi':
                return true_roi, 0
            if relevance_source == 'ood_roi':
                if not self.language_event_h5_relevance_ood_events:
                    raise ValueError("relevance_source='ood_roi' requires relevance_ood_h5_group")
                return _sum_scores(
                    self.language_event_h5_relevance_ood_events,
                    self.language_event_h5_relevance_ood_responses,
                    0,
                ), 0
            raise ValueError(f"Unknown relevance_source: {relevance_source}")
        scores = np.zeros((3, self.num_rois), dtype=np.float32)
        if not self.language_event_relevance_pools:
            return scores, 0
        if relevance_subject_id == 'auto' and self.language_event_subject_relevance_pools:
            relevance_subject_id = self.subject_id_by_index[int(subj_idx)]
        if language_event_sequence_variants is None:
            language_event_sequence_variants = self._sample_language_event_variants_for_sequence()
        for spec in self.language_event_specs:
            value = np.float32(self.language_event_arrays[spec['h5_key']][subj_idx, volume_idx])
            if value == 0.0 or spec['kind'] not in ('pool', 'alias'):
                continue
            variant_idx = language_event_sequence_variants[spec['pool_key']]
            rel = value * self._get_language_event_relevance_row(
                spec['pool_key'],
                variant_idx,
                subject_id=relevance_subject_id,
            )
            piece = spec['piece']
            if piece == 'instruction':
                scores[0] += rel
                continue
            if piece == 'sensory':
                scores[1] += rel
                continue
            assert piece == 'response'
            scores[2] += rel
        return scores, 0

    def _get_volume_condition_cont(self, subj_idx, volume_idx, language_event_sequence_variants=None):
        """Get continuous condition vector for one volume index."""
        if self.condition_cont_source == 'dense':
            return np.asarray(self.condition_cont_labels[subj_idx, volume_idx], dtype=np.float32)

        if self.condition_cont_source == 'indexed':
            emb_idx = int(self.condition_cont_idx_labels[subj_idx, volume_idx])
            if emb_idx < 0 or emb_idx >= self.condition_cont_embeddings.shape[0]:
                return np.zeros(self.condition_cont_dim, dtype=np.float32)
            scale = float(self.condition_cont_scale_labels[subj_idx, volume_idx])
            if scale <= 0:
                return np.zeros(self.condition_cont_dim, dtype=np.float32)
            emb = np.asarray(self.condition_cont_embeddings[emb_idx], dtype=np.float32)
            return emb * np.float32(scale)

        if self.condition_cont_source == 'language_event_pool':
            _, _, vec = self._get_volume_language_event_condition_parts(
                subj_idx, volume_idx, language_event_sequence_variants=language_event_sequence_variants
            )
            return vec

        if self.condition_cont_source == 'event_mapping':
            vec = np.zeros(self.condition_cont_dim, dtype=np.float32)
            for mapping in self.event_mappings:
                value = float(self.event_arrays[mapping['h5_key']][subj_idx, volume_idx])
                transform = mapping['transform']
                if transform == 'identity':
                    vec[mapping['dim']] += np.float32(value)
                elif transform == 'sign_split':
                    pos_dim, neg_dim = mapping['dims']
                    vec[pos_dim] += np.float32(max(0.0, value))
                    vec[neg_dim] += np.float32(max(0.0, -value))
                else:  # onehot
                    cls = int(value)
                    dims = mapping['dims']
                    if 0 <= cls < len(dims):
                        vec[dims[cls]] += np.float32(1.0)
            return vec

        if self.condition_cont_source == 'none':
            return np.zeros(self.condition_cont_dim, dtype=np.float32)
        return np.zeros(self.condition_cont_dim, dtype=np.float32)

    def _extract_sequence_from_timeseries(self, ts, start_pos):
        indices = start_pos + self.sample_offsets
        seq = ts[:, indices]  # [num_rois, T*crop_length]
        num_rois = seq.shape[0]
        # [num_rois, T*crop_length] -> [T, num_rois, crop_length]
        seq = seq.reshape(num_rois, self.sequence_length, self.crop_length)
        return np.transpose(seq, (1, 0, 2)).astype(np.float32, copy=False)

    def _build_conditions_for_sequence(
        self,
        subj_idx,
        start_pos,
        pretrain_condition_ts=None,
        relevance_subject_id='auto',
        relevance_source='train',
    ):
        # For continuous mode, condition uses the first volume index of each crop.
        language_event_sequence_variants = None
        if self.condition_mode == 'cont' and self.condition_cont_source == 'language_event_pool':
            language_event_sequence_variants = self._sample_language_event_variants_for_sequence()

        conditions_disc = []
        conditions_disc_weight = []
        conditions_cont = []
        conditions_mode = []
        relevance_scores = [] if self.has_direct_relevance_scores() else None
        relevance_embedding_type = 0
        for crop_idx in range(self.sequence_length):
            crop_start_idx = crop_idx * self.crop_length
            crop_start = start_pos + crop_start_idx * self.input_stride
            crop_end = start_pos + (crop_start_idx + self.crop_length - 1) * self.input_stride + 1
            if self.condition_mode == 'cont':
                if self.condition_cont_source == 'language_event_pool':
                    cond_disc, cond_disc_weight, cond_cont = self._get_volume_language_event_condition_parts(
                        subj_idx,
                        crop_start,
                        language_event_sequence_variants=language_event_sequence_variants,
                    )
                    cond_relevance = None
                elif self.condition_cont_source == 'pretrain_dynamics':
                    if pretrain_condition_ts is None:
                        raise ValueError("pretrain_dynamics requires pretrain_condition_ts in _build_conditions_for_sequence")
                    cond_cont, cond_relevance = self._get_pretrain_condition_parts(
                        pretrain_condition_ts,
                        crop_start,
                    )
                    cond_disc = -1
                    cond_disc_weight = np.zeros(self.num_conditions, dtype=np.float32)
                else:
                    cond_cont = self._get_volume_condition_cont(
                        subj_idx,
                        crop_start,
                        language_event_sequence_variants=language_event_sequence_variants,
                    )
                    cond_relevance = None
                    cond_disc = -1
                    cond_disc_weight = np.zeros(self.num_conditions, dtype=np.float32)
                conditions_disc.append(cond_disc)
                conditions_disc_weight.append(cond_disc_weight)
                conditions_cont.append(cond_cont)
                conditions_mode.append(1)
                if relevance_scores is not None:
                    if self.condition_cont_source == 'language_event_pool':
                        cond_relevance, relevance_embedding_type = self._get_volume_relevance_scores(
                            subj_idx,
                            crop_start,
                            language_event_sequence_variants=language_event_sequence_variants,
                            relevance_subject_id=relevance_subject_id,
                            relevance_source=relevance_source,
                        )
                        relevance_scores.append(cond_relevance)
                    elif self.condition_cont_source == 'pretrain_dynamics':
                        relevance_scores.append(cond_relevance)
                    else:
                        relevance_scores.append(np.zeros(self.num_rois, dtype=np.float32))
                continue

            cond_disc = self._get_crop_condition(subj_idx, crop_start, crop_end)
            conditions_disc.append(cond_disc)
            conditions_disc_weight.append(np.zeros(self.num_conditions, dtype=np.float32))
            conditions_cont.append(np.zeros(self.condition_cont_dim, dtype=np.float32))
            conditions_mode.append(0)
            if relevance_scores is not None:
                relevance_scores.append(np.zeros(self.num_rois, dtype=np.float32))

        relevance_scores_np = None
        if relevance_scores is not None:
            relevance_scores_np = np.stack(relevance_scores, axis=0).astype(np.float32)

        return (
            np.asarray(conditions_disc, dtype=np.int64),
            np.stack(conditions_disc_weight, axis=0).astype(np.float32),
            np.stack(conditions_cont, axis=0).astype(np.float32),
            np.asarray(conditions_mode, dtype=np.int64),
            relevance_scores_np,
            np.int64(relevance_embedding_type),
        )

    def _sample_splice_gen_weights(self, splice_point):
        assert self.splice_generation_frames is not None
        weights = np.ones(self.splice_generation_frames, dtype=np.float32)
        if splice_point is None:
            return weights
        tail_len = self.splice_generation_frames - splice_point
        if tail_len <= 0:
            return weights
        weights[splice_point:] = self.splice_recovery_weights[:tail_len]
        weights *= np.float32(self.splice_generation_frames / np.sum(weights))
        return weights

    def sample_item(self, idx, p_splice):
        """
        Returns:
            signal: [T, num_rois, crop_length] sequence of crops
            condition: [T] categorical condition indices per crop
        """
        if self.sample_mode == 'anchored_eval':
            subj_idx, start_pos = self.anchored_samples[idx]
            starts = None
        else:
            subj_idx, starts = self.subject_start_pools[idx]
            choice_idx = np.random.randint(0, len(starts))
            start_pos = int(starts[choice_idx])

        ts_raw = self._load_subject_timeseries(subj_idx)
        pretrain_condition_ts = None
        if self.condition_cont_source == 'pretrain_dynamics':
            pretrain_condition_ts = self._prepare_pretrain_condition_timeseries(ts_raw)

        # Z-score (on full timeseries before striding)
        ts = self._zscore_roiwise(ts_raw)

        seq = self._extract_sequence_from_timeseries(ts, start_pos)
        signal = torch.from_numpy(seq)
        splice_point = None
        generation_loss_weight = None

        (
            condition_disc_np,
            condition_disc_weight_np,
            condition_cont_np,
            condition_mode_np,
            relevance_scores_np,
            relevance_embedding_type_np,
        ) = (
            self._build_conditions_for_sequence(
                subj_idx,
                start_pos,
                pretrain_condition_ts=pretrain_condition_ts,
            )
        )

        do_splice = (
            self.sample_mode == 'random_safe'
            and p_splice > 0.0
            and np.random.rand() < p_splice
        )
        if do_splice:
            assert starts is not None
            if self.splice_onset_only:
                donor_pool = self.splice_onset_donor_pools.get(subj_idx, {})
                valid_splice_points = [
                    candidate_splice_point
                    for candidate_splice_point, donor_starts in donor_pool.items()
                    if donor_starts.size > 0
                ]
                if valid_splice_points:
                    splice_point = int(valid_splice_points[np.random.randint(0, len(valid_splice_points))])
                    donor_starts = donor_pool[splice_point]
                    start_pos_2 = int(donor_starts[np.random.randint(0, len(donor_starts))])
                else:
                    start_pos_2 = None
            else:
                start_pos_2 = int(starts[np.random.randint(0, len(starts))])

            if start_pos_2 is not None:
                seq2 = self._extract_sequence_from_timeseries(ts, start_pos_2)
                (
                    cond_disc_2,
                    cond_disc_weight_2,
                    cond_cont_2,
                    cond_mode_2,
                    relevance_scores_2,
                    _,
                ) = self._build_conditions_for_sequence(
                    subj_idx,
                    start_pos_2,
                    pretrain_condition_ts=pretrain_condition_ts,
                )

                # splice_point is in generation-frame coordinates: [1, generation_frames-1]
                if splice_point is None:
                    splice_point = int(np.random.randint(1, self.splice_generation_frames))
                splice_idx = self.splice_context_frames + splice_point
                seq[splice_idx:] = seq2[splice_idx:]
                condition_disc_np[splice_idx:] = cond_disc_2[splice_idx:]
                condition_disc_weight_np[splice_idx:] = cond_disc_weight_2[splice_idx:]
                condition_cont_np[splice_idx:] = cond_cont_2[splice_idx:]
                condition_mode_np[splice_idx:] = cond_mode_2[splice_idx:]
                if relevance_scores_np is not None:
                    relevance_scores_np[splice_idx:] = relevance_scores_2[splice_idx:]
                signal = torch.from_numpy(seq)

        if self.splice_generation_frames is not None:
            generation_loss_weight = torch.from_numpy(self._sample_splice_gen_weights(splice_point))

        distant_signal = None
        distant_start = None
        if self.subject_token_enabled:
            num_rois = ts.shape[0]
            distant_start = self._sample_distant_start(subj_idx, start_pos)
            distant_indices = distant_start + self.subject_sample_offsets
            distant_seq = ts[:, distant_indices]
            distant_seq = distant_seq.reshape(num_rois, self.subject_context_length, self.crop_length)
            distant_seq = np.transpose(distant_seq, (1, 0, 2))  # [T_subj, num_rois, crop_length]
            distant_signal = torch.from_numpy(distant_seq)

        condition_disc = torch.from_numpy(condition_disc_np)  # [T]
        condition_disc_weight = torch.from_numpy(condition_disc_weight_np)  # [T, C]
        condition_cont = torch.from_numpy(condition_cont_np)  # [T, Dc]
        condition_mode = torch.from_numpy(condition_mode_np)  # [T]
        relevance_scores = (
            torch.from_numpy(relevance_scores_np) if relevance_scores_np is not None else None
        )  # [T, R]

        out = {
            'signal': signal,
            # Legacy key retained for backward compatibility
            'condition': condition_disc,
            'condition_disc': condition_disc,
            'condition_disc_weight': condition_disc_weight,
            'condition_cont': condition_cont,
            'condition_mode': condition_mode,
        }
        if relevance_scores is not None:
            out['relevance_scores'] = relevance_scores
            out['relevance_embedding_type'] = torch.tensor(relevance_embedding_type_np, dtype=torch.long)
        if generation_loss_weight is not None:
            out['generation_loss_weight'] = generation_loss_weight
        if distant_signal is not None:
            out['distant_signal'] = distant_signal
        if self.age_values is not None:
            out['age'] = torch.tensor(self.age_values[subj_idx], dtype=torch.float32)
        if self.sex_values is not None:
            out['sex'] = torch.tensor(self.sex_values[subj_idx], dtype=torch.long)
        if self.motion_values is not None:
            out['motion'] = torch.tensor(self.motion_values[subj_idx], dtype=torch.float32)
        if self.field_strength_idx is not None:
            out['field_strength'] = torch.tensor(self.field_strength_idx, dtype=torch.long)
        return out

    def __getitem__(self, idx):
        return self.sample_item(idx, self.p_splice)

    def __del__(self):
        if hasattr(self, 'file') and self.file:
            self.file.close()
