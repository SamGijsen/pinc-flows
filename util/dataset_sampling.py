import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler

from datasets.dynamics_dataset import SequenceDataset


class WeightedDistributedSampler(Sampler):
    def __init__(self, weights, num_replicas, rank, num_samples, seed=0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=g,
        )
        return iter(indices[self.rank:self.total_size:self.num_replicas].tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


class SubjectSpliceConcatDataset(Dataset):
    def __init__(self, datasets, subject_regex):
        self.datasets = list(datasets)
        self.subject_re = re.compile(subject_regex)
        assert self.subject_re.groups == 1
        self.cumulative_sizes = np.cumsum([len(ds) for ds in self.datasets])

        self.subject_by_item = []
        self.subject_pools = {}
        for dataset_idx, ds in enumerate(self.datasets):
            assert ds.sample_mode == 'random_safe'
            assert not ds.splice_onset_only
            subjects = []
            for local_idx, (subj_idx, _) in enumerate(ds.subject_start_pools):
                match = self.subject_re.match(ds.subject_id_by_index[int(subj_idx)])
                assert match is not None
                subject = match.group(1)
                subjects.append(subject)
                self.subject_pools.setdefault(subject, []).append((dataset_idx, local_idx))
            self.subject_by_item.append(subjects)

    def __len__(self):
        return int(self.cumulative_sizes[-1])

    def _resolve_index(self, idx):
        dataset_idx = int(np.searchsorted(self.cumulative_sizes, idx, side='right'))
        prev = 0 if dataset_idx == 0 else int(self.cumulative_sizes[dataset_idx - 1])
        return dataset_idx, int(idx - prev)

    def __getitem__(self, idx):
        dataset_idx, local_idx = self._resolve_index(idx)
        ds = self.datasets[dataset_idx]
        item = ds.sample_item(local_idx, 0.0)

        if ds.p_splice == 0.0 or np.random.rand() >= ds.p_splice:
            return item

        assert ds.splice_context_frames is not None
        assert ds.splice_generation_frames is not None
        subject = self.subject_by_item[dataset_idx][local_idx]
        donor_dataset_idx, donor_local_idx = self.subject_pools[subject][
            np.random.randint(0, len(self.subject_pools[subject]))
        ]
        donor_ds = self.datasets[donor_dataset_idx]
        donor = donor_ds.sample_item(donor_local_idx, 0.0)

        splice_point = int(np.random.randint(1, ds.splice_generation_frames))
        splice_idx = ds.splice_context_frames + splice_point

        item['signal'][splice_idx:] = donor['signal'][splice_idx:]
        item['condition_disc'][splice_idx:] = donor['condition_disc'][splice_idx:]
        item['condition'] = item['condition_disc']
        item['condition_disc_weight'][splice_idx:] = donor['condition_disc_weight'][splice_idx:]
        item['condition_cont'][splice_idx:] = donor['condition_cont'][splice_idx:]
        item['condition_mode'][splice_idx:] = donor['condition_mode'][splice_idx:]
        if 'relevance_scores' in item:
            item['relevance_scores'][splice_idx:] = donor['relevance_scores'][splice_idx:]
        item['generation_loss_weight'] = torch.from_numpy(ds._sample_splice_gen_weights(splice_point))
        return item


def _resolved_path(path):
    return str(Path(path).resolve())


def expand_dataset_specs(data_cfg, split_name, data_paths, reserved_paths=None):
    if reserved_paths is None:
        reserved_paths = set()

    specs_key = f'{split_name}_datasets'
    spec_items = data_cfg.get(specs_key)
    if spec_items is not None:
        if not isinstance(spec_items, (list, tuple)):
            raise TypeError(f"'{specs_key}' must be a list when set")
        dataset_specs = []
        for spec in spec_items:
            if isinstance(spec, str):
                dataset_specs.append({'path': spec})
                continue
            if isinstance(spec, dict):
                if spec.get('path') is None and spec.get('dir_path') is None:
                    raise ValueError(f"Each entry in '{specs_key}' must define 'path' or 'dir_path'")
                dataset_specs.append(spec)
                continue
            raise TypeError(f"Entries in '{specs_key}' must be str or dict")
    else:
        if data_paths is None:
            raise ValueError(
                f"Missing dataset source for split='{split_name}'. "
                f"Set '{specs_key}' or '{split_name}_path'/'train_path'."
            )
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        dataset_specs = [{'path': data_path} for data_path in data_paths]

    expanded_specs = []
    for spec_idx, spec in enumerate(dataset_specs):
        if 'dir_path' not in spec:
            if split_name != 'train' and 'weight' in spec:
                raise ValueError(f"data.{specs_key}[{spec_idx}].weight is only supported for train datasets")
            expanded = dict(spec)
            expanded['path'] = _resolved_path(expanded['path'])
            if split_name == 'train' and expanded['path'] in reserved_paths:
                raise ValueError(
                    f"Train dataset path overlaps validation dataset path: {expanded['path']}"
                )
            expanded['_source_id'] = f'{split_name}:spec:{spec_idx}'
            expanded['_source_weight'] = float(spec.get('weight', 1.0))
            expanded_specs.append(expanded)
            continue

        dir_path = Path(spec['dir_path'])
        if not dir_path.is_dir():
            raise ValueError(f"{dir_path} is not a directory")
        h5_paths = sorted(str(path) for path in dir_path.glob('*.h5'))
        if len(h5_paths) == 0:
            raise ValueError(f"{dir_path} does not contain any .h5 files")
        if split_name != 'train' and 'weight' in spec:
            raise ValueError(f"data.{specs_key}[{spec_idx}].weight is only supported for train datasets")

        source_id = f'{split_name}:dir:{spec_idx}'
        source_weight = float(spec.get('weight', 1.0))
        shared = {k: v for k, v in spec.items() if k not in ('dir_path', 'path')}
        for data_path in h5_paths:
            data_path = _resolved_path(data_path)
            if split_name == 'train' and data_path in reserved_paths:
                continue
            expanded = dict(shared)
            expanded['path'] = data_path
            expanded['_source_id'] = source_id
            expanded['_source_weight'] = source_weight
            expanded_specs.append(expanded)
    return expanded_specs


def _build_split_defaults(
    split_name,
    data_cfg,
    dynamics_cfg,
    tokenizer_cfg,
    default_input_stride,
    default_anchor_crop_index,
):
    def _split_or_global(key, default=None):
        split_key = f'{split_name}_{key}'
        if split_key in data_cfg:
            return data_cfg.get(split_key)
        return data_cfg.get(key, default)

    return {
        'default_input_stride_local': _split_or_global('input_stride', default_input_stride),
        'default_atlas_names': _split_or_global('atlas_names', data_cfg.get('atlas_names')),
        'default_sample_mode': _split_or_global('sample_mode', 'random_safe'),
        'default_ood_tag_label_name': _split_or_global('ood_tag_label_name'),
        'default_allowed_ood_tags': _split_or_global('allowed_ood_tags'),
        'default_anchor_tag_label_name': _split_or_global('anchor_tag_label_name', 'labels/ood_tag_raw'),
        'default_anchor_tags': _split_or_global('anchor_tags'),
        'default_anchor_crop_index_local': _split_or_global('anchor_crop_index', default_anchor_crop_index),
        'default_condition_mode': _split_or_global('condition_mode', 'discrete'),
        'default_condition_label_name': _split_or_global('condition_label_name'),
        'default_condition_cont_label_name': _split_or_global('condition_cont_label_name'),
        'default_condition_cont_idx_label_name': _split_or_global('condition_cont_idx_label_name'),
        'default_condition_cont_scale_label_name': _split_or_global('condition_cont_scale_label_name'),
        'default_condition_cont_embeddings_name': _split_or_global('condition_cont_embeddings_name'),
        'default_condition_cont_language_events': _split_or_global('condition_cont_language_events'),
        'default_pretrain_dynamics': _split_or_global('pretrain_dynamics'),
        'default_condition_cont_dim': _split_or_global(
        'condition_cont_dim',
        dynamics_cfg.get('condition_cont_dim'),
        ),
        'default_unlabeled_condition': _split_or_global('unlabeled_condition', 0),
        'default_event_mapping': _split_or_global('event_mapping'),
        'default_tr_seconds': _split_or_global('tr_seconds'),
        'default_age_label_name': _split_or_global('age_label_name'),
        'default_sex_label_name': _split_or_global('sex_label_name'),
        'default_motion_label_name': _split_or_global('motion_label_name'),
        'default_field_strength_t': _split_or_global('field_strength_t'),
        'default_p_splice': _split_or_global('p_splice', 0.0),
        'default_splice_tr_seconds': _split_or_global('splice_tr_seconds', 0.72),
        'default_splice_recovery_gamma_shape': _split_or_global('splice_recovery_gamma_shape', 6.0),
        'default_splice_onset_only': _split_or_global('splice_onset_only', False),
        'default_splice_onset_event_paths': _split_or_global('splice_onset_event_paths'),
        'default_splice_onset_threshold': _split_or_global('splice_onset_threshold', 1e-6),
        'default_splice_scope': _split_or_global('splice_scope', 'run'),
        'default_splice_subject_regex': _split_or_global('splice_subject_regex', r'^([^_]+)'),
        'crop_length': tokenizer_cfg.get('input_timesteps', 3),
    }


def _build_dataset(
    spec,
    *,
    defaults,
    subject_ids_path,
    sequence_length,
    num_conditions,
    subject_token_enabled,
    subject_context_length,
    subject_min_gap,
    splice_context_frames,
    splice_generation_frames,
    relevance_mode,
    relevance_precomputed_path,
    relevance_h5_group,
    relevance_ood_h5_group,
    unconditioned_pretraining,
):
    def _pick(key, default):
        return spec[key] if key in spec else default

    data_path = spec['path']
    input_stride_local = _pick('input_stride', defaults['default_input_stride_local'])
    atlas_names = _pick('atlas_names', defaults['default_atlas_names'])

    sample_mode = _pick('sample_mode', defaults['default_sample_mode'])
    ood_tag_label_name = _pick('ood_tag_label_name', defaults['default_ood_tag_label_name'])
    allowed_ood_tags = _pick('allowed_ood_tags', defaults['default_allowed_ood_tags'])
    anchor_tag_label_name = _pick('anchor_tag_label_name', defaults['default_anchor_tag_label_name'])
    anchor_tags = _pick('anchor_tags', defaults['default_anchor_tags'])
    anchor_crop_index = _pick('anchor_crop_index', defaults['default_anchor_crop_index_local'])

    condition_mode = _pick('condition_mode', defaults['default_condition_mode'])
    condition_label_name = _pick('condition_label_name', defaults['default_condition_label_name'])
    condition_cont_label_name = _pick('condition_cont_label_name', defaults['default_condition_cont_label_name'])
    condition_cont_idx_label_name = _pick('condition_cont_idx_label_name', defaults['default_condition_cont_idx_label_name'])
    condition_cont_scale_label_name = _pick('condition_cont_scale_label_name', defaults['default_condition_cont_scale_label_name'])
    condition_cont_embeddings_name = _pick('condition_cont_embeddings_name', defaults['default_condition_cont_embeddings_name'])
    condition_cont_language_events = _pick('condition_cont_language_events', defaults['default_condition_cont_language_events'])
    pretrain_dynamics_cfg = _pick('pretrain_dynamics', defaults['default_pretrain_dynamics'])
    condition_cont_dim = _pick('condition_cont_dim', defaults['default_condition_cont_dim'])
    unlabeled_condition = _pick('unlabeled_condition', defaults['default_unlabeled_condition'])
    event_mapping = _pick('event_mapping', defaults['default_event_mapping'])
    tr_seconds = _pick('tr_seconds', defaults['default_tr_seconds'])

    age_label_name = _pick('age_label_name', defaults['default_age_label_name'])
    sex_label_name = _pick('sex_label_name', defaults['default_sex_label_name'])
    motion_label_name = _pick('motion_label_name', defaults['default_motion_label_name'])
    field_strength_t = _pick('field_strength_t', defaults['default_field_strength_t'])
    p_splice = _pick('p_splice', defaults['default_p_splice'])
    splice_tr_seconds = _pick('splice_tr_seconds', defaults['default_splice_tr_seconds'])
    splice_recovery_gamma_shape = _pick('splice_recovery_gamma_shape', defaults['default_splice_recovery_gamma_shape'])
    splice_onset_only = _pick('splice_onset_only', defaults['default_splice_onset_only'])
    splice_onset_event_paths = _pick('splice_onset_event_paths', defaults['default_splice_onset_event_paths'])
    splice_onset_threshold = _pick('splice_onset_threshold', defaults['default_splice_onset_threshold'])
    subject_ids_path_local = _pick('subject_ids_path', subject_ids_path)
    if pretrain_dynamics_cfg is not None and condition_mode != 'cont':
        raise ValueError("pretrain_dynamics requires condition_mode='cont'")
    if pretrain_dynamics_cfg is not None and relevance_precomputed_path is not None:
        raise ValueError(
            "pretrain_dynamics does not support dynamics.relevance.use_precomputed_relevance"
        )
    if (not unconditioned_pretraining) and relevance_h5_group is not None and condition_cont_language_events is not None:
        condition_cont_language_events = dict(condition_cont_language_events)
        condition_cont_language_events['relevance_h5_group'] = relevance_h5_group
        if relevance_ood_h5_group is not None:
            condition_cont_language_events['relevance_ood_h5_group'] = relevance_ood_h5_group

    ds = SequenceDataset(
        data_path=data_path,
        sequence_length=sequence_length,
        crop_length=defaults['crop_length'],
        condition_label_name=condition_label_name,
        num_conditions=num_conditions,
        atlas_names=atlas_names,
        subject_ids_path=subject_ids_path_local,
        input_stride=input_stride_local,
        unlabeled_condition=unlabeled_condition,
        condition_mode=condition_mode,
        condition_cont_label_name=condition_cont_label_name,
        condition_cont_idx_label_name=condition_cont_idx_label_name,
        condition_cont_scale_label_name=condition_cont_scale_label_name,
        condition_cont_embeddings_name=condition_cont_embeddings_name,
        condition_cont_language_events=condition_cont_language_events,
        pretrain_dynamics=pretrain_dynamics_cfg,
        unconditioned_pretraining=unconditioned_pretraining,
        condition_relevance_path=None if pretrain_dynamics_cfg is not None else relevance_precomputed_path,
        condition_cont_dim=condition_cont_dim,
        event_mapping=event_mapping,
        tr_seconds=tr_seconds,
        sample_mode=sample_mode,
        ood_tag_label_name=ood_tag_label_name,
        allowed_ood_tags=allowed_ood_tags,
        anchor_tag_label_name=anchor_tag_label_name,
        anchor_tags=anchor_tags,
        anchor_crop_index=anchor_crop_index,
        subject_token_enabled=subject_token_enabled,
        subject_context_length=subject_context_length,
        subject_min_gap=subject_min_gap,
        age_label_name=age_label_name,
        sex_label_name=sex_label_name,
        motion_label_name=motion_label_name,
        field_strength_t=field_strength_t,
        p_splice=p_splice,
        splice_context_frames=splice_context_frames,
        splice_generation_frames=splice_generation_frames,
        splice_tr_seconds=splice_tr_seconds,
        splice_recovery_gamma_shape=splice_recovery_gamma_shape,
        splice_onset_only=splice_onset_only,
        splice_onset_event_paths=splice_onset_event_paths,
        splice_onset_threshold=splice_onset_threshold,
        force_zero_relevance_scores=unconditioned_pretraining and relevance_mode != 'none',
    )
    if len(ds) == 0:
        return None
    return ds


def build_dataset_and_weights(
    *,
    split_name,
    data_cfg,
    dynamics_cfg,
    tokenizer_cfg,
    subject_ids_path,
    data_paths,
    default_input_stride,
    default_anchor_crop_index,
    sequence_length,
    num_conditions,
    subject_token_enabled,
    subject_context_length,
    subject_min_gap,
    splice_context_frames,
    splice_generation_frames,
    relevance_mode,
    relevance_precomputed_path,
    relevance_h5_group,
    relevance_ood_h5_group,
    unconditioned_pretraining,
):
    defaults = _build_split_defaults(
        split_name,
        data_cfg,
        dynamics_cfg,
        tokenizer_cfg,
        default_input_stride,
        default_anchor_crop_index,
    )

    reserved_paths = set()
    if split_name == 'train':
        val_data_paths = data_cfg.get('val_path', data_paths)
        val_specs = expand_dataset_specs(data_cfg, 'val', val_data_paths)
        reserved_paths = {_resolved_path(spec['path']) for spec in val_specs}

    dataset_specs = expand_dataset_specs(
        data_cfg,
        split_name,
        data_paths,
        reserved_paths=reserved_paths,
    )

    datasets = []
    dataset_source_ids = []
    dataset_source_weights = []
    for spec in dataset_specs:
        ds = _build_dataset(
            spec,
            defaults=defaults,
            subject_ids_path=subject_ids_path,
            sequence_length=sequence_length,
            num_conditions=num_conditions,
            subject_token_enabled=subject_token_enabled,
            subject_context_length=subject_context_length,
            subject_min_gap=subject_min_gap,
            splice_context_frames=splice_context_frames,
            splice_generation_frames=splice_generation_frames,
            relevance_mode=relevance_mode,
            relevance_precomputed_path=relevance_precomputed_path,
            relevance_h5_group=relevance_h5_group,
            relevance_ood_h5_group=relevance_ood_h5_group,
            unconditioned_pretraining=unconditioned_pretraining,
        )
        if ds is None:
            continue
        datasets.append(ds)
        dataset_source_ids.append(spec['_source_id'])
        dataset_source_weights.append(float(spec['_source_weight']))

    if len(datasets) == 0:
        raise ValueError(f"No usable datasets for split='{split_name}'")

    sample_weights = None
    if split_name == 'train' and any(float(w) != 1.0 for w in dataset_source_weights):
        source_lengths = {}
        for source_id, ds in zip(dataset_source_ids, datasets):
            source_lengths[source_id] = source_lengths.get(source_id, 0) + len(ds)
        parts = []
        for source_id, source_weight, ds in zip(dataset_source_ids, dataset_source_weights, datasets):
            item_weight = float(source_weight) / float(source_lengths[source_id])
            parts.append(np.full(len(ds), item_weight, dtype=np.float64))
        sample_weights = np.concatenate(parts, axis=0)

    if split_name == 'train':
        splice_scope = defaults['default_splice_scope']
        if splice_scope == 'subject':
            return SubjectSpliceConcatDataset(
                datasets,
                defaults['default_splice_subject_regex'],
            ), sample_weights
        if splice_scope != 'run':
            raise ValueError(f"Unknown splice_scope: {splice_scope}")

    if len(datasets) == 1:
        return datasets[0], sample_weights

    return ConcatDataset(datasets), sample_weights


def build_expanded_val_datasets(
    *,
    data_cfg,
    dynamics_cfg,
    tokenizer_cfg,
    subject_ids_path,
    data_paths,
    default_input_stride,
    default_anchor_crop_index,
    sequence_length,
    num_conditions,
    subject_token_enabled,
    subject_context_length,
    subject_min_gap,
    splice_context_frames,
    splice_generation_frames,
    relevance_mode,
    relevance_precomputed_path,
    relevance_h5_group,
    relevance_ood_h5_group,
    unconditioned_pretraining,
):
    defaults = _build_split_defaults(
        'val',
        data_cfg,
        dynamics_cfg,
        tokenizer_cfg,
        default_input_stride,
        default_anchor_crop_index,
    )
    dataset_specs = expand_dataset_specs(data_cfg, 'val', data_paths)
    out = []
    for spec in dataset_specs:
        ds = _build_dataset(
            spec,
            defaults=defaults,
            subject_ids_path=subject_ids_path,
            sequence_length=sequence_length,
            num_conditions=num_conditions,
            subject_token_enabled=subject_token_enabled,
            subject_context_length=subject_context_length,
            subject_min_gap=subject_min_gap,
            splice_context_frames=splice_context_frames,
            splice_generation_frames=splice_generation_frames,
            relevance_mode=relevance_mode,
            relevance_precomputed_path=relevance_precomputed_path,
            relevance_h5_group=relevance_h5_group,
            relevance_ood_h5_group=relevance_ood_h5_group,
            unconditioned_pretraining=unconditioned_pretraining,
        )
        if ds is None:
            continue
        out.append((spec, ds))
    return out


def build_named_val_datasets(
    *,
    data_cfg,
    dynamics_cfg,
    tokenizer_cfg,
    subject_ids_path,
    data_paths,
    default_input_stride,
    default_anchor_crop_index,
    sequence_length,
    num_conditions,
    subject_token_enabled,
    subject_context_length,
    subject_min_gap,
    splice_context_frames,
    splice_generation_frames,
    relevance_mode,
    relevance_precomputed_path,
    relevance_h5_group,
    relevance_ood_h5_group,
    unconditioned_pretraining,
):
    specs = expand_dataset_specs(data_cfg, 'val', data_paths)
    has_name = ['name' in spec for spec in specs]
    if not any(has_name):
        return {}
    if not all(has_name):
        raise ValueError("Every data.val_datasets entry must define 'name' when any validation dataset is named")

    defaults = _build_split_defaults(
        'val',
        data_cfg,
        dynamics_cfg,
        tokenizer_cfg,
        default_input_stride,
        default_anchor_crop_index,
    )

    grouped = {}
    for spec in specs:
        ds = _build_dataset(
            spec,
            defaults=defaults,
            subject_ids_path=subject_ids_path,
            sequence_length=sequence_length,
            num_conditions=num_conditions,
            subject_token_enabled=subject_token_enabled,
            subject_context_length=subject_context_length,
            subject_min_gap=subject_min_gap,
            splice_context_frames=splice_context_frames,
            splice_generation_frames=splice_generation_frames,
            relevance_mode=relevance_mode,
            relevance_precomputed_path=relevance_precomputed_path,
            relevance_h5_group=relevance_h5_group,
            relevance_ood_h5_group=relevance_ood_h5_group,
            unconditioned_pretraining=unconditioned_pretraining,
        )
        if ds is None:
            continue
        grouped.setdefault(spec['name'], []).append(ds)

    named = {}
    for name, datasets in grouped.items():
        if len(datasets) == 1:
            named[name] = datasets[0]
            continue
        named[name] = ConcatDataset(datasets)
    return named
