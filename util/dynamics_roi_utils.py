import torch

from util.yeo17 import load_schaefer_yeo7_spec_for_atlas


def normalize_roi_encoder_config(dynamics_cfg, tokenizer_cfg, data_cfg):
    roi_encoder_raw = dynamics_cfg.get('roi_encoder')
    if roi_encoder_raw is None:
        return None

    assert isinstance(roi_encoder_raw, dict)
    mode = str(roi_encoder_raw['mode']).lower()
    assert mode in ('passthrough', 'linear', 'hemispheric', 'yeo7')

    if mode == 'passthrough':
        assert roi_encoder_raw.get('emb_dim') is None
        return {'mode': 'passthrough'}

    crop_length = int(tokenizer_cfg.get('input_timesteps', data_cfg.get('patch_size', 1)))
    assert crop_length == 1

    emb_dim = int(roi_encoder_raw['emb_dim'])
    num_rois = int(data_cfg['num_rois'])
    if mode == 'linear':
        return {
            'mode': 'linear',
            'input_dim': num_rois,
            'num_tokens': 1,
            'emb_dim': emb_dim,
        }

    if mode == 'hemispheric':
        assert num_rois % 2 == 0
        return {
            'mode': 'hemispheric',
            'input_dim': num_rois // 2,
            'num_tokens': 2,
            'emb_dim': emb_dim,
        }

    atlas_names = data_cfg['atlas_names']
    assert isinstance(atlas_names, list) and len(atlas_names) == 1
    spec = load_schaefer_yeo7_spec_for_atlas(atlas_names[0])
    parcel_to_network_idx = spec['parcel_to_network_idx']
    assert len(parcel_to_network_idx) == num_rois
    group_indices = [
        torch.nonzero(torch.as_tensor(parcel_to_network_idx) == network_idx).flatten().tolist()
        for network_idx in range(spec['n_networks'])
    ]
    input_dim = max(len(indices) for indices in group_indices)
    return {
        'mode': 'yeo7',
        'input_dim': input_dim,
        'num_tokens': 7,
        'emb_dim': emb_dim,
        'num_rois': num_rois,
        'group_indices': group_indices,
    }


def is_raw_roi_encoder_config(roi_encoder):
    return roi_encoder is not None and roi_encoder['mode'] in ('linear', 'hemispheric', 'yeo7')


def _encode_grouped(signal, roi_encoder):
    out = signal.new_zeros((*signal.shape[:-1], roi_encoder['num_tokens'], roi_encoder['input_dim']))
    for token_idx, indices in enumerate(roi_encoder['group_indices']):
        idx = torch.as_tensor(indices, device=signal.device)
        out[..., token_idx, :len(indices)] = signal.index_select(-1, idx)
    return out


def encode_roi_signals(signal, roi_encoder):
    assert is_raw_roi_encoder_config(roi_encoder)
    assert signal.shape[-1] == 1
    signal = signal.squeeze(-1)
    if roi_encoder['mode'] == 'linear':
        return signal.unsqueeze(-2)
    if roi_encoder['mode'] == 'yeo7':
        return _encode_grouped(signal, roi_encoder)

    mid = int(roi_encoder['input_dim'])
    return torch.stack([signal[..., :mid], signal[..., mid:]], dim=-2)


def _decode_grouped(z, roi_encoder):
    signal = z.new_zeros((*z.shape[:-2], roi_encoder['num_rois']))
    for token_idx, indices in enumerate(roi_encoder['group_indices']):
        idx = torch.as_tensor(indices, device=z.device)
        signal.index_copy_(-1, idx, z[..., token_idx, :len(indices)])
    return signal


def decode_roi_tokens(z, roi_encoder, keep_crop_dim=False):
    assert is_raw_roi_encoder_config(roi_encoder)
    if roi_encoder['mode'] == 'linear':
        signal = z.squeeze(-2)
    elif roi_encoder['mode'] == 'yeo7':
        assert z.shape[-2] == 7
        signal = _decode_grouped(z, roi_encoder)
    else:
        assert z.shape[-2] == 2
        signal = torch.cat([z[..., 0, :], z[..., 1, :]], dim=-1)
    if keep_crop_dim:
        return signal.unsqueeze(-1)
    return signal
