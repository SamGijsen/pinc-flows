import time
from collections import defaultdict

import torch
from torch.amp import autocast

from util.dynamics_roi_utils import encode_roi_signals, is_raw_roi_encoder_config


def _is_main_process(rank):
    return rank == 0


def _capture_rng_state(device):
    state = {'cpu': torch.random.get_rng_state()}
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng_state(state, device):
    torch.random.set_rng_state(state['cpu'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)


def _compute_loss_log_data(dynamics, prepared, device, drop_mask_override=None, drop_mask_global_override=None):
    z_clean = prepared['z_clean']
    B, T = z_clean.shape[:2]
    # Legacy RNG burn
    sigma = torch.rand(B, T, device=device)
    _, log_data = dynamics.compute_loss(
        z_clean,
        sigma,
        condition_disc=prepared['condition_disc'],
        condition_disc_weight=prepared['condition_disc_weight'],
        condition_cont=prepared['condition_cont'],
        condition_mode=prepared['condition_mode'],
        relevance_scores=prepared['relevance_scores'],
        relevance_embedding_type=prepared['relevance_embedding_type'],
        generation_loss_weight=prepared['generation_loss_weight'],
        subject_latents=prepared['subject_latents'],
        age=prepared['age'],
        sex=prepared['sex'],
        motion=prepared['motion'],
        field_strength=prepared['field_strength'],
        drop_mask_override=drop_mask_override,
        drop_mask_global_override=drop_mask_global_override,
    )
    return log_data


@torch.inference_mode()
def encode_sequence(tokenizer, signal, device, use_amp=False, num_chunks=4):
    """
    Encode a sequence of crops through a tokenizer-like encoder.

    Args:
        tokenizer: encoder module such as PassthroughEncoder
        signal: [B, T, num_rois, crop_length] sequence of crops
        device: torch device
    Returns:
        z: [B, T, num_latents, latent_dim] encoded latents
    """
    B, T, num_rois, crop_len = signal.shape

    # Flatten batch and time for encoding
    signal_flat = signal.view(B * T, num_rois, crop_len).to(device, non_blocking=True)

    # Frozen tokenizer path: skip autograd bookkeeping during latent encoding.
    with autocast(device_type=device.type, enabled=(use_amp and device.type == 'cuda')):
        n = signal_flat.shape[0]
        num_chunks = max(int(num_chunks), 1)
        chunk_size = (n + num_chunks - 1) // num_chunks
        if chunk_size >= n:
            z_flat = tokenizer.encoder(signal_flat)  # [B*T, K, D]
        else:
            chunks = []
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunks.append(tokenizer.encoder(signal_flat[start:end]))
            z_flat = torch.cat(chunks, dim=0)  # [B*T, K, D]

    # Reshape back
    K, D = z_flat.shape[1], z_flat.shape[2]
    z = z_flat.view(B, T, K, D)

    return z


def encode_batch_signal(
    batch,
    signal_key,
    tokenizer,
    device,
    *,
    one_roi_one_token=False,
    roi_encoder=None,
    tokenizer_use_amp=False,
):
    if signal_key not in batch:
        raise KeyError(f"Batch is missing {signal_key!r}")
    signal = batch[signal_key]
    if one_roi_one_token:
        z = signal.to(device, non_blocking=True)
        if z.shape[-1] != 1:
            raise ValueError(
                f"one_roi_one_token expects {signal_key} shape [B,T,R,1], got {tuple(z.shape)}"
            )
        return z
    if is_raw_roi_encoder_config(roi_encoder):
        return encode_roi_signals(signal.to(device, non_blocking=True), roi_encoder)
    return encode_sequence(tokenizer, signal, device, use_amp=tokenizer_use_amp)


def move_batch_tensors(batch, keys, device, *, non_blocking=True):
    out = {}
    for key in keys:
        value = batch.get(key)
        out[key] = value.to(device, non_blocking=non_blocking) if value is not None else None
    return out


def prepare_batch(
    batch,
    tokenizer,
    device,
    subject_token_enabled,
    one_roi_one_token=False,
    roi_encoder=None,
    tokenizer_use_amp=False,
    measure_tokenizer_time=False,
):
    condition_disc = batch.get('condition_disc', batch.get('condition'))
    if condition_disc is None:
        raise KeyError("Batch is missing 'condition_disc' (or legacy 'condition')")

    if measure_tokenizer_time and device.type == 'cuda':
        torch.cuda.synchronize(device)
    t0 = time.time() if measure_tokenizer_time else None

    z_clean = encode_batch_signal(
        batch,
        'signal',
        tokenizer,
        device,
        one_roi_one_token=one_roi_one_token,
        roi_encoder=roi_encoder,
        tokenizer_use_amp=tokenizer_use_amp,
    )
    subject_latents = None
    if subject_token_enabled:
        subject_latents = encode_batch_signal(
            batch,
            'distant_signal',
            tokenizer,
            device,
            one_roi_one_token=one_roi_one_token,
            roi_encoder=roi_encoder,
            tokenizer_use_amp=tokenizer_use_amp,
        )

    tokenizer_encode_time_s = None
    if measure_tokenizer_time and (not one_roi_one_token) and (not is_raw_roi_encoder_config(roi_encoder)):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        tokenizer_encode_time_s = time.time() - t0

    out = {
        'z_clean': z_clean,
        'subject_latents': subject_latents,
        'condition_disc': condition_disc.to(device, non_blocking=True),
        'tokenizer_encode_time_s': tokenizer_encode_time_s,
    }
    out.update(move_batch_tensors(batch, (
        'condition_disc_weight',
        'condition_cont',
        'condition_mode',
        'relevance_scores',
        'relevance_embedding_type',
        'generation_loss_weight',
        'age',
        'sex',
        'motion',
        'field_strength',
    ), device))
    return out


def train_epoch(dynamics, tokenizer, train_loader, optimizer, scaler, device, epoch, rank, config):
    """Run one training epoch."""
    dynamics.train()

    losses = defaultdict(list)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)

    model_ref = dynamics.module if hasattr(dynamics, 'module') else dynamics
    subject_token_enabled = bool(getattr(model_ref, 'subject_token_enabled', False))
    one_roi_one_token = bool(getattr(model_ref, 'one_roi_one_token', False))
    roi_encoder = getattr(model_ref, 'roi_encoder', None)
    tokenizer_use_amp = bool(config.get('tokenizer_encode_amp', config.get('use_amp', True))) and (device.type == 'cuda')

    log_every = config.get('log_every', 50)

    for batch_idx, batch in enumerate(train_loader):
        do_log_step = _is_main_process(rank) and (batch_idx % log_every == 0)
        measure_tok_time = False
        prepared = prepare_batch(
            batch,
            tokenizer,
            device,
            subject_token_enabled,
            one_roi_one_token=one_roi_one_token,
            roi_encoder=roi_encoder,
            tokenizer_use_amp=tokenizer_use_amp,
            measure_tokenizer_time=measure_tok_time,
        )
        z_clean = prepared['z_clean']
        subject_latents = prepared['subject_latents']
        condition_disc = prepared['condition_disc']
        condition_disc_weight = prepared['condition_disc_weight']
        condition_cont = prepared['condition_cont']
        condition_mode = prepared['condition_mode']
        relevance_scores = prepared['relevance_scores']
        relevance_embedding_type = prepared['relevance_embedding_type']
        generation_loss_weight = prepared['generation_loss_weight']
        age = prepared['age']
        sex = prepared['sex']
        motion = prepared['motion']
        field_strength = prepared['field_strength']

        B, T = z_clean.shape[:2]

        # Legacy RNG burn
        sigma = torch.rand(B, T, device=device)

        # Legacy RNG burn
        _ = torch.rand(1).item()

        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda', enabled=(scaler is not None)):
            loss, log_data = dynamics.compute_loss(
                z_clean,
                sigma,
                condition_disc=condition_disc,
                condition_disc_weight=condition_disc_weight,
                condition_cont=condition_cont,
                condition_mode=condition_mode,
                relevance_scores=relevance_scores,
                relevance_embedding_type=relevance_embedding_type,
                generation_loss_weight=generation_loss_weight,
                subject_latents=subject_latents,
                age=age,
                sex=sex,
                motion=motion,
                field_strength=field_strength,
            )

        scaled_loss = loss / gradient_accumulation_steps
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(dynamics.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(dynamics.parameters(), max_norm=1.0)
                optimizer.step()

        for k, v in log_data.items():
            losses[k].append(v)

        if do_log_step:
            loss_str = ' | '.join([f"{k}: {v:.4f}" for k, v in log_data.items()])
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] {loss_str}")

    avg_losses = {k: sum(v) / len(v) for k, v in losses.items()}
    return avg_losses


@torch.no_grad()
def validate(
    dynamics,
    tokenizer,
    val_loader,
    device,
    rank,
    epoch=None,
    log_every=50,
    run_unconditioned=False,
    num_passes=1,
):
    """Run validation."""
    dynamics.eval()

    num_passes = int(num_passes)
    if num_passes <= 0:
        raise ValueError(f"num_passes must be > 0, got {num_passes}")

    losses = defaultdict(list)
    model_ref = dynamics.module if hasattr(dynamics, 'module') else dynamics
    subject_token_enabled = bool(getattr(model_ref, 'subject_token_enabled', False))
    one_roi_one_token = bool(getattr(model_ref, 'one_roi_one_token', False))
    roi_encoder = getattr(model_ref, 'roi_encoder', None)
    tokenizer_use_amp = (device.type == 'cuda')

    num_batches = len(val_loader)
    for pass_idx in range(num_passes):
        for batch_idx, batch in enumerate(val_loader):
            global_batch_idx = pass_idx * num_batches + batch_idx
            do_log_step = _is_main_process(rank) and (global_batch_idx % log_every == 0)
            prepared = prepare_batch(
                batch,
                tokenizer,
                device,
                subject_token_enabled,
                one_roi_one_token=one_roi_one_token,
                roi_encoder=roi_encoder,
                tokenizer_use_amp=tokenizer_use_amp,
                measure_tokenizer_time=False,
            )
            rng_state = _capture_rng_state(device)
            log_data = _compute_loss_log_data(dynamics, prepared, device)

            for k, v in log_data.items():
                losses[k].append(v)

            if do_log_step:
                loss_str = ' | '.join([f"{k}: {v:.4f}" for k, v in log_data.items()])
                prefix = f"Epoch {epoch} - Val" if epoch is not None else "Val"
                print(
                    f"{prefix} pass {pass_idx + 1}/{num_passes} "
                    f"[{batch_idx}/{num_batches}] {loss_str}"
                )

            if run_unconditioned:
                B = prepared['z_clean'].shape[0]
                drop_mask = torch.ones(B, dtype=torch.bool, device=device)
                _restore_rng_state(rng_state, device)
                log_data_unconditioned = _compute_loss_log_data(
                    dynamics,
                    prepared,
                    device,
                    drop_mask_override=drop_mask,
                    drop_mask_global_override=drop_mask,
                )
                for k, v in log_data_unconditioned.items():
                    losses[f'{k}_unconditioned'].append(v)

    avg_losses = {k: sum(v) / len(v) for k, v in losses.items()}
    return avg_losses


def save_checkpoint(dynamics, optimizer, epoch, config, save_path, extra_state=None):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': dynamics.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }
    if extra_state is not None:
        checkpoint.update(extra_state)
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")
