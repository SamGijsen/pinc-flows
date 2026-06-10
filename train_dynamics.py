import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from util.dataset_sampling import (
    WeightedDistributedSampler,
    build_dataset_and_weights,
    build_named_val_datasets,
)
from util.dynamics_runtime import build_dynamics_runtime, load_checkpoint_weights, load_config
from util.dynamics_training_utils import save_checkpoint, train_epoch, validate


def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def _backfill_metric_series(training_log, key, fill_value=None):
    target_len = max(
        len(training_log.get('epoch', [])),
        len(training_log.get('train_loss', [])),
        len(training_log.get('val_loss', [])),
        len(training_log.get('train_metrics', [])),
        len(training_log.get('val_metrics', [])),
    )
    series = training_log.setdefault(key, [])
    missing = target_len - len(series)
    if missing > 0:
        series.extend([fill_value] * missing)


def main(config):
    # Setup distributed
    is_distributed, rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if is_main_process(rank):
        print(f"Training dynamics model")
        print(f"Distributed: {is_distributed}, World size: {world_size}")
        print(f"Device: {device}")

    # Shared dynamics runtime setup
    tokenizer_cfg = config['tokenizer']
    dynamics_cfg = config['dynamics']
    data_cfg = config['data']

    runtime = build_dynamics_runtime(
        config,
        device,
        eval_mode=False,
        log_fn=print if is_main_process(rank) else None,
    )
    tokenizer = runtime.tokenizer
    dynamics = runtime.model
    pretrain_dynamics_enabled = runtime.pretrain_dynamics_enabled
    unconditioned_pretraining = runtime.unconditioned_pretraining
    subject_token_enabled = runtime.subject_token_enabled
    subject_context_length = runtime.subject_context_length
    subject_min_gap = runtime.subject_min_gap
    relevance_mode = runtime.relevance.mode
    relevance_precomputed_path = runtime.relevance.precomputed_path
    relevance_h5_group = runtime.relevance.h5_group
    num_conditions = runtime.num_conditions

    if is_main_process(rank):
        num_params = sum(p.numel() for p in dynamics.parameters())
        print(f"Dynamics model: {num_params:,} parameters")
        print(f"Global condition tokens enabled: {runtime.global_condition_enabled}")

    # Wrap for DDP
    if is_distributed:
        dynamics = DDP(dynamics, device_ids=[local_rank])

    context_frames = dynamics_cfg.get('context_frames', 8)
    generation_frames = dynamics_cfg.get('generation_frames', 16)
    sequence_length = context_frames + generation_frames
    default_anchor_crop_index = context_frames
    if is_main_process(rank):
        print(f"Parallel mode: {context_frames} (context) + {generation_frames} (generation) = {sequence_length}")
    splice_context_frames = context_frames
    splice_generation_frames = generation_frames

    default_input_stride = data_cfg.get('input_stride', 1)

    if is_main_process(rank):
        print("Creating train dataset...")
    train_data_paths = data_cfg.get('train_path')
    train_dataset, train_sample_weights = build_dataset_and_weights(
        split_name='train',
        data_cfg=data_cfg,
        dynamics_cfg=dynamics_cfg,
        tokenizer_cfg=tokenizer_cfg,
        subject_ids_path=data_cfg.get('train_subject_ids_path'),
        data_paths=train_data_paths,
        default_input_stride=default_input_stride,
        default_anchor_crop_index=default_anchor_crop_index,
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
        relevance_ood_h5_group=None,
        unconditioned_pretraining=unconditioned_pretraining,
    )

    if is_main_process(rank):
        print("Creating val dataset...")
    val_data_paths = data_cfg.get('val_path', train_data_paths)
    val_dataset, _ = build_dataset_and_weights(
        split_name='val',
        data_cfg=data_cfg,
        dynamics_cfg=dynamics_cfg,
        tokenizer_cfg=tokenizer_cfg,
        subject_ids_path=data_cfg.get('val_subject_ids_path'),
        data_paths=val_data_paths,
        default_input_stride=default_input_stride,
        default_anchor_crop_index=default_anchor_crop_index,
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
        relevance_ood_h5_group=None,
        unconditioned_pretraining=unconditioned_pretraining,
    )

    # Create data loaders
    batch_size = int(config.get('batch_size', 32))
    steps_per_epoch = int(config.get('steps_per_epoch', 100))
    train_samples_per_epoch = steps_per_epoch * batch_size
    if train_sample_weights is None:
        train_sample_weights = np.ones(len(train_dataset), dtype=np.float64)
        if is_main_process(rank):
            print(
                f"Using uniform train sampling across {len(train_sample_weights)} samples "
                f"for {steps_per_epoch} steps per epoch"
            )
    else:
        if is_main_process(rank):
            print(
                f"Using weighted train sampling across {len(train_sample_weights)} samples "
                f"for {steps_per_epoch} steps per epoch"
            )
    if is_distributed:
        train_sampler = WeightedDistributedSampler(
            train_sample_weights,
            num_replicas=world_size,
            rank=rank,
            num_samples=train_samples_per_epoch,
        )
    else:
        train_sampler = WeightedRandomSampler(
            torch.as_tensor(train_sample_weights, dtype=torch.double),
            num_samples=train_samples_per_epoch,
            replacement=True,
        )
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        drop_last=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )
    named_val_loaders = {}
    if is_main_process(rank):
        named_val_datasets = build_named_val_datasets(
            data_cfg=data_cfg,
            dynamics_cfg=dynamics_cfg,
            tokenizer_cfg=tokenizer_cfg,
            subject_ids_path=data_cfg.get('val_subject_ids_path'),
            data_paths=val_data_paths,
            default_input_stride=default_input_stride,
            default_anchor_crop_index=default_anchor_crop_index,
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
            relevance_ood_h5_group=None,
            unconditioned_pretraining=unconditioned_pretraining,
        )
        for name, ds in named_val_datasets.items():
            named_val_loaders[name] = DataLoader(
                ds,
                batch_size=config.get('batch_size', 32),
                shuffle=False,
                drop_last=False,
                num_workers=config.get('num_workers', 4),
                pin_memory=True,
            )

    # Optimizer LR
    opt_cfg = config.get('optimizer', {})
    batch_size = config.get('batch_size', 32)

    base_lr = opt_cfg.get('lr', 1e-4)
    scaled_lr = base_lr

    if is_main_process(rank):
        print(f"Parallel mode LR: using config value directly ({scaled_lr:.2e})")

    optimizer = optim.AdamW(
        dynamics.parameters(),
        lr=scaled_lr,
        weight_decay=opt_cfg.get('weight_decay', 0.01),
        betas=opt_cfg.get('betas', (0.9, 0.999)),
    )

    # Learning rate schedule: linear warmup + (cosine decay OR constant OR cosine-then-constant)
    warmup_epochs = opt_cfg.get('warmup_epochs', 3)
    lr_schedule = opt_cfg.get('lr_schedule', 'cosine')  # 'cosine', 'constant', or 'cosine_constant'
    min_lr_ratio = opt_cfg.get('min_lr_ratio', 0.1)  # For cosine: decay to this fraction of max
    cosine_epochs = opt_cfg.get('cosine_epochs', None)  # For cosine_constant: when to stop decay
    eta_min = scaled_lr * min_lr_ratio

    from torch.optim.lr_scheduler import LambdaLR

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1.0 / max(warmup_epochs, 1),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    if lr_schedule == 'constant':
        constant_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, constant_scheduler],
            milestones=[warmup_epochs],
        )
        schedule_desc = f"{warmup_epochs} epoch warmup, then constant at {scaled_lr:.2e}"

    elif lr_schedule == 'cosine_constant':
        # Cosine decay until cosine_epochs, then constant at eta_min
        if cosine_epochs is None:
            cosine_epochs = config.get('num_epochs', 100)  # Default: full cosine
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs - warmup_epochs,
            eta_min=eta_min,
        )
        # After cosine ends, stay at eta_min (multiply by min_lr_ratio since optimizer has scaled_lr)
        constant_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min_lr_ratio)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler, constant_scheduler],
            milestones=[warmup_epochs, cosine_epochs],
        )
        schedule_desc = f"{warmup_epochs} epoch warmup, cosine to {eta_min:.2e} at epoch {cosine_epochs}, then constant"

    else:
        # Cosine decay (default) - decays over full training
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.get('num_epochs', 100) - warmup_epochs,
            eta_min=eta_min,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        schedule_desc = f"{warmup_epochs} epoch warmup, cosine decay to {eta_min:.2e}"

    if is_main_process(rank):
        print(f"LR schedule: {schedule_desc}")

    # Mixed precision
    scaler = GradScaler() if config.get('use_amp', True) else None

    num_epochs = config.get('num_epochs', 100)
    start_epoch = 0

    # Checkpoint loading modes:
    # - resume_checkpoint: full training resume (model + optimizer + epoch/scheduler)
    # - weights_only_checkpoint: model weights only for eval/decode/probe (start_epoch stays 0)
    resume_path = config.get('resume_checkpoint')
    weights_only_path = config.get('weights_only_checkpoint')
    if resume_path and weights_only_path:
        raise ValueError("Specify only one of resume_checkpoint or weights_only_checkpoint")

    model_to_load = dynamics.module if hasattr(dynamics, 'module') else dynamics

    if weights_only_path:
        if is_main_process(rank):
            print(f"Loading weights-only checkpoint: {weights_only_path}")
        checkpoint = load_checkpoint_weights(model_to_load, weights_only_path, device)
        if is_main_process(rank):
            ckpt_epoch = checkpoint.get('epoch', None) if isinstance(checkpoint, dict) else None
            if ckpt_epoch is not None:
                print(f"Loaded model weights from checkpoint epoch {ckpt_epoch}; using start_epoch=0")
            else:
                print("Loaded model weights; using start_epoch=0")
    elif resume_path:
        if is_main_process(rank):
            print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = load_checkpoint_weights(model_to_load, resume_path, device)
        if 'optimizer_state_dict' not in checkpoint:
            raise KeyError("resume_checkpoint is missing optimizer_state_dict")
        if 'epoch' not in checkpoint:
            raise KeyError("resume_checkpoint is missing epoch")
        # Load optimizer state for true resume.
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Set start epoch
        start_epoch = checkpoint['epoch'] + 1
        # Fast-forward scheduler to correct epoch
        for _ in range(start_epoch):
            scheduler.step()
        if is_main_process(rank):
            print(f"Resumed from epoch {checkpoint['epoch']}, starting at epoch {start_epoch}")
    output_dir = Path(config.get('output_dir', 'checkpoints'))
    run_name = config.get('run_name', 'dynamics_default')
    save_dir = output_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = config.get('evaluation', {})
    eval_cfg['input_stride'] = default_input_stride
    hcp_eval_cfg = eval_cfg.get('hcp', {})
    hcp_eval_enabled = bool(hcp_eval_cfg.get('enabled', False))

    if is_main_process(rank):
        with open(save_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Saved config to: {save_dir / 'config.yaml'}")

    training_log_path = save_dir / 'training_log.yaml'
    if resume_path and training_log_path.exists():
        with open(training_log_path) as f:
            training_log = yaml.safe_load(f)
        _backfill_metric_series(training_log, 'val_loss_unconditioned')
        _backfill_metric_series(training_log, 'val_loss_by_name')
        _backfill_metric_series(training_log, 'val_loss_unconditioned_by_name')
        _backfill_metric_series(training_log, 'val_metrics_by_name')
    else:
        training_log = {
            'train_loss': [],
            'val_loss': [],
            'val_loss_unconditioned': [],
            'val_loss_by_name': [],
            'val_loss_unconditioned_by_name': [],
            'epoch': [],
            'train_metrics': [],
            'val_metrics': [],
            'val_metrics_by_name': [],
        }

    for epoch in range(start_epoch, num_epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        train_losses = train_epoch(
            dynamics, tokenizer, train_loader, optimizer, scaler, device, epoch, rank, config
        )
        scheduler.step()
        epoch_time_s = time.time() - epoch_start_time

        # Log training
        if is_main_process(rank):
            train_str = ' | '.join([f"{k}: {v:.4f}" for k, v in train_losses.items()])
            print(f"Epoch {epoch + 1} - Train: {train_str} | epoch_time {epoch_time_s:.2f}s")

            training_log['epoch'].append(epoch + 1)
            training_log['train_loss'].append(train_losses.get('dynamics_loss', 0))
            training_log['train_metrics'].append({k: float(v) for k, v in train_losses.items()})

            # Quick val loss at interval
            val_loss_interval = eval_cfg.get('val_loss_interval', 5)
            if (epoch + 1) % val_loss_interval == 0:
                # random_safe validation samples a fresh window per subject on each pass
                val_num_passes = int(eval_cfg.get('val_loss_max_batches', 1))
                val_losses = validate(
                    dynamics,
                    tokenizer,
                    val_loader,
                    device,
                    rank,
                    epoch=epoch + 1,
                    log_every=config.get('log_every', 50),
                    run_unconditioned=pretrain_dynamics_enabled,
                    num_passes=val_num_passes,
                )
                val_str = ' | '.join([f"{k}: {v:.4f}" for k, v in val_losses.items()])
                print(f"Epoch {epoch + 1} - Val: {val_str}")
                training_log['val_loss'].append(val_losses.get('dynamics_loss', 0))
                training_log['val_loss_unconditioned'].append(val_losses.get('dynamics_loss_unconditioned'))
                training_log['val_metrics'].append({k: float(v) for k, v in val_losses.items()})
                val_losses_by_name = {}
                val_loss_by_name = {}
                val_loss_unconditioned_by_name = {}
                for name, loader in named_val_loaders.items():
                    group_losses = validate(
                        dynamics,
                        tokenizer,
                        loader,
                        device,
                        rank,
                        epoch=epoch + 1,
                        log_every=config.get('log_every', 50),
                        run_unconditioned=pretrain_dynamics_enabled,
                        num_passes=val_num_passes,
                    )
                    group_str = ' | '.join([f"{k}: {v:.4f}" for k, v in group_losses.items()])
                    print(f"Epoch {epoch + 1} - Val[{name}]: {group_str}")
                    val_losses_by_name[name] = {k: float(v) for k, v in group_losses.items()}
                    val_loss_by_name[name] = group_losses.get('dynamics_loss', 0.0)
                    val_loss_unconditioned_by_name[name] = group_losses.get('dynamics_loss_unconditioned')
                training_log['val_loss_by_name'].append(val_loss_by_name)
                training_log['val_loss_unconditioned_by_name'].append(val_loss_unconditioned_by_name)
                training_log['val_metrics_by_name'].append(val_losses_by_name)
            else:
                training_log['val_loss'].append(None)
                training_log['val_loss_unconditioned'].append(None)
                training_log['val_metrics'].append(None)
                training_log['val_loss_by_name'].append(None)
                training_log['val_loss_unconditioned_by_name'].append(None)
                training_log['val_metrics_by_name'].append(None)

            if (epoch + 1) % val_loss_interval == 0:
                with open(save_dir / 'training_log.yaml', 'w') as f:
                    yaml.dump(training_log, f, default_flow_style=False)

        if is_main_process(rank) and (epoch + 1) % config.get('save_every', 10) == 0:
            save_checkpoint(
                dynamics.module if is_distributed else dynamics,
                optimizer,
                epoch,
                config,
                save_dir / f'dynamics_epoch{epoch:04d}.pt',
                extra_state=None,
            )

    if is_distributed:
        dist.barrier()

    if is_main_process(rank):
        with open(save_dir / 'training_log.yaml', 'w') as f:
            yaml.dump(training_log, f, default_flow_style=False)
        print(f"Saved training log to: {save_dir / 'training_log.yaml'}")

        if hcp_eval_enabled:
            final_checkpoint_path = save_dir / 'dynamics_final.pt'
            if not final_checkpoint_path.exists():
                final_epoch = max(max(start_epoch, num_epochs), 1) - 1
                save_checkpoint(
                    dynamics.module if is_distributed else dynamics,
                    optimizer,
                    final_epoch,
                    config,
                    final_checkpoint_path,
                    extra_state=None,
                )
            from eval.hcp import run_hcp_eval

            run_hcp_eval(
                config_path=save_dir / 'config.yaml',
                checkpoint_path=final_checkpoint_path,
                output_dir=save_dir / hcp_eval_cfg.get('out_subdir', 'hcp_eval'),
            )

    cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--steps-per-epoch', type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.steps_per_epoch is not None:
        config['steps_per_epoch'] = int(args.steps_per_epoch)
    main(config)
