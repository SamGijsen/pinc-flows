"""Training/loss routines for FMRIDynamics."""

import torch

class DynamicsLossMixin:
    def _latent_mse(self, pred, target, weight=None):
        mse = (pred - target) ** 2
        if weight is not None:
            mse = weight * mse

        mask = getattr(self, "roi_encoder_valid_mask", None)
        if mask is None:
            return mse.mean()

        mask = mask.to(device=pred.device, dtype=pred.dtype).view(1, 1, *mask.shape)
        if weight is not None:
            mask = weight * mask
        return (mse * mask).sum() / mask.sum()

    def compute_loss(
        self,
        z_clean,
        sigma,
        condition=None,
        condition_disc=None,
        condition_disc_weight=None,
        condition_cont=None,
        condition_mode=None,
        relevance_scores=None,
        relevance_embedding_type=None,
        generation_loss_weight=None,
        subject_latents=None,
        age=None,
        sex=None,
        motion=None,
        field_strength=None,
        drop_mask_override=None,
        drop_mask_global_override=None,
    ):
        """
        Compute parallel flow matching loss.

        Args:
            z_clean: [B, T, num_latents, latent_dim] clean latent sequences (from frozen tokenizer)
            sigma: retained for legacy callers; parallel loss samples generation sigma internally
            condition: [B, T] legacy categorical condition indices (backward-compatible)
            condition_disc: [B, T] discrete condition indices
            condition_disc_weight: [B, T] optional additive weights for discrete condition embeddings
            condition_cont: [B, T, D_cond] continuous condition vectors
            condition_mode: [B, T] 0=discrete, 1=continuous
            relevance_scores: [B, T, R] optional direct per-ROI simtok scores
            relevance_embedding_type: [B] optional simtok type ids
            generation_loss_weight: [B, generation_frames] optional per-frame weighting for parallel generation loss
            drop_mask_override: optional forced condition dropout mask, [B] or [B, T]
            drop_mask_global_override: optional forced global-prefix dropout mask, [B]

        Returns:
            loss: scalar MSE loss
            log_data: dict with metrics
        """
        subject_token = None
        if self.subject_token_enabled:
            if subject_latents is None:
                assert self.subject_allow_missing_token, (
                    "subject_latents missing. Set subject_allow_missing_token=True to allow masked subject token."
                )
            else:
                subject_token = self.build_subject_token(subject_latents)

        return self._compute_parallel_loss(
            z_clean, condition=condition, condition_disc=condition_disc,
            condition_disc_weight=condition_disc_weight,
            condition_cont=condition_cont, condition_mode=condition_mode,
            relevance_scores=relevance_scores,
            relevance_embedding_type=relevance_embedding_type,
            generation_loss_weight=generation_loss_weight,
            subject_token=subject_token,
            age=age,
            sex=sex,
            motion=motion,
            field_strength=field_strength,
            drop_mask_override=drop_mask_override,
            drop_mask_global_override=drop_mask_global_override,
        )

    def _compute_parallel_loss(
        self,
        z_clean,
        condition=None,
        condition_disc=None,
        condition_disc_weight=None,
        condition_cont=None,
        condition_mode=None,
        relevance_scores=None,
        relevance_embedding_type=None,
        generation_loss_weight=None,
        subject_token=None,
        age=None,
        sex=None,
        motion=None,
        field_strength=None,
        drop_mask_override=None,
        drop_mask_global_override=None,
    ):
        """
        Compute loss for parallel denoising (non-AR mode).

        Context frames (first context_frames) are kept clean with sigma=1.
        Generation frames (next generation_frames) get random sigma and are denoised.
        If parallel_shared_sigma=True, all generation frames in a sample share one sigma.
        Loss is computed only on generation frames.

        Args:
            z_clean: [B, T, num_latents, latent_dim] clean latent sequences
                     T = context_frames + generation_frames
            condition: legacy categorical condition indices
            condition_disc: [B, T] discrete condition indices
            condition_disc_weight: [B, T] optional additive weights for discrete condition embeddings
            condition_cont: [B, T, D_cond] continuous condition vectors
            condition_mode: [B, T] 0=discrete, 1=continuous
            generation_loss_weight: [B, generation_frames] optional per-frame weights for generation loss

        Returns:
            loss: scalar MSE loss (on generation frames only)
            log_data: dict with metrics
        """
        B, T = z_clean.shape[:2]
        device = z_clean.device
        ctx_len = self.context_frames
        gen_len = self.generation_frames
        condition_disc_all, condition_cont_all, condition_mode_all = self._normalize_condition_inputs(
            condition, condition_disc, condition_cont, condition_mode, B, T, device
        )
        condition_disc_weight_all = None
        if condition_disc_weight is not None:
            condition_disc_weight_all = condition_disc_weight.to(device=device, dtype=torch.float32)
        relevance_scores_all = None
        if relevance_scores is not None:
            relevance_scores_all = relevance_scores.to(device=device, dtype=torch.float32)
        relevance_embedding_type_all = None
        if relevance_embedding_type is not None:
            relevance_embedding_type_all = relevance_embedding_type.to(device=device, dtype=torch.long)
        generation_loss_weight_all = None
        if generation_loss_weight is not None:
            generation_loss_weight_all = generation_loss_weight.to(device=device, dtype=torch.float32)

        assert T == ctx_len + gen_len, f"Expected T={ctx_len + gen_len}, got {T}"
        if generation_loss_weight_all is not None:
            if generation_loss_weight_all.shape != (B, gen_len):
                raise ValueError(
                    f"generation_loss_weight must have shape {(B, gen_len)}, "
                    f"got {tuple(generation_loss_weight_all.shape)}"
                )
            if not torch.isfinite(generation_loss_weight_all).all():
                raise ValueError("generation_loss_weight contains non-finite values")
            if (generation_loss_weight_all < 0).any():
                raise ValueError("generation_loss_weight must be non-negative")
            frame_weight = generation_loss_weight_all
        else:
            frame_weight = torch.ones((B, gen_len), device=device, dtype=torch.float32)

        # Split into context and generation
        z_context = z_clean[:, :ctx_len]      # [B, ctx_len, K, D]
        z_generation = z_clean[:, ctx_len:]   # [B, gen_len, K, D]

        # Context: always sigma=1 (clean), no noise added
        sigma_context = torch.ones(B, ctx_len, device=device)

        # Generation: random sigma, apply flow interpolation
        if self.parallel_shared_sigma:
            sigma_generation = torch.rand(B, 1, device=device).expand(B, gen_len)
        else:
            sigma_generation = torch.rand(B, gen_len, device=device)
        z_noise_gen = torch.randn_like(z_generation)
        sigma_gen_expanded = sigma_generation[:, :, None, None]
        z_noisy_generation = (1 - sigma_gen_expanded) * z_noise_gen + sigma_gen_expanded * z_generation

        # Concatenate full sequence
        z_noisy = torch.cat([z_context, z_noisy_generation], dim=1)
        sigma = torch.cat([sigma_context, sigma_generation], dim=1)

        # Condition dropout (for classifier-free guidance)
        drop_mask = drop_mask_override
        drop_mask_relevance = None
        drop_mask_text_condition_token = None
        drop_mask_context = None
        drop_mask_global = drop_mask_global_override
        if getattr(self, "unconditioned_pretraining", False):
            drop_mask = torch.ones(B, dtype=torch.bool, device=device)
            if self.simtok_enabled:
                drop_mask_relevance = torch.ones(B, dtype=torch.bool, device=device)
        if self.training:
            if self.p_drop_context > 0.0:
                drop_mask_context = torch.rand(B, device=device) < self.p_drop_context
            if not getattr(self, "unconditioned_pretraining", False):
                if float(getattr(self, "p_drop_relevance", 0.0)) > 0.0:
                    drop_mask_relevance = torch.rand(B, device=device) < float(self.p_drop_relevance)
                if float(getattr(self, "p_drop_text_condition_token", 0.0)) > 0.0:
                    p_drop_text = float(self.p_drop_text_condition_token)
                    drop_mask_text_condition_token = torch.rand(B, device=device) < p_drop_text
                p_drop_joint = float(getattr(self, "p_drop_condition_and_context", 0.0))
                p_drop_ctx = float(getattr(self, "p_drop_condition_context", 0.0))
                p_drop_fut = float(getattr(self, "p_drop_condition_future", 0.0))
                p_drop_global = float(getattr(self, "p_drop_global_condition", 0.0))
                split_drop_active = (p_drop_joint > 0.0) or (p_drop_ctx > 0.0) or (p_drop_fut > 0.0) or (p_drop_global > 0.0)
                if split_drop_active:
                    if drop_mask is None and (p_drop_ctx > 0.0 or p_drop_fut > 0.0):
                        drop_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
                        if p_drop_ctx > 0.0:
                            ctx_seq_drop = torch.rand(B, device=device) < p_drop_ctx
                            drop_mask[:, :ctx_len] = ctx_seq_drop.unsqueeze(1)
                        if p_drop_fut > 0.0:
                            fut_seq_drop = torch.rand(B, device=device) < p_drop_fut
                            drop_mask[:, ctx_len:] = fut_seq_drop.unsqueeze(1)
                    if p_drop_joint > 0.0:
                        joint_drop = torch.rand(B, device=device) < p_drop_joint
                        drop_mask_context = joint_drop if drop_mask_context is None else (drop_mask_context | joint_drop)
                        if drop_mask is None:
                            drop_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
                        drop_mask[:, :ctx_len] |= joint_drop.unsqueeze(1)
                        drop_mask[:, ctx_len:] |= joint_drop.unsqueeze(1)
                    if drop_mask_global is None and p_drop_global > 0.0 and self.global_condition_enabled:
                        drop_mask_global = torch.rand(B, device=device) < p_drop_global

        # Forward pass with precomputed task masks.
        block_mask_task = None
        task_block_masks = None
        if self.factorized_attention_enabled:
            task_block_masks = self.get_factorized_task_masks(T, device)
        else:
            num_prefix_tokens = self.num_prefix_tokens_task
            block_mask_task = self.get_block_mask(
                T, device, num_prefix_tokens=num_prefix_tokens
            )
        pred = self.forward(
            z_noisy,
            sigma,
            condition_disc=condition_disc_all,
            condition_disc_weight=condition_disc_weight_all,
            condition_cont=condition_cont_all,
            condition_mode=condition_mode_all,
            relevance_scores=relevance_scores_all,
            relevance_embedding_type=relevance_embedding_type_all,
            subject_token=subject_token,
            age=age,
            sex=sex,
            motion=motion,
            field_strength=field_strength,
            drop_mask=drop_mask,
            drop_mask_relevance=drop_mask_relevance,
            drop_mask_text_condition_token=drop_mask_text_condition_token,
            drop_mask_context=drop_mask_context,
            drop_mask_global=drop_mask_global,
            block_mask=block_mask_task,
            task_block_masks=task_block_masks,
        )

        # Extract generation predictions only
        pred_gen = pred[:, ctx_len:]  # [B, gen_len, K, D]

        # Compute target based on prediction type
        if self.prediction_type == 'v':
            # V-prediction: target is velocity v = clean - noise
            target_gen = z_generation - z_noise_gen
        else:
            # X-prediction: target is clean signal
            target_gen = z_generation

        if self.ramp_loss_weight:
            weight = (0.9 * (1 - sigma_generation) + 0.1) * frame_weight # inverted!
        else:
            weight = frame_weight
        loss = self._latent_mse(pred_gen, target_gen, weight[:, :, None, None])

        log_data = {
            'dynamics_loss': loss.item(),
        }

        return loss, log_data
