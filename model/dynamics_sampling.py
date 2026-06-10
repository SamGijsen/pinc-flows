"""Sampling/inference routines for FMRIDynamics."""

import torch

class DynamicsSamplingMixin:
    def sample(self, z_context, cond_context, cond_future, num_steps=16, num_future=1,
               context_sigma=None, guidance_scale=1.0, context_guidance_scale=1.0,
               cond_cont_context=None, cond_cont_future=None,
               cond_mode_context=None, cond_mode_future=None,
               relevance_scores_context=None, relevance_scores_future=None,
               relevance_embedding_type=None,
               subject_token=None,
               age=None,
               sex=None,
               motion=None,
               field_strength=None):
        """
        Generate future latents via Euler integration.

        Args:
            z_context: [B, T_ctx, num_latents, latent_dim] context latents (clean)
            cond_context: [B, T_ctx] condition indices for context
            cond_future: [B, T_fut] condition indices for future timesteps
            num_steps: number of parallel Euler integration steps
            num_future: number of future timesteps to generate
            context_sigma: retained for legacy callers; parallel sampling keeps context clean.
            guidance_scale: classifier-free guidance scale. 1.0 = no guidance (default),
                           >1.0 = amplified conditioning, 0.0 = unconditional.
                           Requires condition-token dropout during training.
            context_guidance_scale: classifier-free guidance scale for recent-signal context.

        Returns:
            z_future: [B, T_fut, num_latents, latent_dim] generated future latents
        """
        return self._sample_parallel(
            z_context,
            cond_context,
            cond_future,
            num_steps,
            guidance_scale,
            context_guidance_scale,
            cond_cont_context=cond_cont_context,
            cond_cont_future=cond_cont_future,
            cond_mode_context=cond_mode_context,
            cond_mode_future=cond_mode_future,
            relevance_scores_context=relevance_scores_context,
            relevance_scores_future=relevance_scores_future,
            relevance_embedding_type=relevance_embedding_type,
            subject_token=subject_token,
            age=age,
            sex=sex,
            motion=motion,
            field_strength=field_strength,
        )

    @torch.no_grad()
    def _sample_parallel(
        self,
        z_context,
        cond_context,
        cond_future,
        num_steps,
        guidance_scale=1.0,
        context_guidance_scale=1.0,
        cond_cont_context=None,
        cond_cont_future=None,
        cond_mode_context=None,
        cond_mode_future=None,
        relevance_scores_context=None,
        relevance_scores_future=None,
        relevance_embedding_type=None,
        subject_token=None,
        age=None,
        sex=None,
        motion=None,
        field_strength=None,
    ):
        """
        Generate multiple future latents simultaneously via parallel Euler integration.

        All generation frames are denoised in parallel with bilateral attention,
        conditioned on clean context frames.

        Args:
            z_context: [B, ctx_len, num_latents, latent_dim] context latents (clean)
            cond_context: [B, ctx_len] condition indices for context
            cond_future: [B, gen_len] condition indices for generation frames
            num_steps: number of Euler integration steps
            guidance_scale: task CFG scale. 1.0 = conditional only, >1.0 = amplified conditioning
            context_guidance_scale: context-latent CFG scale.

        Returns:
            z_future: [B, gen_len, num_latents, latent_dim] generated future latents
        """
        B, ctx_len, K, D = z_context.shape
        gen_len = cond_future.shape[1]
        T = ctx_len + gen_len
        device = z_context.device

        use_task_cfg = guidance_scale != 1.0
        use_context_cfg = context_guidance_scale != 1.0
        use_joint_cfg = (
            use_task_cfg
            and use_context_cfg
            and float(guidance_scale) == float(context_guidance_scale)
        )
        use_task_uncond = use_task_cfg or use_context_cfg
        if use_task_uncond and not hasattr(self, 'condition_mask_token'):
            raise ValueError(
                "guidance needs condition_mask_token "
                "(condition-token dropout was not enabled during training)"
            )
        if use_context_cfg and not hasattr(self, 'context_mask_token'):
            raise ValueError(
                "context_guidance_scale != 1.0 but model has no context_mask_token "
                "(p_drop_context=0 during training)"
            )

        # Initialize ALL generation frames as noise
        z_future = torch.randn(B, gen_len, K, D, device=device)

        dt = 1.0 / num_steps

        # Get task-stage masks once and reuse for all Euler steps.
        block_mask = None
        task_block_masks = None
        if self.factorized_attention_enabled:
            task_block_masks = self.get_factorized_task_masks(T, device)
        else:
            num_prefix_tokens = self.num_prefix_tokens_task
            block_mask = self.get_block_mask(
                T, device, num_prefix_tokens=num_prefix_tokens
            )

        if use_task_uncond:
            drop_mask_uncond = torch.ones(B, device=device, dtype=torch.bool)
        if use_context_cfg:
            drop_mask_context_uncond = torch.ones(B, device=device, dtype=torch.bool)

        for step in range(num_steps):
            sigma_val = step / num_steps

            # Context: sigma=1 (clean), Generation: current sigma
            sigma_ctx = torch.ones(B, ctx_len, device=device)
            sigma_gen = torch.full((B, gen_len), sigma_val, device=device)
            sigma = torch.cat([sigma_ctx, sigma_gen], dim=1)

            # Full sequence with conditions
            z_full = torch.cat([z_context, z_future], dim=1)
            cond_full = torch.cat([cond_context, cond_future], dim=1)
            cond_full_cont = (
                torch.cat([cond_cont_context, cond_cont_future], dim=1)
                if cond_cont_context is not None and cond_cont_future is not None
                else None
            )
            cond_full_mode = (
                torch.cat([cond_mode_context, cond_mode_future], dim=1)
                if cond_mode_context is not None and cond_mode_future is not None
                else None
            )
            rel_full = (
                torch.cat([relevance_scores_context, relevance_scores_future], dim=1)
                if relevance_scores_context is not None and relevance_scores_future is not None
                else None
            )

            # Forward with parallel mask (all generation frames predicted at once)
            pred_cond = self.forward(
                z_full,
                sigma,
                condition_disc=cond_full,
                condition_cont=cond_full_cont,
                condition_mode=cond_full_mode,
                relevance_scores=rel_full,
                relevance_embedding_type=relevance_embedding_type,
                subject_token=subject_token,
                age=age,
                sex=sex,
                motion=motion,
                field_strength=field_strength,
                block_mask=block_mask,
                task_block_masks=task_block_masks,
            )
            pred_cond_future = pred_cond[:, ctx_len:]

            if use_task_uncond and not use_joint_cfg:
                pred_task_uncond = self.forward(
                    z_full,
                    sigma,
                    condition_disc=cond_full,
                    condition_cont=cond_full_cont,
                    condition_mode=cond_full_mode,
                    relevance_scores=rel_full,
                    relevance_embedding_type=relevance_embedding_type,
                    subject_token=subject_token,
                    age=age,
                    sex=sex,
                    motion=motion,
                    field_strength=field_strength,
                    drop_mask=drop_mask_uncond,
                    block_mask=block_mask,
                    task_block_masks=task_block_masks,
                )
                pred_task_uncond_future = pred_task_uncond[:, ctx_len:]
            if use_context_cfg:
                pred_all_uncond = self.forward(
                    z_full,
                    sigma,
                    condition_disc=cond_full,
                    condition_cont=cond_full_cont,
                    condition_mode=cond_full_mode,
                    relevance_scores=rel_full,
                    relevance_embedding_type=relevance_embedding_type,
                    subject_token=subject_token,
                    age=age,
                    sex=sex,
                    motion=motion,
                    field_strength=field_strength,
                    drop_mask=drop_mask_uncond,
                    drop_mask_context=drop_mask_context_uncond,
                    block_mask=block_mask,
                    task_block_masks=task_block_masks,
                )
                pred_all_uncond_future = pred_all_uncond[:, ctx_len:]

            # Euler step (all generation frames simultaneously)
            if sigma_val < 1.0 - 1e-6:
                if self.prediction_type == 'v':
                    # V-prediction: use predicted velocity directly
                    velocity_cond = pred_cond_future
                    if use_task_uncond and not use_joint_cfg:
                        velocity_task_uncond = pred_task_uncond_future
                    if use_context_cfg:
                        velocity_all_uncond = pred_all_uncond_future
                else:
                    # X-prediction: compute velocity from predicted clean
                    velocity_cond = (pred_cond_future - z_future) / (1 - sigma_val)
                    if use_task_uncond and not use_joint_cfg:
                        velocity_task_uncond = (pred_task_uncond_future - z_future) / (1 - sigma_val)
                    if use_context_cfg:
                        velocity_all_uncond = (pred_all_uncond_future - z_future) / (1 - sigma_val)

                velocity = velocity_cond
                if use_joint_cfg:
                    velocity = velocity + (guidance_scale - 1.0) * (
                        velocity_cond - velocity_all_uncond
                    )
                elif use_context_cfg:
                    velocity = velocity + (context_guidance_scale - 1.0) * (
                        velocity_task_uncond - velocity_all_uncond
                    )
                if use_task_cfg and not use_joint_cfg:
                    velocity = velocity + (guidance_scale - 1.0) * (
                        velocity_cond - velocity_task_uncond
                    )

                z_future = z_future + velocity * dt

        return z_future
