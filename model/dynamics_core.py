"""Core FMRIDynamics model definition and forward pass."""

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask

from .dynamics_layers import (
    RMSNorm,
    DynamicsBlock,
    get_timestep_embedding,
    create_parallel_denoising_mask_mod,
)
from .dynamics_losses import DynamicsLossMixin
from .dynamics_sampling import DynamicsSamplingMixin

class FMRIDynamics(DynamicsLossMixin, DynamicsSamplingMixin, nn.Module):
    """
    Flow matching dynamics model for fMRI latent sequences.

    Input per timestep:
    - condition: categorical condition index (e.g., 0=fixation, 1=shape, 2=face)
    - sigma: signal level for flow matching
    - z_noisy: noisy latent tokens [num_latents × latent_dim]

    Output:
    - z_clean_pred: predicted clean latent tokens [num_latents × latent_dim]

    Args:
        num_latents: number of latent tokens from tokenizer (default 18)
        latent_dim: dimension of each latent token (default 32)
        d_model: transformer hidden dimension (default None → latent_dim, no projection)
        num_heads: number of attention heads (default 8)
        num_layers: number of transformer layers (default 6)
        num_conditions: number of categorical conditions (default 3: fixation/shape/face)
        mlp_ratio: MLP hidden dim multiplier (default 4.0)
        dropout: dropout rate (default 0.0)
        max_context_length: maximum number of timesteps (default 16)
    """

    def __init__(
        self,
        num_latents=18,
        latent_dim=32,
        d_model=None,  # defaults to latent_dim (no projection)
        num_heads=8,
        num_layers=6,
        num_conditions=3,
        condition_cont_dim=None,  # Optional continuous condition dimension (e.g. DreamSim embedding dim)
        condition_cont_layout='standard',  # standard | h5_v2
        condition_cont_use_projection=True,  # If False, require condition_cont_dim == d_model and use identity
        condition_cont_proj_bias=True,  # If True, projection adds learned bias even for zero condition_cont.
        mlp_ratio=4.0,
        dropout=0.0,
        max_context_length=16,
        p_drop_context=0.0,
        p_drop_condition_and_context=0.0,
        p_drop_condition_context=0.0,
        p_drop_condition_future=0.0,
        p_drop_global_condition=0.0,
        p_drop_instruct=0.0,
        p_drop_sensory=0.0,
        p_drop_response=0.0,
        p_drop_relevance=0.0,
        p_drop_text_condition_token=0.0,
        unconditioned_pretraining=False,
        ramp_loss_weight=False,
        # AR training params
        ar_training_steps=0,  # 0 = disabled
        ar_training_denoise_steps=2,  # Euler steps for AR generation during training
        context_sigma=1.0,  # sigma for imperfect context (AR-generated or corrupted)
        # Attention stabilization
        soft_cap=30.0,  # attention logit soft capping (None = disabled)
        # Register tokens (within-timestep scratchpad)
        num_registers=0,  # Number of register tokens per timestep (0 = disabled)
        register_temporal_embed=False,  # Add temporal position embedding to registers
        # Prediction parameterization
        prediction_type='x',  # 'x' = predict clean, 'v' = predict velocity
        # Parallel denoising params
        parallel_shared_sigma=False,  # If True, sample one sigma per sample across all generation frames
        context_frames=8,  # Number of clean context frames (sigma=1)
        generation_frames=16,  # Number of frames to denoise in parallel
        # Subject token conditioning
        subject_token_enabled=False,
        subject_context_length=None,
        subject_encoder_layers=2,
        subject_allow_missing_token=False,
        # Global conditioning prefix tokens
        global_condition_enabled=False,
        roi_encoder=None,  # {'mode': 'linear'|'hemispheric'|'yeo7', 'input_dim': int, 'num_tokens': int, 'emb_dim': int}
        # ROI-token mode: one scalar BOLD token per ROI
        one_roi_one_token=False,
        roi_simtok_mixer=False,
        roi_language_embeddings=None,  # [num_latents, E] (truncated to d_model)
        # Optional similarity token: cosine(condition_cont, ROI language embeddings)
        simtok_enabled=False,
        simtok_roi_embeddings=None,  # [num_rois, condition_cont_dim] for the standard backend
        simtok_num_rois=None,
        relevance_mode=None,  # none | fixed
        relevance_include_condition_token=True,  # if False, condition token is scorer-only (not inserted into dynamics tokens)
        relevance_level_type_embedding_enabled=True,
        # Task-stage factorized attention
        factorized_attention=None,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.d_model = d_model if d_model is not None else latent_dim
        self.num_conditions = num_conditions
        self.condition_cont_dim = condition_cont_dim
        self.condition_cont_layout = str(condition_cont_layout)
        self.condition_cont_use_projection = bool(condition_cont_use_projection)
        self.condition_cont_proj_bias = bool(condition_cont_proj_bias)
        self.max_context_length = max_context_length
        self.p_drop_context = p_drop_context
        self.p_drop_condition_and_context = float(p_drop_condition_and_context)
        self.p_drop_condition_context = float(p_drop_condition_context)
        self.p_drop_condition_future = float(p_drop_condition_future)
        self.p_drop_global_condition = float(p_drop_global_condition)
        self.p_drop_instruct = float(p_drop_instruct)
        self.p_drop_sensory = float(p_drop_sensory)
        self.p_drop_response = float(p_drop_response)
        self.p_drop_relevance = float(p_drop_relevance)
        self.p_drop_text_condition_token = float(p_drop_text_condition_token)
        self.unconditioned_pretraining = bool(unconditioned_pretraining)
        self.ramp_loss_weight = ramp_loss_weight
        self.ar_training_steps = ar_training_steps
        self.ar_training_denoise_steps = ar_training_denoise_steps
        self.context_sigma = context_sigma
        self.num_registers = num_registers
        self.register_temporal_embed = register_temporal_embed
        self.prediction_type = prediction_type
        # Legacy compatibility attribute. The model is always parallel now.
        self.parallel_generation = True
        self.parallel_shared_sigma = bool(parallel_shared_sigma)
        self.context_frames = context_frames
        self.generation_frames = generation_frames
        # Legacy compatibility attribute. Two-stage/base blocks were removed.
        self.num_base_layers = 0
        self.subject_token_enabled = subject_token_enabled
        self.subject_context_length = subject_context_length
        self.subject_allow_missing_token = subject_allow_missing_token
        self.global_condition_enabled = bool(global_condition_enabled)
        self.one_roi_one_token = bool(one_roi_one_token)
        self.roi_encoder = None if roi_encoder is None else dict(roi_encoder)
        self.roi_encoder_mode = None if roi_encoder is None else str(roi_encoder['mode']).lower()
        if self.roi_encoder_mode is not None:
            if self.roi_encoder_mode not in ('linear', 'hemispheric', 'yeo7'):
                raise ValueError(
                    f"Unsupported roi_encoder.mode={self.roi_encoder_mode!r}. "
                    "Expected one of: 'linear', 'hemispheric', 'yeo7'."
                )
            self.roi_encoder_input_dim = int(roi_encoder['input_dim'])
            self.roi_encoder_num_tokens = int(roi_encoder['num_tokens'])
            self.roi_encoder_emb_dim = int(roi_encoder['emb_dim'])
            if self.roi_encoder_mode == 'yeo7':
                self.roi_encoder_group_indices = [
                    torch.as_tensor(indices, dtype=torch.long)
                    for indices in roi_encoder['group_indices']
                ]
                valid_mask = torch.zeros(
                    self.roi_encoder_num_tokens,
                    self.roi_encoder_input_dim,
                    dtype=torch.float32,
                )
                for token_idx, indices in enumerate(self.roi_encoder_group_indices):
                    valid_mask[token_idx, :len(indices)] = 1.0
                self.register_buffer("roi_encoder_valid_mask", valid_mask, persistent=False)
            else:
                self.roi_encoder_group_indices = None
                self.register_buffer("roi_encoder_valid_mask", None, persistent=False)
            if self.one_roi_one_token:
                raise ValueError("roi_encoder is not used with one_roi_one_token=True")
            if self.latent_dim != self.roi_encoder_input_dim:
                raise ValueError(
                    f"roi_encoder input_dim={self.roi_encoder_input_dim} must match latent_dim={self.latent_dim}"
                )
            if self.num_latents != self.roi_encoder_num_tokens:
                raise ValueError(
                    f"roi_encoder num_tokens={self.roi_encoder_num_tokens} must match num_latents={self.num_latents}"
                )
            if self.d_model != self.roi_encoder_emb_dim:
                raise ValueError(
                    f"roi_encoder emb_dim={self.roi_encoder_emb_dim} must match d_model={self.d_model}"
                )
        else:
            self.roi_encoder_input_dim = None
            self.roi_encoder_num_tokens = 0
            self.roi_encoder_emb_dim = None
            self.roi_encoder_group_indices = None
            self.register_buffer("roi_encoder_valid_mask", None, persistent=False)
        if relevance_mode is None:
            relevance_mode = 'fixed' if bool(simtok_enabled) else 'none'
        self.relevance_mode = str(relevance_mode).lower()
        if self.relevance_mode not in ('none', 'fixed'):
            raise ValueError(
                f"Unsupported relevance_mode={relevance_mode!r}. "
                "Expected one of: 'none', 'fixed'."
            )
        self.relevance_use_cosine = self.relevance_mode == 'fixed'
        self.simtok_enabled = self.relevance_mode != 'none'
        if self.condition_cont_layout not in ('standard', 'h5_v2'):
            raise ValueError("condition_cont_layout must be 'standard' or 'h5_v2'")
        if (not bool(relevance_include_condition_token)) and (not self.simtok_enabled):
            raise ValueError(
                "relevance_include_condition_token=False requires relevance_mode != 'none' "
                "(a relevance token must be present per timestep)."
            )
        self.relevance_include_condition_token = bool(relevance_include_condition_token)
        if self.p_drop_text_condition_token > 0.0 and not self.relevance_include_condition_token:
            raise ValueError(
                "p_drop_text_condition_token requires relevance_include_condition_token=True"
            )
        self.relevance_level_type_embedding_enabled = bool(relevance_level_type_embedding_enabled)
        split_drop_enabled = (
            self.p_drop_condition_context > 0.0
            or self.p_drop_condition_and_context > 0.0
            or self.p_drop_condition_future > 0.0
            or self.p_drop_global_condition > 0.0
            or self.p_drop_instruct > 0.0
            or self.p_drop_sensory > 0.0
            or self.p_drop_response > 0.0
        )
        for key, value in (
            ("p_drop_context", float(self.p_drop_context)),
            ("p_drop_condition_and_context", self.p_drop_condition_and_context),
            ("p_drop_condition_context", self.p_drop_condition_context),
            ("p_drop_condition_future", self.p_drop_condition_future),
            ("p_drop_global_condition", self.p_drop_global_condition),
            ("p_drop_instruct", self.p_drop_instruct),
            ("p_drop_sensory", self.p_drop_sensory),
            ("p_drop_response", self.p_drop_response),
            ("p_drop_relevance", self.p_drop_relevance),
            ("p_drop_text_condition_token", self.p_drop_text_condition_token),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{key} must be in [0, 1], got {value}")
        if self.subject_token_enabled:
            assert subject_context_length is not None and int(subject_context_length) > 0, (
                "subject_context_length must be set and > 0 when subject_token_enabled=True"
            )
            assert int(subject_encoder_layers) > 0, (
                "subject_encoder_layers must be > 0 when subject_token_enabled=True"
            )
            self.subject_context_length = int(subject_context_length)

        # Per-timestep layout: [(optional condition), (optional simtok), sigma, latents, registers]
        self.num_simtok_tokens = 0
        if self.simtok_enabled:
            self.num_simtok_tokens = self.roi_encoder_num_tokens if self.roi_encoder_mode is not None else 1
        self.num_task_condition_tokens = (
            (1 if self.relevance_include_condition_token else 0)
            + self.num_simtok_tokens
        )
        self.task_latent_start_idx = self.num_task_condition_tokens + 1
        self.signal_tokens_per_timestep = 1 + num_latents + num_registers
        self.tokens_per_timestep = self.num_task_condition_tokens + self.signal_tokens_per_timestep

        # Condition embedding: direct lookup by condition index
        # e.g., 0=fixation, 1=shape, 2=face
        self.condition_embed = nn.Embedding(num_conditions, self.d_model)
        self.condition_tripartite = self.condition_cont_layout == 'h5_v2'
        self.condition_tripartite_input_dim = None
        self.condition_tripartite_instruction_dim = None
        self.condition_tripartite_sensory_dim = None
        self.condition_tripartite_response_dim = None
        if self.condition_tripartite:
            if condition_cont_dim is None:
                raise ValueError("condition_cont_layout='h5_v2' requires condition_cont_dim")
            if not self.condition_cont_use_projection:
                raise ValueError("condition_cont_layout='h5_v2' requires condition_cont_use_projection=True")
            if (int(condition_cont_dim) - 2) % 3 != 0:
                raise ValueError(
                    "condition_cont_layout='h5_v2' expects condition_cont_dim = 3 * piece_dim + 2 "
                    f"(got {condition_cont_dim})"
                )
            self.condition_tripartite_input_dim = (int(condition_cont_dim) - 2) // 3
            self.condition_tripartite_instruction_dim = (2 * self.d_model) // 5
            self.condition_tripartite_sensory_dim = (2 * self.d_model) // 5
            self.condition_tripartite_response_dim = (
                self.d_model
                - self.condition_tripartite_instruction_dim
                - self.condition_tripartite_sensory_dim
            )
            if self.condition_tripartite_instruction_dim <= 0 or self.condition_tripartite_sensory_dim <= 0:
                raise ValueError("condition_cont_layout='h5_v2' requires d_model >= 5")
            self.condition_cont_proj = None
            self.condition_instruct_proj = nn.Linear(
                self.condition_tripartite_input_dim,
                self.condition_tripartite_instruction_dim,
                bias=False,
            )
            self.condition_sensory_proj = nn.Linear(
                self.condition_tripartite_input_dim,
                self.condition_tripartite_sensory_dim,
                bias=False,
            )
            self.condition_response_proj = nn.Linear(
                self.condition_tripartite_input_dim,
                self.condition_tripartite_response_dim,
                bias=False,
            )
            self.condition_instruct_mask_token = nn.Parameter(
                torch.randn(self.condition_tripartite_instruction_dim) * 0.02
            )
            self.condition_sensory_mask_token = nn.Parameter(
                torch.randn(self.condition_tripartite_sensory_dim) * 0.02
            )
            self.condition_response_mask_token = nn.Parameter(
                torch.randn(self.condition_tripartite_response_dim) * 0.02
            )
            self.condition_response_no_response_token = nn.Parameter(
                torch.randn(self.condition_tripartite_response_dim) * 0.02
            )
        else:
            if condition_cont_dim is None:
                self.condition_cont_proj = None
            elif self.condition_cont_use_projection:
                self.condition_cont_proj = nn.Linear(
                    condition_cont_dim,
                    self.d_model,
                    bias=self.condition_cont_proj_bias,
                )
            else:
                if int(condition_cont_dim) != int(self.d_model):
                    raise ValueError(
                        "condition_cont_use_projection=False requires condition_cont_dim == d_model "
                        f"(got condition_cont_dim={condition_cont_dim}, d_model={self.d_model})"
                    )
                self.condition_cont_proj = nn.Identity()

        if (self.p_drop_instruct > 0.0 or self.p_drop_sensory > 0.0 or self.p_drop_response > 0.0) and not self.condition_tripartite:
            raise ValueError(
                "Piece dropout requires condition_cont_layout='h5_v2'"
            )

        # Learnable mask token for condition dropout (classifier-free guidance)
        # Only initialized if any condition-token dropout is enabled.
        if (
            self.p_drop_condition_and_context > 0.0
            or self.p_drop_condition_context > 0.0
            or self.p_drop_condition_future > 0.0
            or self.p_drop_text_condition_token > 0.0
        ):
            self.condition_mask_token = nn.Parameter(torch.randn(self.d_model) * 0.02)
        if self.simtok_enabled and (
            self.p_drop_relevance > 0.0
            or self.p_drop_condition_and_context > 0.0
            or self.p_drop_condition_context > 0.0
            or self.p_drop_condition_future > 0.0
        ):
            self.relevance_mask_token = nn.Parameter(torch.randn(self.d_model) * 0.02)
        if float(p_drop_context) > 0.0 or self.p_drop_condition_and_context > 0.0:
            self.context_mask_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        # Sigma (signal level) embedding: sinusoidal + MLP
        self.sigma_embed = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        if self.roi_encoder_mode is not None:
            self.latent_in_proj = None
            self.latent_out_proj = None
            self.roi_in_proj = nn.ModuleList([
                nn.Linear(self.roi_encoder_input_dim, self.d_model)
                for _ in range(self.roi_encoder_num_tokens)
            ])
            self.roi_out_proj = nn.ModuleList([
                nn.Linear(self.d_model, self.roi_encoder_input_dim)
                for _ in range(self.roi_encoder_num_tokens)
            ])
        # Latent projection: only if d_model != latent_dim
        elif self.d_model != latent_dim:
            self.latent_in_proj = nn.Linear(latent_dim, self.d_model)
            self.latent_out_proj = nn.Linear(self.d_model, latent_dim)
            self.roi_in_proj = None
            self.roi_out_proj = None
        else:
            self.latent_in_proj = None
            self.latent_out_proj = None
            self.roi_in_proj = None
            self.roi_out_proj = None

        if self.one_roi_one_token:
            if roi_language_embeddings is None:
                raise ValueError("one_roi_one_token=True requires roi_language_embeddings")
            roi_emb = torch.as_tensor(roi_language_embeddings, dtype=torch.float32)
            if roi_emb.ndim != 2:
                raise ValueError(
                    f"roi_language_embeddings must have rank 2 [num_latents, emb_dim], got shape {tuple(roi_emb.shape)}"
                )
            if roi_emb.shape[0] != self.num_latents:
                raise ValueError(
                    f"roi_language_embeddings first dim must match num_latents={self.num_latents}, got {roi_emb.shape[0]}"
                )
            if roi_emb.shape[1] < self.d_model:
                raise ValueError(
                    f"roi_language_embeddings dim {roi_emb.shape[1]} must be >= d_model={self.d_model}"
                )
            self.register_buffer(
                "roi_language_embed",
                roi_emb[:, :self.d_model].view(1, 1, self.num_latents, self.d_model),
                persistent=True,
            )
            self.roi_pos_embed = nn.Parameter(
                torch.randn(1, 1, self.num_latents, self.d_model) * 0.02
            )
        else:
            self.register_buffer("roi_language_embed", None, persistent=False)
            self.roi_pos_embed = None

        if self.simtok_enabled:
            if simtok_roi_embeddings is None:
                self.register_buffer("simtok_roi_embeddings", None, persistent=False)
                if simtok_num_rois is None:
                    raise ValueError("simtok_enabled=True requires simtok_num_rois when simtok_roi_embeddings is None")
                self.simtok_num_rois = int(simtok_num_rois)
            else:
                simtok_roi = torch.as_tensor(simtok_roi_embeddings, dtype=torch.float32)
                simtok_roi = simtok_roi / simtok_roi.norm(dim=1, keepdim=True).clamp(min=1e-8)
                self.register_buffer("simtok_roi_embeddings", simtok_roi, persistent=True)
                self.simtok_num_rois = int(simtok_roi.shape[0])
                if simtok_num_rois is not None and int(simtok_num_rois) != self.simtok_num_rois:
                    raise ValueError(
                        f"simtok_num_rois={simtok_num_rois} does not match simtok_roi_embeddings rows {self.simtok_num_rois}"
                    )
        else:
            self.register_buffer("simtok_roi_embeddings", None, persistent=False)
            self.simtok_num_rois = 0
        if self.roi_encoder_mode == 'linear' and self.simtok_enabled:
            if self.simtok_num_rois != self.roi_encoder_input_dim:
                raise ValueError(
                    f"linear roi_encoder requires simtok_num_rois={self.roi_encoder_input_dim}, got {self.simtok_num_rois}"
                )
        if self.roi_encoder_mode == 'hemispheric' and self.simtok_enabled:
            if self.simtok_num_rois != 2 * self.roi_encoder_input_dim:
                raise ValueError(
                    f"hemispheric roi_encoder requires simtok_num_rois={2 * self.roi_encoder_input_dim}, got {self.simtok_num_rois}"
                )
        if self.roi_encoder_mode == 'yeo7' and self.simtok_enabled:
            if self.simtok_num_rois != int(roi_encoder['num_rois']):
                raise ValueError(
                    f"yeo7 roi_encoder requires simtok_num_rois={int(roi_encoder['num_rois'])}, got {self.simtok_num_rois}"
                )
        self.roi_simtok_mixer_enabled = bool(roi_simtok_mixer)
        if self.roi_simtok_mixer_enabled:
            assert self.simtok_enabled
            assert self.num_latents == 1
            assert not self.one_roi_one_token
            assert self.roi_encoder_mode is None
            assert self.simtok_num_rois <= self.latent_dim
            self.roi_simtok_mixer = nn.Linear(2, 2)
        self.relevance_score_override = None
        self.relevance_debug_print = False
        self._relevance_debug_printed = False

        # Token type embeddings for signal tokens [sigma, latents, registers] and prefix tokens.
        self.token_type_embed = nn.Parameter(torch.randn(1, self.signal_tokens_per_timestep, self.d_model) * 0.02)
        if self.relevance_include_condition_token:
            self.condition_type_embed = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        else:
            self.register_parameter("condition_type_embed", None)
        if self.simtok_enabled:
            self.simtok_type_embed = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
            self.relevance_type_embedding = nn.Embedding(3, self.d_model)
            nn.init.zeros_(self.relevance_type_embedding.weight)

        # Register tokens (learnable scratchpad for within-timestep computation)
        if num_registers > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, num_registers, self.d_model) * 0.02)

        # Temporal position embedding.
        max_seq_len = max(max_context_length, context_frames + generation_frames)
        self.temporal_embed = nn.Parameter(torch.randn(1, max_seq_len, 1, self.d_model) * 0.02)

        self._parse_factorized_attention_config(factorized_attention, num_layers)

        self.blocks = nn.ModuleList([
            DynamicsBlock(self.d_model, num_heads, mlp_ratio, dropout, soft_cap=soft_cap)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(self.d_model)

        if self.subject_token_enabled:
            self.subject_type_embed = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
            self.subject_mask_token = nn.Parameter(torch.randn(1, self.d_model) * 0.02)
            self.subject_cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
            self.subject_spatial_embed = nn.Parameter(
                torch.randn(1, 1, self.num_latents, self.d_model) * 0.02
            )
            self.subject_temporal_embed = nn.Parameter(
                torch.randn(1, self.subject_context_length, 1, self.d_model) * 0.02
            )
            subject_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=num_heads,
                dim_feedforward=int(self.d_model * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.subject_encoder = nn.TransformerEncoder(subject_layer, num_layers=subject_encoder_layers)
            self.subject_encoder_norm = RMSNorm(self.d_model)

        if self.global_condition_enabled:
            self.age_proj = nn.Linear(1, self.d_model)
            self.age_mask_token = nn.Parameter(torch.randn(1, self.d_model) * 0.02)
            self.age_type_embed = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

            self.sex_embed = nn.Embedding(2, self.d_model)
            self.sex_mask_token = nn.Parameter(torch.randn(1, self.d_model) * 0.02)
            self.sex_type_embed = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

            self.motion_proj = nn.Linear(1, self.d_model)
            self.motion_mask_token = nn.Parameter(torch.randn(1, self.d_model) * 0.02)
            self.motion_type_embed = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

            self.field_strength_embed = nn.Embedding(2, self.d_model)
            self.field_strength_mask_token = nn.Parameter(torch.randn(1, self.d_model) * 0.02)
            self.field_strength_type_embed = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        self.num_prefix_tokens_task = (1 if self.subject_token_enabled else 0) + (
            4 if self.global_condition_enabled else 0
        )

        # Parallel denoising mask.
        self.mask_mod = create_parallel_denoising_mask_mod(
            self.tokens_per_timestep,
            context_frames,
            num_prefix_tokens=self.num_prefix_tokens_task,
        )
        self._block_mask_cache = {}
        self._factorized_task_mask_cache = {}
        self._factorized_mask_sparsity_logged = set()

    @staticmethod
    def convert_legacy_state_dict(state_dict):
        """Convert old checkpoint formats. (did some simplification for the public release and stripped out a bunch of unused functionality.
        This is the cost thereof!)

        Supported legacy formats:
        - two-stage checkpoints with num_base_layers=0:
          task_blocks.X.* -> blocks.X.*, token_type_embed_base -> token_type_embed
        - pre-two-stage checkpoints:
          blocks.X.* stays blocks.X.*, token_type_embed is split if condition_type_embed is absent

        Returns converted state_dict (or original if already new format).
        """
        keys = set(state_dict.keys())
        if any(k.startswith('base_blocks.') for k in keys):
            raise ValueError(
                "Cannot load two-stage checkpoints with base_blocks into the simplified dynamics model. "
                "Only checkpoints trained with num_base_layers=0 are supported."
            )

        has_two_stage_task = any(k.startswith('task_blocks.') for k in keys)
        has_old_combined_token_type = (
            'token_type_embed' in keys
            and 'condition_type_embed' not in keys
            and 'token_type_embed_base' not in keys
        )
        needs_conversion = (
            has_two_stage_task
            or 'token_type_embed_base' in keys
            or has_old_combined_token_type
        )
        if not needs_conversion:
            return state_dict

        new_state_dict = {}
        split_combined_token_type = has_old_combined_token_type
        for key, value in state_dict.items():
            if key.startswith('task_blocks.'):
                new_key = key.replace('task_blocks.', 'blocks.', 1)
                new_state_dict[new_key] = value
            elif key == 'token_type_embed_base':
                new_state_dict['token_type_embed'] = value
            elif key == 'token_type_embed' and split_combined_token_type:
                # Old shape: [1, condition + sigma + latents + registers, D].
                new_state_dict['condition_type_embed'] = value[:, 0:1, :]
                new_state_dict['token_type_embed'] = value[:, 1:, :]
            else:
                new_state_dict[key] = value

        return new_state_dict

    def _parse_factorized_attention_config(self, factorized_attention, num_layers):
        cfg = factorized_attention or {}
        if not isinstance(cfg, dict):
            raise ValueError(
                "dynamics.factorized_attention must be a dict when provided "
                f"(got {type(cfg).__name__})"
            )

        self.factorized_attention_enabled = bool(cfg.get("enabled", False))

        stage = str(cfg.get("stage", "task")).strip().lower()
        if self.factorized_attention_enabled and stage != "task":
            raise ValueError(
                f"dynamics.factorized_attention.stage must be 'task' in v1, got {cfg.get('stage')}"
            )
        self.factorized_attention_stage = stage
        temporal_layer_type = str(cfg.get("temporal_layer_type", "attention")).strip().lower()
        if temporal_layer_type != "attention":
            raise ValueError(
                "dynamics.factorized_attention.temporal_layer_type must be 'attention', "
                f"got {cfg.get('temporal_layer_type')}"
            )
        self.factorized_temporal_layer_type = temporal_layer_type

        allowed_block_sizes = {16, 32, 64, 128}
        block_size_spatial = int(cfg.get("block_size_spatial", 32))
        block_size_temporal = int(cfg.get("block_size_temporal", 16))
        if block_size_spatial not in allowed_block_sizes:
            raise ValueError(
                "dynamics.factorized_attention.block_size_spatial must be one of "
                f"{sorted(allowed_block_sizes)}, got {block_size_spatial}"
            )
        if block_size_temporal not in allowed_block_sizes:
            raise ValueError(
                "dynamics.factorized_attention.block_size_temporal must be one of "
                f"{sorted(allowed_block_sizes)}, got {block_size_temporal}"
            )
        self.factorized_block_size_spatial = block_size_spatial
        self.factorized_block_size_temporal = block_size_temporal

        mask_build_device_raw = cfg.get("mask_build_device", "cpu")
        self.factorized_mask_build_device = torch.device(mask_build_device_raw)
        if self.factorized_mask_build_device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "dynamics.factorized_attention.mask_build_device must resolve to cpu/cuda, "
                f"got {mask_build_device_raw}"
            )

        cycle_pattern = bool(cfg.get("cycle_pattern", True))
        task_pattern_raw = cfg.get("task_pattern", ["S", "T"])
        if isinstance(task_pattern_raw, str):
            task_pattern_raw = [task_pattern_raw]
        if not isinstance(task_pattern_raw, (list, tuple)):
            raise ValueError(
                "dynamics.factorized_attention.task_pattern must be a list/tuple of 'S'/'T'"
            )
        normalized_pattern = [str(item).strip().upper() for item in task_pattern_raw]
        if self.factorized_attention_enabled:
            if num_layers <= 0:
                raise ValueError("factorized_attention.enabled=True requires at least one block")
            if self.factorized_temporal_layer_type == "attention":
                if len(normalized_pattern) == 0:
                    raise ValueError("dynamics.factorized_attention.task_pattern must be non-empty when enabled")
                invalid = [p for p in normalized_pattern if p not in {"S", "T"}]
                if invalid:
                    raise ValueError(
                        "dynamics.factorized_attention.task_pattern entries must be 'S'/'T', "
                        f"got invalid values {invalid}"
                    )

        self.factorized_task_pattern = normalized_pattern
        self.factorized_cycle_pattern = cycle_pattern
        if self.factorized_attention_enabled:
            self.factorized_task_schedule = self._expand_factorized_task_pattern(
                normalized_pattern,
                cycle_pattern,
                num_layers,
            )
        else:
            self.factorized_task_schedule = []

    def _expand_factorized_task_pattern(self, pattern, cycle_pattern, num_layers):
        if num_layers == 0:
            return []
        if not pattern:
            return []
        if cycle_pattern:
            return [pattern[i % len(pattern)] for i in range(num_layers)]
        if len(pattern) != num_layers:
            raise ValueError(
                "dynamics.factorized_attention.task_pattern length must match number of blocks "
                f"({num_layers}) when cycle_pattern=false, got {len(pattern)}"
            )
        return list(pattern)

    def _log_factorized_mask_sparsity_once(self, kind, num_timesteps, block_size, block_mask):
        log_key = (kind, num_timesteps, block_size)
        if log_key in self._factorized_mask_sparsity_logged:
            return
        self._factorized_mask_sparsity_logged.add(log_key)
        sparsity = float(block_mask.sparsity())
        if not (0.0 <= sparsity <= 100.0):
            print(
                f"[FMRIDynamics] factorized mask {kind} "
                f"(T={num_timesteps}, block={block_size}) "
                f"sparsity=nan (invalid BlockMask.sparsity={sparsity:.6f})"
            )
            return
        print(
            f"[FMRIDynamics] factorized mask {kind} "
            f"(T={num_timesteps}, block={block_size}) "
            f"sparsity={sparsity:.6f}"
        )

    def get_factorized_task_masks(self, num_timesteps, device):
        """Get cached task-stage masks for axial factorized attention."""
        device = torch.device(device)
        num_prefix_tokens = self.num_prefix_tokens_task
        full_mask = self.get_block_mask(
            num_timesteps,
            device,
            num_prefix_tokens=num_prefix_tokens,
        )
        if not self.factorized_attention_enabled:
            return {"S": full_mask, "T": full_mask, "FULL": full_mask}
        if num_prefix_tokens > 0:
            raise ValueError(
                "factorized_attention.enabled=true uses axial task attention and currently "
                "does not support prefix tokens (subject/global condition tokens)."
            )

        tokens_per_timestep = self.tokens_per_timestep
        masks = {"FULL": full_mask}
        for mask_kind in ("S", "T"):
            if mask_kind == "S":
                def mask_mod(b, h, q_idx, kv_idx):
                    return q_idx >= 0
                q_len = tokens_per_timestep
                kv_len = tokens_per_timestep
                block_size = self.factorized_block_size_spatial
            else:
                ctx = self.context_frames
                def mask_mod(b, h, q_idx, kv_idx):
                    q_is_generation = q_idx >= ctx
                    kv_is_context = kv_idx < ctx
                    return q_is_generation | kv_is_context
                q_len = num_timesteps
                kv_len = num_timesteps
                block_size = self.factorized_block_size_temporal

            cache_key = (
                mask_kind,
                num_timesteps,
                device,
                num_prefix_tokens,
                self.factorized_block_size_spatial,
                self.factorized_block_size_temporal,
                self.factorized_mask_build_device,
            )
            if cache_key not in self._factorized_task_mask_cache:
                block_mask = create_block_mask(
                    mask_mod,
                    B=None, H=None,
                    Q_LEN=q_len, KV_LEN=kv_len,
                    device=self.factorized_mask_build_device,
                    BLOCK_SIZE=block_size,
                )
                if self.factorized_mask_build_device != device:
                    block_mask = block_mask.to(device)
                self._factorized_task_mask_cache[cache_key] = block_mask
                self._log_factorized_mask_sparsity_once(
                    mask_kind, num_timesteps, block_size, block_mask
                )
            masks[mask_kind] = self._factorized_task_mask_cache[cache_key]
        return masks

    def get_block_mask(self, num_timesteps, device, num_prefix_tokens=None):
        """Get or create the FlexAttention block mask for given sequence length.

        Args:
            num_timesteps: number of timesteps in sequence
            device: torch device
        """
        if num_prefix_tokens is None:
            num_prefix_tokens = self.num_prefix_tokens_task
        assert num_prefix_tokens == self.num_prefix_tokens_task, (
            f"Expected num_prefix_tokens={self.num_prefix_tokens_task}, got {num_prefix_tokens}"
        )
        cache_key = (num_timesteps, device, num_prefix_tokens)
        if cache_key not in self._block_mask_cache:
            tokens_per_timestep = self.tokens_per_timestep
            total_tokens = num_prefix_tokens + num_timesteps * tokens_per_timestep
            self._block_mask_cache[cache_key] = create_block_mask(
                self.mask_mod,
                B=None, H=None,
                Q_LEN=total_tokens, KV_LEN=total_tokens,
                device=device,
            )
        return self._block_mask_cache[cache_key]

    def _normalize_condition_inputs(self, condition, condition_disc, condition_cont, condition_mode, B, T, device):
        """
        Normalize mixed condition inputs to a unified representation.

        Priority:
        1) condition_disc (explicit)
        2) condition (legacy positional categorical input)
        """
        if condition_disc is None and condition is not None:
            condition_disc = condition

        if condition_disc is None and condition_cont is None:
            condition_disc = torch.zeros((B, T), dtype=torch.long, device=device)

        # Infer mode if not provided
        if condition_mode is None:
            if condition_cont is None:
                condition_mode = torch.zeros((B, T), dtype=torch.long, device=device)
            elif condition_disc is None:
                condition_mode = torch.ones((B, T), dtype=torch.long, device=device)
            else:
                # Mixed fallback: -1 in discrete channel marks continuous positions
                condition_mode = (condition_disc < 0).long()
        else:
            condition_mode = condition_mode.long()

        # Ensure discrete tensor exists (for unified downstream logic)
        if condition_disc is None:
            condition_disc = torch.zeros((B, T), dtype=torch.long, device=device)
        else:
            condition_disc = condition_disc.long()

        # Ensure continuous tensor exists if needed
        if condition_cont is None:
            if condition_mode.any():
                D = self.condition_cont_dim if self.condition_cont_dim is not None else self.d_model
                condition_cont = torch.zeros((B, T, D), dtype=torch.float32, device=device)
        else:
            condition_cont = condition_cont.float()

        return condition_disc, condition_cont, condition_mode

    def _sample_condition_piece_drop_mask(self, drop_mask, B, T, device):
        if not self.condition_tripartite:
            return None

        piece_drop_mask = torch.zeros((B, T, 3), dtype=torch.bool, device=device)
        if self.training:
            probs = (self.p_drop_instruct, self.p_drop_sensory, self.p_drop_response)
            ctx_len = min(int(self.context_frames), T)
            for piece_idx, p_drop in enumerate(probs):
                if p_drop <= 0.0:
                    continue
                if ctx_len > 0:
                    ctx_drop = torch.rand(B, device=device) < p_drop
                    piece_drop_mask[:, :ctx_len, piece_idx] = ctx_drop.unsqueeze(1)
                if ctx_len < T:
                    fut_drop = torch.rand(B, device=device) < p_drop
                    piece_drop_mask[:, ctx_len:, piece_idx] = fut_drop.unsqueeze(1)

        if drop_mask is not None:
            if drop_mask.ndim == 1:
                piece_drop_mask |= drop_mask[:, None, None]
            else:
                piece_drop_mask |= drop_mask[:, :, None]
        return piece_drop_mask

    def _build_tripartite_condition_tokens(self, condition_cont, piece_drop_mask):
        assert self.condition_tripartite
        raw = condition_cont[..., :-2]
        response_special = condition_cont[..., -2:]
        instruct_raw, sensory_raw, response_raw = torch.split(
            raw,
            self.condition_tripartite_input_dim,
            dim=-1,
        )

        instruct = self.condition_instruct_proj(instruct_raw)
        sensory = self.condition_sensory_proj(sensory_raw)
        response = self.condition_response_proj(response_raw)
        response = response + (
            response_special[..., 0:1] * self.condition_response_no_response_token.view(1, 1, -1)
        )
        response = response + (
            response_special[..., 1:2] * self.condition_response_mask_token.view(1, 1, -1)
        )

        if piece_drop_mask is not None:
            instruct = self._apply_drop_mask(
                instruct,
                self.condition_instruct_mask_token.view(1, 1, -1),
                piece_drop_mask[..., 0],
            )
            sensory = self._apply_drop_mask(
                sensory,
                self.condition_sensory_mask_token.view(1, 1, -1),
                piece_drop_mask[..., 1],
            )
            response = self._apply_drop_mask(
                response,
                self.condition_response_mask_token.view(1, 1, -1),
                piece_drop_mask[..., 2],
            )

        return torch.cat([instruct, sensory, response], dim=-1)

    def set_relevance_score_override(self, fn=None):
        """Set optional runtime override for per-timestep relevance scores.

        If set, fn must accept (task_tokens, roi_rows, condition_cont) and return [B, T, R].
        """
        self.relevance_score_override = fn
        self._relevance_debug_printed = False

    def _build_condition_tokens(
        self,
        condition=None,
        condition_disc=None,
        condition_disc_weight=None,
        condition_cont=None,
        condition_mode=None,
        drop_mask=None,
        drop_mask_text_condition_token=None,
        piece_drop_mask=None,
        B=None,
        T=None,
        device=None,
    ):
        """
        Build per-timestep condition tokens for mixed discrete/continuous conditioning.

        Returns:
            cond_tokens: [B, T, 1, d_model]
        """
        condition_disc, condition_cont, condition_mode = self._normalize_condition_inputs(
            condition, condition_disc, condition_cont, condition_mode, B, T, device
        )
        if condition_disc_weight is not None:
            condition_disc_weight = condition_disc_weight.to(device=device, dtype=torch.float32)

        # Discrete path (clamp invalid IDs like -1 to 0 before embedding lookup)
        disc_ids = condition_disc.clamp(min=0, max=self.num_conditions - 1)
        disc_tokens = self.condition_embed(disc_ids)  # [B, T, d_model]

        # Continuous path (optional)
        if condition_cont is not None and self.condition_tripartite:
            cont_tokens = self._build_tripartite_condition_tokens(condition_cont, piece_drop_mask)
        elif condition_cont is not None and self.condition_cont_proj is not None:
            cont_tokens = self.condition_cont_proj(condition_cont)  # [B, T, d_model]
        else:
            cont_tokens = None
            if condition_mode.bool().any():
                raise ValueError(
                    "condition_mode includes continuous entries but condition_cont_proj is not set. "
                    "Set dynamics.condition_cont_dim when using continuous conditioning."
                )

        # Select source per timestep: 0=discrete, 1=continuous
        if cont_tokens is None:
            cond_tokens = disc_tokens
        else:
            use_cont = condition_mode.bool().unsqueeze(-1)  # [B, T, 1]
            cond_tokens = torch.where(use_cont, cont_tokens, disc_tokens)

        if condition_disc_weight is not None:
            if condition_disc_weight.ndim == 2:
                # [B, T, d_model] + [B, T, 1] * [B, T, d_model]
                disc_add = condition_disc_weight.unsqueeze(-1) * disc_tokens
            else:
                # [B, T, C] @ [C, d_model] -> [B, T, d_model]
                disc_add = torch.matmul(condition_disc_weight, self.condition_embed.weight)
            cond_tokens = cond_tokens + disc_add

        condition_drop_mask = self._combine_drop_masks(drop_mask, drop_mask_text_condition_token, T)

        # Apply condition dropout: mask may be [B] or [B, T].
        if condition_drop_mask is not None and condition_drop_mask.any() and hasattr(self, 'condition_mask_token'):
            if condition_drop_mask.ndim == 1:
                drop_mask_expanded = condition_drop_mask[:, None, None].float()  # [B, 1, 1]
            else:
                drop_mask_expanded = condition_drop_mask.unsqueeze(-1).float()  # [B, T, 1]
            mask_token = self.condition_mask_token[None, None, :]  # [1, 1, d_model]
            cond_tokens = (1 - drop_mask_expanded) * cond_tokens + drop_mask_expanded * mask_token

        cond_tokens = cond_tokens.unsqueeze(2)  # [B, T, 1, d_model]
        return cond_tokens

    def _compute_simtok_scores(
        self,
        condition=None,
        condition_disc=None,
        condition_disc_weight=None,
        condition_cont=None,
        condition_mode=None,
        relevance_scores=None,
        piece_drop_mask=None,
        B=None,
        T=None,
        device=None,
    ):
        _, condition_cont, condition_mode = self._normalize_condition_inputs(
            condition, condition_disc, condition_cont, condition_mode, B, T, device
        )
        if condition_cont is None:
            condition_cont = torch.zeros((B, T, self.condition_cont_dim), dtype=torch.float32, device=device)

        task_tokens = self._build_condition_tokens(
            condition=condition,
            condition_disc=condition_disc,
            condition_disc_weight=condition_disc_weight,
            condition_cont=condition_cont,
            condition_mode=condition_mode,
            drop_mask=None,
            piece_drop_mask=piece_drop_mask,
            B=B,
            T=T,
            device=device,
        ).squeeze(2)

        cond_vec = condition_cont.to(dtype=torch.float32)
        sim = None
        sim_source = self.relevance_mode
        roi_rows = None
        direct_relevance = None
        if relevance_scores is not None:
            direct_relevance = relevance_scores.to(device=device, dtype=cond_vec.dtype)
            if direct_relevance.ndim == 4:
                if piece_drop_mask is not None:
                    keep = (~piece_drop_mask).to(dtype=direct_relevance.dtype).unsqueeze(-1)
                    direct_relevance = direct_relevance * keep
                direct_relevance = direct_relevance.sum(dim=2)
        if direct_relevance is not None:
            sim = direct_relevance
            sim_source = "precomputed"
        else:
            if self.condition_tripartite:
                raise ValueError(
                    "condition_cont_layout='h5_v2' requires direct relevance_scores when simtok is enabled"
                )
            if self.simtok_roi_embeddings is None:
                raise ValueError("simtok_enabled=True requires relevance_scores or simtok_roi_embeddings")
            roi_rows = self.simtok_roi_embeddings.to(device=device, dtype=cond_vec.dtype)
            if self.relevance_score_override is not None:
                sim = self.relevance_score_override(task_tokens, roi_rows, cond_vec)
                sim_source = "override"
            elif self.relevance_use_cosine:
                cond_norm = cond_vec / cond_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                roi_norm = roi_rows / roi_rows.norm(dim=1, keepdim=True).clamp(min=1e-8)
                sim = torch.matmul(cond_norm, roi_norm.transpose(0, 1))

        if sim.shape[-1] != self.simtok_num_rois:
            raise ValueError(f"Expected relevance scores width {self.simtok_num_rois}, got {sim.shape[-1]}")
        sim = sim.to(device=device, dtype=cond_vec.dtype)
        if self.relevance_debug_print and (not self._relevance_debug_printed):
            sim_detached = sim.detach()
            print(
                "[relevance] "
                f"source={sim_source} mean={float(sim_detached.mean()):.6f} "
                f"std={float(sim_detached.std()):.6f} "
                f"max={float(sim_detached.max()):.6f}"
            )
            self._relevance_debug_printed = True
        return sim

    def _combine_drop_masks(self, left, right, T):
        if left is None:
            return right
        if right is None:
            return left
        if left.ndim == 1 and right.ndim == 1:
            return left | right
        if left.ndim == 1:
            left = left[:, None].expand(-1, T)
        if right.ndim == 1:
            right = right[:, None].expand(-1, T)
        return left | right

    def _build_simtok_drop_mask(self, drop_mask, drop_mask_relevance, T):
        return self._combine_drop_masks(drop_mask_relevance, drop_mask, T)

    def _project_roi_tokens_in(self, roi_tokens):
        assert self.roi_encoder_mode is not None
        projected = [
            proj(roi_tokens[:, :, token_idx, :])
            for token_idx, proj in enumerate(self.roi_in_proj)
        ]
        return torch.stack(projected, dim=2)

    def _project_roi_tokens_out(self, latent_tokens):
        assert self.roi_encoder_mode is not None
        projected = [
            proj(latent_tokens[:, :, token_idx, :])
            for token_idx, proj in enumerate(self.roi_out_proj)
        ]
        return torch.stack(projected, dim=2)

    def _project_simtok_scores_to_tokens(self, sim, B, T, device):
        if self.roi_encoder_mode == 'linear':
            return self.roi_in_proj[0](sim).unsqueeze(2)
        if self.roi_encoder_mode == 'hemispheric':
            mid = self.roi_encoder_input_dim
            left = self.roi_in_proj[0](sim[:, :, :mid])
            right = self.roi_in_proj[1](sim[:, :, mid:])
            return torch.stack([left, right], dim=2)
        if self.roi_encoder_mode == 'yeo7':
            grouped = sim.new_zeros((B, T, self.roi_encoder_num_tokens, self.roi_encoder_input_dim))
            for token_idx, indices in enumerate(self.roi_encoder_group_indices):
                idx = indices.to(device=device)
                grouped[:, :, token_idx, :len(indices)] = sim.index_select(-1, idx)
            return self._project_roi_tokens_in(grouped)
        sim_padded = torch.zeros((B, T, self.latent_dim), dtype=sim.dtype, device=device)
        sim_padded[:, :, :self.simtok_num_rois] = sim
        sim_latent = sim_padded.unsqueeze(2)
        if self.latent_in_proj is not None:
            return self.latent_in_proj(sim_latent)
        return sim_latent

    def _mix_roi_latent_and_simtok_scores(self, z_noisy, sim, simtok_drop_mask):
        z_mixed = z_noisy.clone()
        sim_for_mix = sim
        if simtok_drop_mask is not None and simtok_drop_mask.any():
            if simtok_drop_mask.ndim == 1:
                sim_for_mix = sim_for_mix.masked_fill(simtok_drop_mask[:, None, None], 0.0)
            else:
                sim_for_mix = sim_for_mix.masked_fill(simtok_drop_mask[:, :, None], 0.0)
        x = torch.stack([z_mixed[:, :, 0, :self.simtok_num_rois], sim_for_mix], dim=-1)
        x = self.roi_simtok_mixer(x)
        z_mixed[:, :, 0, :self.simtok_num_rois] = x[..., 0]
        return z_mixed, x[..., 1]

    def _build_simtok_tokens(
        self,
        condition=None,
        condition_disc=None,
        condition_disc_weight=None,
        condition_cont=None,
        condition_mode=None,
        relevance_scores=None,
        simtok_drop_mask=None,
        piece_drop_mask=None,
        B=None,
        T=None,
        device=None,
    ):
        """
        Build per-timestep relevance tokens from continuous condition embeddings.

        Shapes:
            condition_cont: [B, T, D_cond]
            relevance_scores: [B, T, R]
            simtok_roi_embeddings: [R, D_cond]
            simtok output: [B, T, 1, d_model]
        """
        sim = self._compute_simtok_scores(
            condition=condition,
            condition_disc=condition_disc,
            condition_disc_weight=condition_disc_weight,
            condition_cont=condition_cont,
            condition_mode=condition_mode,
            relevance_scores=relevance_scores,
            piece_drop_mask=piece_drop_mask,
            B=B,
            T=T,
            device=device,
        )
        sim_tokens = self._project_simtok_scores_to_tokens(sim, B, T, device)
        if simtok_drop_mask is not None and simtok_drop_mask.any():
            assert hasattr(self, 'relevance_mask_token')
            if simtok_drop_mask.ndim == 1:
                drop_mask_expanded = simtok_drop_mask[:, None, None, None].float()
            else:
                drop_mask_expanded = simtok_drop_mask[:, :, None, None].float()
            mask_token = self.relevance_mask_token[None, None, None, :]
            sim_tokens = (1 - drop_mask_expanded) * sim_tokens + drop_mask_expanded * mask_token
        return sim_tokens

    def build_subject_token(self, subject_latents):
        """
        Encode distant subject context into one token.

        Args:
            subject_latents: [B, T_subj, K, D_latent]
        Returns:
            subject_token: [B, d_model]
        """
        assert self.subject_token_enabled, "build_subject_token called but subject_token_enabled=False"
        B, T_subj, K, D = subject_latents.shape
        assert T_subj == self.subject_context_length, (
            f"Expected subject_context_length={self.subject_context_length}, got {T_subj}"
        )
        assert K == self.num_latents, f"Expected num_latents={self.num_latents}, got {K}"
        assert D == self.latent_dim, f"Expected latent_dim={self.latent_dim}, got {D}"

        if self.roi_encoder_mode is not None:
            subj_tokens = self._project_roi_tokens_in(subject_latents)
        elif self.latent_in_proj is not None:
            subj_tokens = self.latent_in_proj(subject_latents)
        else:
            subj_tokens = subject_latents

        subj_tokens = subj_tokens + self.subject_spatial_embed + self.subject_temporal_embed[:, :T_subj]
        subj_tokens = subj_tokens.view(B, T_subj * K, self.d_model)

        cls = self.subject_cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        subj_seq = torch.cat([cls, subj_tokens], dim=1)  # [B, 1 + T_subj*K, d_model]
        subj_seq = self.subject_encoder(subj_seq)
        subj_seq = self.subject_encoder_norm(subj_seq)
        return subj_seq[:, 0, :]  # [B, d_model]

    def _apply_drop_mask(self, token_vec, mask_token, drop_mask):
        if drop_mask is None:
            return token_vec
        keep = (~drop_mask).float().unsqueeze(-1)
        return keep * token_vec + (1.0 - keep) * mask_token

    def _build_subject_prefix_token(self, subject_token, drop_mask, B, device):
        assert self.subject_token_enabled
        if subject_token is None:
            assert self.subject_allow_missing_token, (
                "Subject token missing. Set subject_allow_missing_token=True to use subject mask token."
            )
            subj = self.subject_mask_token.expand(B, -1)
        else:
            assert subject_token.shape == (B, self.d_model), (
                f"Expected subject_token shape {(B, self.d_model)}, got {tuple(subject_token.shape)}"
            )
            subj = subject_token
        mask_tok = self.subject_mask_token.expand(B, -1)
        subj = self._apply_drop_mask(subj, mask_tok, drop_mask)
        return (subj.unsqueeze(1) + self.subject_type_embed).to(device)

    def _build_global_prefix_tokens(self, age, sex, motion, field_strength, drop_mask, B, device):
        assert self.global_condition_enabled

        age_mask = self.age_mask_token.expand(B, -1)
        if age is None:
            age_vec = age_mask
        else:
            age_in = age.float()
            age_valid = torch.isfinite(age_in) & (age_in >= 0.0) & (age_in <= 100.0)
            age_safe = torch.where(age_valid, age_in, torch.zeros_like(age_in))
            age_proj = self.age_proj((age_safe.clamp(0.0, 100.0) / 100.0).unsqueeze(-1))
            age_vec = torch.where(age_valid.unsqueeze(-1), age_proj, age_mask)
        age_vec = self._apply_drop_mask(age_vec, age_mask, drop_mask)

        sex_mask = self.sex_mask_token.expand(B, -1)
        if sex is None:
            sex_vec = sex_mask
        else:
            sex_in = sex.long()
            sex_valid = (sex_in == 0) | (sex_in == 1)
            sex_safe = torch.where(sex_valid, sex_in, torch.zeros_like(sex_in))
            sex_emb = self.sex_embed(sex_safe.clamp(min=0, max=1))
            sex_vec = torch.where(sex_valid.unsqueeze(-1), sex_emb, sex_mask)
        sex_vec = self._apply_drop_mask(sex_vec, sex_mask, drop_mask)

        motion_mask = self.motion_mask_token.expand(B, -1)
        if motion is None:
            motion_vec = motion_mask
        else:
            motion_in = motion.float()
            motion_valid = torch.isfinite(motion_in)
            motion_safe = torch.where(motion_valid, motion_in, torch.zeros_like(motion_in))
            motion_proj = self.motion_proj(motion_safe.unsqueeze(-1))
            motion_vec = torch.where(motion_valid.unsqueeze(-1), motion_proj, motion_mask)
        motion_vec = self._apply_drop_mask(motion_vec, motion_mask, drop_mask)

        fs_mask = self.field_strength_mask_token.expand(B, -1)
        if field_strength is None:
            fs_vec = fs_mask
        else:
            fs_in = field_strength.long()
            fs_valid = (fs_in == 0) | (fs_in == 1)
            fs_emb = self.field_strength_embed(fs_in.clamp(min=0, max=1))
            fs_vec = torch.where(fs_valid.unsqueeze(-1), fs_emb, fs_mask)
        fs_vec = self._apply_drop_mask(fs_vec, fs_mask, drop_mask)

        return torch.cat(
            [
                (age_vec.unsqueeze(1) + self.age_type_embed),
                (sex_vec.unsqueeze(1) + self.sex_type_embed),
                (motion_vec.unsqueeze(1) + self.motion_type_embed),
                (fs_vec.unsqueeze(1) + self.field_strength_type_embed),
            ],
            dim=1,
        ).to(device)

    def _build_prefix_tokens(
        self,
        subject_token,
        age,
        sex,
        motion,
        field_strength,
        drop_mask,
        drop_mask_global,
        B,
        device,
    ):
        if self.num_prefix_tokens_task == 0:
            return None
        drop_mask_subject = drop_mask if (drop_mask is not None and drop_mask.ndim == 1) else None
        drop_mask_global_eff = drop_mask_global if drop_mask_global is not None else drop_mask_subject
        pieces = []
        if self.subject_token_enabled:
            pieces.append(self._build_subject_prefix_token(subject_token, drop_mask_subject, B, device))
        if self.global_condition_enabled:
            pieces.append(
                self._build_global_prefix_tokens(
                    age, sex, motion, field_strength, drop_mask_global_eff, B, device
                )
            )
        return torch.cat(pieces, dim=1)

    def forward(
        self,
        z_noisy,
        sigma,
        condition=None,
        condition_disc=None,
        condition_disc_weight=None,
        condition_cont=None,
        condition_mode=None,
        relevance_scores=None,
        relevance_embedding_type=None,
        subject_token=None,
        age=None,
        sex=None,
        motion=None,
        field_strength=None,
        drop_mask=None,
        drop_mask_relevance=None,
        drop_mask_text_condition_token=None,
        drop_mask_context=None,
        drop_mask_global=None,
        block_mask=None,
        task_block_masks=None,
    ):
        """
        Forward pass: predict clean latents from noisy latents.

        Token layout per timestep:
        [condition, (optional simtok), sigma, latents, registers]

        Args:
            z_noisy: [B, T, num_latents, latent_dim] noisy latent sequences
            sigma: [B, T] signal levels τ ∈ [0, 1] for each timestep
            condition: [B, T] legacy categorical condition indices (backward-compatible)
            condition_disc: [B, T] discrete condition indices (with -1 allowed at continuous positions)
            condition_disc_weight: [B, T] optional additive weights for discrete condition embeddings
            condition_cont: [B, T, D_cond] continuous condition vectors
            condition_mode: [B, T] 0=discrete, 1=continuous
            relevance_scores: [B, T, R] optional direct per-ROI simtok scores
            relevance_embedding_type: [B] or [B, T] optional simtok type ids (0=roi, 1=network, 2=mask)
            subject_token: [B, d_model] global subject token from distant-context encoder
            age: [B] age in years
            sex: [B] sex code (0/1; other values -> mask token)
            motion: [B] per-run motion scalar
            field_strength: [B] field-strength bucket (0=3T, 1=7T)
            drop_mask: [B] or [B, T] bool tensor for condition token dropout
            drop_mask_relevance: [B] or [B, T] bool tensor for simtok dropout
            drop_mask_text_condition_token: [B] or [B, T] bool tensor for text condition-token-only dropout
            drop_mask_context: [B] bool tensor for context latent-token dropout
            drop_mask_global: [B] bool tensor for global-prefix token dropout
            block_mask: optional pre-computed FlexAttention block mask (for task stage parallel mode)
            task_block_masks: optional dict of pre-computed task masks with keys in {'S','T','FULL'}

        Returns:
            z_pred: [B, T, num_latents, latent_dim] predicted clean latents
        """
        B, T, K, D = z_noisy.shape
        device = z_noisy.device

        assert K == self.num_latents, f"Expected {self.num_latents} latents, got {K}"
        assert D == self.latent_dim, f"Expected latent_dim {self.latent_dim}, got {D}"
        max_seq = max(self.max_context_length, self.context_frames + self.generation_frames)
        assert T <= max_seq, f"Sequence length {T} exceeds max {max_seq}"
        condition_piece_drop_mask = self._sample_condition_piece_drop_mask(drop_mask, B, T, device)
        text_condition_drop_mask = drop_mask_text_condition_token
        if float(getattr(self, "p_drop_text_condition_token", 0.0)) >= 1.0:
            force_text_drop = torch.ones((B, T), dtype=torch.bool, device=device)
            text_condition_drop_mask = self._combine_drop_masks(text_condition_drop_mask, force_text_drop, T)
        simtok_drop_mask = None
        mixed_sim = None
        relevance_type_ids = None

        # === Build signal tokens: [sigma, latents, registers] ===
        tokens_list = []

        # 1. Sigma embedding: sinusoidal + MLP
        sigma_flat = sigma.reshape(-1)  # [B*T]
        sigma_sinusoidal = get_timestep_embedding(sigma_flat * 1000, self.d_model)  # [B*T, d_model]
        sigma_embed = self.sigma_embed(sigma_sinusoidal)  # [B*T, d_model]
        sigma_tokens = sigma_embed.view(B, T, 1, self.d_model)  # [B, T, 1, d_model]
        tokens_list.append(sigma_tokens)

        # 2. Latent tokens: project to d_model if needed
        if self.roi_encoder_valid_mask is not None:
            z_noisy = z_noisy * self.roi_encoder_valid_mask.to(z_noisy).view(
                1, 1, self.num_latents, self.latent_dim
            )
        if self.roi_simtok_mixer_enabled:
            sim = self._compute_simtok_scores(
                condition=condition,
                condition_disc=condition_disc,
                condition_disc_weight=condition_disc_weight,
                condition_cont=condition_cont,
                condition_mode=condition_mode,
                relevance_scores=relevance_scores,
                piece_drop_mask=condition_piece_drop_mask,
                B=B,
                T=T,
                device=device,
            )
            simtok_drop_mask = self._build_simtok_drop_mask(drop_mask, drop_mask_relevance, T)
            z_noisy, mixed_sim = self._mix_roi_latent_and_simtok_scores(z_noisy, sim, simtok_drop_mask)
        if self.roi_encoder_mode is not None:
            latent_tokens = self._project_roi_tokens_in(z_noisy)
        elif self.latent_in_proj is not None:
            latent_tokens = self.latent_in_proj(z_noisy)
        else:
            latent_tokens = z_noisy  # d_model == latent_dim, no projection
        if self.one_roi_one_token:
            # [B, T, K, D]: add frozen semantic ROI descriptors and learned ROI identity.
            latent_tokens = latent_tokens + self.roi_language_embed + self.roi_pos_embed
        if drop_mask_context is not None and drop_mask_context.any():
            ctx_len = min(int(self.context_frames), T)
            if ctx_len > 0:
                keep = (~drop_mask_context).to(dtype=latent_tokens.dtype).view(B, 1, 1, 1)
                mask = self.context_mask_token.view(1, 1, 1, self.d_model)
                latent_tokens[:, :ctx_len] = (
                    keep * latent_tokens[:, :ctx_len]
                    + (1.0 - keep) * mask
                )
        tokens_list.append(latent_tokens)

        # 3. Register tokens (learnable scratchpad, shared across timesteps)
        if self.num_registers > 0:
            reg_tokens = self.register_tokens.unsqueeze(1).expand(B, T, -1, -1)
            tokens_list.append(reg_tokens)

        # Concatenate signal tokens: [B, T, signal_tokens_per_timestep, d_model]
        tokens = torch.cat(tokens_list, dim=2)

        # Token type embedding for signal tokens: [sigma, latents, registers]
        tokens = tokens + self.token_type_embed

        # Temporal position embedding
        tokens = tokens + self.temporal_embed[:, :T, :, :]

        # Optionally add temporal embedding to register tokens
        if self.num_registers > 0 and self.register_temporal_embed:
            reg_start = 1 + self.num_latents  # sigma at 0, latents at 1:K+1, registers at K+1:
            tokens[:, :, reg_start:, :] = tokens[:, :, reg_start:, :] + self.temporal_embed[:, :T, 0:1, :]

        # === Insert conditioning token(s) and run transformer blocks ===

        task_prefix_tokens = []
        if self.relevance_include_condition_token:
            # Build condition tokens (mixed discrete/continuous)
            cond_tokens = self._build_condition_tokens(
                condition=condition,
                condition_disc=condition_disc,
                condition_disc_weight=condition_disc_weight,
                condition_cont=condition_cont,
                condition_mode=condition_mode,
                drop_mask=drop_mask,
                drop_mask_text_condition_token=text_condition_drop_mask,
                piece_drop_mask=condition_piece_drop_mask,
                B=B,
                T=T,
                device=device,
            )

            # Add embeddings to condition token
            cond_tokens = cond_tokens + self.condition_type_embed  # Token type for condition
            cond_tokens = cond_tokens + self.temporal_embed[:, :T, :, :]  # Same temporal position
            task_prefix_tokens.append(cond_tokens)
        if self.simtok_enabled:
            simtok_drop_mask = self._build_simtok_drop_mask(drop_mask, drop_mask_relevance, T)
            if self.roi_simtok_mixer_enabled:
                assert mixed_sim is not None
                simtok_tokens = self._project_simtok_scores_to_tokens(mixed_sim, B, T, device)
                if simtok_drop_mask is not None and simtok_drop_mask.any():
                    assert hasattr(self, 'relevance_mask_token')
                    if simtok_drop_mask.ndim == 1:
                        drop_mask_expanded = simtok_drop_mask[:, None, None, None].float()
                    else:
                        drop_mask_expanded = simtok_drop_mask[:, :, None, None].float()
                    mask_token = self.relevance_mask_token[None, None, None, :]
                    simtok_tokens = (1 - drop_mask_expanded) * simtok_tokens + drop_mask_expanded * mask_token
            else:
                simtok_tokens = self._build_simtok_tokens(
                    condition=condition,
                    condition_disc=condition_disc,
                    condition_disc_weight=condition_disc_weight,
                    condition_cont=condition_cont,
                    condition_mode=condition_mode,
                    relevance_scores=relevance_scores,
                    simtok_drop_mask=simtok_drop_mask,
                    piece_drop_mask=condition_piece_drop_mask,
                    B=B,
                    T=T,
                    device=device,
                )
            if relevance_embedding_type is None:
                relevance_type_ids = torch.zeros((B, T), device=device, dtype=torch.long)
            else:
                relevance_type_ids = relevance_embedding_type.to(device=device, dtype=torch.long)
                if relevance_type_ids.ndim == 1:
                    assert relevance_type_ids.shape == (B,)
                    relevance_type_ids = relevance_type_ids[:, None].expand(-1, T)
                else:
                    assert relevance_type_ids.shape == (B, T)
            if simtok_drop_mask is not None and simtok_drop_mask.any():
                if simtok_drop_mask.ndim == 1:
                    simtok_drop_mask = simtok_drop_mask[:, None].expand(-1, T)
                relevance_type_ids = relevance_type_ids.clone()
                relevance_type_ids[simtok_drop_mask] = 2
            if not self.relevance_level_type_embedding_enabled:
                relevance_type_ids = relevance_type_ids.masked_fill(relevance_type_ids == 1, 0)
            simtok_tokens = simtok_tokens + self.relevance_type_embedding(relevance_type_ids).unsqueeze(2)
            simtok_tokens = simtok_tokens + self.simtok_type_embed
            simtok_tokens = simtok_tokens + self.temporal_embed[:, :T, :, :]
            task_prefix_tokens.append(simtok_tokens)

        # Concatenate: [condition, (optional simtok), sigma, latents, registers]
        tokens = torch.cat(task_prefix_tokens + [tokens], dim=2)
        if self.factorized_attention_enabled:
            num_prefix_tokens = self.num_prefix_tokens_task
            if num_prefix_tokens > 0:
                raise ValueError(
                    "factorized_attention.enabled=true uses axial task attention and currently "
                    "does not support prefix tokens (subject/global condition tokens)."
                )
            if block_mask is not None:
                raise ValueError(
                    "block_mask override is incompatible with factorized_attention.enabled=true "
                    "(axial task attention)."
                )
            if task_block_masks is None:
                task_block_masks = self.get_factorized_task_masks(T, device)

            S_task = tokens.shape[2]
            for layer_idx, block in enumerate(self.blocks):
                layer_mode = self.factorized_task_schedule[layer_idx]
                if layer_mode == "S":
                    # Spatial pass: reshape [B, T, S, D] -> [B*T, S, D].
                    x = tokens.view(B * T, S_task, self.d_model)
                    x = block(x, task_block_masks["S"])
                    tokens = x.view(B, T, S_task, self.d_model)
                elif layer_mode == "T":
                    # Temporal pass: reshape [B, T, S, D] -> [B*S, T, D].
                    x = tokens.permute(0, 2, 1, 3).contiguous().view(B * S_task, T, self.d_model)
                    x = block(x, task_block_masks["T"])
                    tokens = x.view(B, S_task, T, self.d_model).permute(0, 2, 1, 3).contiguous()
                else:
                    raise ValueError(f"Unsupported factorized layer mode '{layer_mode}'")

            tokens = self.final_norm(tokens.view(B, T * S_task, self.d_model)).view(
                B, T, S_task, self.d_model
            )
        else:
            # Flatten for attention: [B, T * tokens_per_timestep, d_model]
            tokens = tokens.view(B, T * self.tokens_per_timestep, self.d_model)
            num_prefix_tokens = self.num_prefix_tokens_task
            if num_prefix_tokens > 0:
                prefix_tokens = self._build_prefix_tokens(
                    subject_token, age, sex, motion, field_strength, drop_mask, drop_mask_global, B, device
                )
                tokens = torch.cat([prefix_tokens, tokens], dim=1)  # [B, P + T*tokens_per_timestep, d_model]

            if block_mask is None:
                if task_block_masks is not None:
                    block_mask_task = task_block_masks["FULL"]
                else:
                    block_mask_task = self.get_block_mask(
                        T, device, num_prefix_tokens=num_prefix_tokens
                    )
            else:
                # Legacy/explicit full-mask override.
                block_mask_task = block_mask

            for block in self.blocks:
                tokens = block(tokens, block_mask_task)

            tokens = self.final_norm(tokens)
            if num_prefix_tokens > 0:
                tokens = tokens[:, num_prefix_tokens:, :]

            # Reshape back: [B, T, tokens_per_timestep, d_model]
            tokens = tokens.view(B, T, self.tokens_per_timestep, self.d_model)

        # Extract latent tokens only (skip task-condition token(s) and sigma)
        latent_out = tokens[:, :, self.task_latent_start_idx:self.task_latent_start_idx+self.num_latents, :]

        # Project back to latent_dim if needed
        if self.roi_encoder_mode is not None:
            z_pred = self._project_roi_tokens_out(latent_out)
        elif self.latent_out_proj is not None:
            z_pred = self.latent_out_proj(latent_out)
        else:
            z_pred = latent_out  # d_model == latent_dim, no projection

        return z_pred
