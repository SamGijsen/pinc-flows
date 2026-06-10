"""Building blocks for fMRI dynamics model."""

import math
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean subtraction, ~1.3x faster than LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight).to(dtype)


def generate_soft_cap_score_mod(soft_cap: float):
    """Generate attention score_mod for tanh soft capping (Gemma 2 / Grok-1 style)."""
    soft_cap = float(soft_cap)
    def score_mod(score, b, h, q_idx, kv_idx):
        cap = score.new_tensor(soft_cap)
        return cap * torch.tanh(score / cap)
    return score_mod

def get_timestep_embedding(timesteps, dim, max_period=10000):
    """
    Sinusoidal timestep embeddings for signal level τ.

    Args:
        timesteps: [B] tensor of values in [0, 1] or [0, 1000]
        dim: embedding dimension
    Returns:
        [B, dim] embeddings
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def create_block_causal_mask_mod(tokens_per_timestep, num_prefix_tokens=0):
    """
    Create mask_mod function for block-causal FlexAttention.

    Rules:
    - Tokens can attend to all tokens in same timestep (full attention within block)
    - Tokens can attend to all tokens in previous timesteps (causal across time)
    - Cannot attend to future timesteps

    Args:
        tokens_per_timestep: number of tokens per timestep (e.g., 20 = 1 task + 1 sigma + 18 latents)
    """
    def mask_mod(b, h, q_idx, kv_idx):
        q_is_prefix = q_idx < num_prefix_tokens
        kv_is_prefix = kv_idx < num_prefix_tokens

        q_rel = torch.clamp(q_idx - num_prefix_tokens, min=0)
        kv_rel = torch.clamp(kv_idx - num_prefix_tokens, min=0)
        q_timestep = q_rel // tokens_per_timestep
        kv_timestep = kv_rel // tokens_per_timestep
        causal_ok = kv_timestep <= q_timestep

        # Prefix query: prefix-only. Non-prefix query: prefix + causal within sequence.
        return torch.where(q_is_prefix, kv_is_prefix, kv_is_prefix | causal_ok)
    return mask_mod


def create_parallel_denoising_mask_mod(tokens_per_timestep, context_length, num_prefix_tokens=0):
    """
    Create mask_mod function for parallel denoising FlexAttention.

    Rules:
    - Context tokens (t < context_length): full bilateral attention within context
    - Generation tokens (t >= context_length): full attention to everything
      (all context + all other generation tokens)

    This enables parallel denoising of multiple future frames while conditioning
    on clean context frames.

    Args:
        tokens_per_timestep: number of tokens per timestep
        context_length: number of context timesteps (clean frames with sigma=1)
    """
    def mask_mod(b, h, q_idx, kv_idx):
        q_is_prefix = q_idx < num_prefix_tokens
        kv_is_prefix = kv_idx < num_prefix_tokens

        q_rel = torch.clamp(q_idx - num_prefix_tokens, min=0)
        kv_rel = torch.clamp(kv_idx - num_prefix_tokens, min=0)
        q_timestep = q_rel // tokens_per_timestep
        kv_timestep = kv_rel // tokens_per_timestep

        q_is_generation = q_timestep >= context_length
        kv_is_context = kv_timestep < context_length
        parallel_ok = q_is_generation | kv_is_context

        # Prefix query: prefix-only. Non-prefix query: prefix + parallel rule.
        return torch.where(q_is_prefix, kv_is_prefix, kv_is_prefix | parallel_ok)

    return mask_mod


def create_factorized_spatial_mask_mod(tokens_per_timestep, num_prefix_tokens=0):
    """
    Create spatial task mask for time-major task layout [t][s].

    Rules:
    - Prefix query: prefix-only.
    - Non-prefix query: prefix + same timestep only.
    """
    def mask_mod(b, h, q_idx, kv_idx):
        q_is_prefix = q_idx < num_prefix_tokens
        kv_is_prefix = kv_idx < num_prefix_tokens

        q_rel = torch.clamp(q_idx - num_prefix_tokens, min=0)
        kv_rel = torch.clamp(kv_idx - num_prefix_tokens, min=0)
        q_timestep = q_rel // tokens_per_timestep
        kv_timestep = kv_rel // tokens_per_timestep
        spatial_ok = q_timestep == kv_timestep

        return torch.where(q_is_prefix, kv_is_prefix, kv_is_prefix | spatial_ok)

    return mask_mod


def create_factorized_temporal_mask_mod(
    tokens_per_timestep,
    num_timesteps,
    num_prefix_tokens=0,
    parallel_mode=False,
    context_length=None,
):
    """
    Create temporal task mask for space-major task layout [s][t].

    Rules:
    - Prefix query: prefix-only.
    - Non-parallel mode: same slot only, causal over time.
    - Parallel mode: same slot only; context query -> context-only,
      generation query -> context + generation.
    """
    if parallel_mode and context_length is None:
        raise ValueError("context_length is required for parallel temporal factorized mask")

    def mask_mod(b, h, q_idx, kv_idx):
        q_is_prefix = q_idx < num_prefix_tokens
        kv_is_prefix = kv_idx < num_prefix_tokens

        q_rel = torch.clamp(q_idx - num_prefix_tokens, min=0)
        kv_rel = torch.clamp(kv_idx - num_prefix_tokens, min=0)

        # Space-major flattening: rel_idx = slot * T + timestep
        q_slot = q_rel // num_timesteps
        kv_slot = kv_rel // num_timesteps
        q_timestep = q_rel % num_timesteps
        kv_timestep = kv_rel % num_timesteps

        same_slot = q_slot == kv_slot
        if parallel_mode:
            q_is_generation = q_timestep >= context_length
            kv_is_context = kv_timestep < context_length
            temporal_ok = same_slot & (q_is_generation | kv_is_context)
        else:
            temporal_ok = same_slot & (kv_timestep <= q_timestep)

        return torch.where(q_is_prefix, kv_is_prefix, kv_is_prefix | temporal_ok)

    return mask_mod


class DynamicsBlock(nn.Module):
    """
    Transformer block for dynamics model with block-causal attention.

    Pre-norm transformer with modern stabilization:
    - RMSNorm (faster than LayerNorm)
    - QK-Norm (prevents attention logit explosion)
    - Soft capping (bounds attention scores)
    """

    def __init__(self, d_model, num_heads, mlp_ratio=4.0, dropout=0.0, soft_cap=30.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Attention with RMSNorm
        self.norm1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.q_norm = RMSNorm(self.head_dim)  # QK-Norm
        self.k_norm = RMSNorm(self.head_dim)  # QK-Norm
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # Soft capping score_mod (None = disabled)
        self.score_mod = generate_soft_cap_score_mod(soft_cap) if soft_cap is not None else None

        # MLP with RMSNorm
        self.norm2 = RMSNorm(d_model)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, block_mask):
        """
        Args:
            x: [B, N, D] input tokens
            block_mask: FlexAttention block mask for block-causal attention
        """
        B, N, D = x.shape

        # Self-attention with FlexAttention
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # QK-Norm: normalize Q and K per head
        q = self.q_norm(q)
        k = self.k_norm(k)

        # FlexAttention with block-causal mask and soft capping
        attn_out = flex_attention(q, k, v, block_mask=block_mask, score_mod=self.score_mod)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        x = x + self.attn_dropout(self.proj(attn_out))

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x
