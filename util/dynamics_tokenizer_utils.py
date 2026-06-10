import torch
import torch.nn as nn
import torch.nn.functional as F


def next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


class PassthroughEncoder(nn.Module):
    """
    Simple encoder for no-tokenizer mode with power-of-2 padding.

    Pads input to next power of 2 (required by flex_attention head_dim).
    Unpads in generate() for reconstruction.

    Args:
        num_rois: Number of ROIs (e.g., 450)
        crop_length: Temporal length per crop (e.g., 1)
        num_latents: Number of output latent tokens (typically 1)
    """

    def __init__(self, num_rois, crop_length, num_latents=1):
        super().__init__()
        self.num_rois = num_rois
        self.crop_length = crop_length
        self.input_timesteps = crop_length  # Alias for tokenizer interface
        self.num_latents = num_latents

        input_dim = num_rois * crop_length
        # Pad to next power of 2 for flex_attention compatibility
        self.latent_dim = next_power_of_2(input_dim)
        self.pad_size = self.latent_dim - input_dim

        print(f"PassthroughEncoder: {input_dim} -> {self.latent_dim} (pad {self.pad_size})")

    def forward(self, x):
        """
        Args:
            x: [B, num_rois, crop_length]
        Returns:
            z: [B, num_latents, latent_dim] (padded to power of 2)
        """
        B = x.shape[0]
        x_flat = x.view(B, -1)  # [B, num_rois * crop_length]

        # Pad with zeros to power of 2
        if self.pad_size > 0:
            z_flat = F.pad(x_flat, (0, self.pad_size))  # [B, latent_dim]
        else:
            z_flat = x_flat

        return z_flat.view(B, self.num_latents, self.latent_dim)

    def encoder(self, x):
        """Alias for forward() to match tokenizer interface."""
        return self.forward(x)

    @torch.no_grad()
    def generate(self, z, num_steps=1, shape=None):
        """Identity decoder: unpad and reshape latents back to signal space."""
        B = z.shape[0]
        z_flat = z.view(B, -1)  # [B, latent_dim]

        # Remove padding
        if self.pad_size > 0:
            z_flat = z_flat[:, :self.num_rois * self.crop_length]

        return z_flat.view(B, self.num_rois, self.crop_length)
