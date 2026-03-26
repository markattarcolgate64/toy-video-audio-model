"""DiT (Diffusion Transformer) for video generation in pixel space.

Operates on pixel videos of shape (B, 3, 16, 32, 32):
  - Patchify 32x32 spatial into 4x4 patches -> 8x8 = 64 spatial tokens per frame
  - 16 frames * 64 patches = 1024 tokens total
  - Each patch: 4*4*3 = 48 values, projected to hidden_dim

Uses adaLN-Zero conditioning from timestep embedding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning.

    Standard transformer: x = x + Attn(LN(x)); x = x + MLP(LN(x))
    DiT block:            x = x + gate1 * Attn(adaLN(x)); x = x + gate2 * MLP(adaLN(x))

    adaLN: LayerNorm(x) * (1 + scale) + shift, where scale/shift come from timestep
    Gates are initialized to zero for stable training.
    """

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads

        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # MLP
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )

        # adaLN modulation: projects timestep embedding to 6 * hidden_dim
        # (scale1, shift1, gate1, scale2, shift2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        # Initialize gate projections to zero
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, t_emb):
        """
        x: (B, N, D) — token sequence
        t_emb: (B, D) — timestep embedding
        """
        # Compute all 6 modulation parameters from timestep
        mod = self.adaLN_modulation(t_emb).unsqueeze(1)  # (B, 1, 6*D)
        scale1, shift1, gate1, scale2, shift2, gate2 = mod.chunk(6, dim=-1)

        # Attention path with adaLN
        x_norm = self.norm1(x) * (1 + scale1) + shift1
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate1 * attn_out

        # MLP path with adaLN
        x_norm = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.mlp(x_norm)

        return x


class DiT(nn.Module):
    """Diffusion Transformer for pixel-space video generation.

    Input: (B, in_channels, T, H, W) + timestep
    Output: (B, in_channels, T, H, W) — predicted noise
    """

    def __init__(
        self,
        in_channels=3,
        num_frames=16,
        image_size=32,
        patch_size=4,
        hidden_dim=384,
        num_heads=6,
        num_layers=6,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Patch dimensions
        self.num_spatial_patches = (image_size // patch_size) ** 2     # 64
        self.num_tokens = self.num_spatial_patches * num_frames         # 1024
        self.patch_dim = in_channels * patch_size * patch_size          # 48

        # Patchify: linear projection from patch values to hidden dim
        self.patch_embed = nn.Linear(self.patch_dim, hidden_dim)

        # Learned positional embeddings for all 1024 tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Timestep embedding: sinusoidal -> MLP
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(hidden_dim // 3),  # 128-dim
            nn.Linear(hidden_dim // 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

        # Final layer: adaLN + linear projection back to patch values
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),  # scale, shift
        )
        nn.init.zeros_(self.final_modulation[1].weight)
        nn.init.zeros_(self.final_modulation[1].bias)
        self.final_linear = nn.Linear(hidden_dim, self.patch_dim)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def patchify(self, x):
        """Convert pixel video to patch tokens.

        (B, C, T, H, W) -> (B, num_tokens, patch_dim)
        """
        B, C, T, H, W = x.shape
        p = self.patch_size
        h_patches = H // p  # 8
        w_patches = W // p  # 8

        # Reshape to extract patches
        # (B, C, T, h_patches, p, w_patches, p)
        x = x.reshape(B, C, T, h_patches, p, w_patches, p)
        # (B, T, h_patches, w_patches, C, p, p)
        x = x.permute(0, 2, 3, 5, 1, 4, 6)
        # (B, T * h_patches * w_patches, C * p * p)
        x = x.reshape(B, T * h_patches * w_patches, C * p * p)
        return x

    def unpatchify(self, x):
        """Convert patch tokens back to pixel video.

        (B, num_tokens, patch_dim) -> (B, C, T, H, W)
        """
        B = x.shape[0]
        C = self.in_channels
        T = self.num_frames
        p = self.patch_size
        h_patches = self.image_size // p
        w_patches = self.image_size // p

        # (B, T, h_patches, w_patches, C, p, p)
        x = x.reshape(B, T, h_patches, w_patches, C, p, p)
        # (B, C, T, h_patches, p, w_patches, p)
        x = x.permute(0, 4, 1, 2, 5, 3, 6)
        # (B, C, T, H, W)
        x = x.reshape(B, C, T, h_patches * p, w_patches * p)
        return x

    def forward(self, x, t):
        """
        x: (B, C, T, H, W) — noisy pixel video
        t: (B,) — diffusion timestep indices
        Returns: (B, C, T, H, W) — predicted noise
        """
        # Patchify and project
        tokens = self.patchify(x)           # (B, 1024, 48)
        tokens = self.patch_embed(tokens)    # (B, 1024, 384)

        # Add positional embeddings
        tokens = tokens + self.pos_embed

        # Timestep embedding
        t_emb = self.time_embed(t)  # (B, 384)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, t_emb)

        # Final layer with adaLN
        mod = self.final_modulation(t_emb).unsqueeze(1)  # (B, 1, 2*D)
        scale, shift = mod.chunk(2, dim=-1)
        tokens = self.final_norm(tokens) * (1 + scale) + shift
        tokens = self.final_linear(tokens)  # (B, 1024, 48)

        # Unpatchify back to pixel video shape
        output = self.unpatchify(tokens)  # (B, C, T, H, W)
        return output
