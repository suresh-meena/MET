"""
met/tokenizers/video_tokenizer.py
==================================
Video front-end: patch embeddings → shared latent space ℝ^{B×L×D}.

Tiers 0-4: uses frozen ViT (DINOv2 / CLIP-ViT-B) for fast prototyping.
Tier 5: optionally fine-tune ViT end-to-end.

Pipeline:
    raw video (B, T_raw, H, W, 3)
      → ViT patch features (B, T_raw, D_tok)
        → temporal resample to L tokens via F.interpolate
          → linear projection U_v: D_tok → D
            → add learned positional embedding pos_emb_v: (L, D)
              → x_v_0: (B, L, D)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VideoTokenizer(nn.Module):
    """
    Converts raw ViT patch features to shared-width tokens x_v ∈ ℝ^{B×L×D}.

    In Tiers 0-4, ViT features are extracted OFFLINE and passed in directly
    (frozen backbone). This class handles the resample + project + pos_emb step.

    Args:
        D_tok:  ViT output feature dimension (e.g., 768 for CLIP-ViT-B)
        D:      shared state width (must match METConfig.D)
        L:      target sequence length (token budget per clip)
    """

    def __init__(self, D_tok: int = 768, D: int = 256, L: int = 64) -> None:
        super().__init__()
        self.D_tok = D_tok
        self.D = D
        self.L = L

        # Linear projection: D_tok → D
        self.U_v = nn.Linear(D_tok, D, bias=False)
        nn.init.normal_(self.U_v.weight, std=0.02)

        # Learned positional embedding: (1, L, D) — broadcast over batch
        self.pos_emb = nn.Parameter(torch.zeros(1, L, D))
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, z_v: Tensor) -> Tensor:
        """
        Args:
            z_v: (B, T_raw, D_tok)   ViT patch features at original frame rate

        Returns:
            x_v: (B, L, D)           projected + position-encoded tokens

        Temporal resampling:
            If T_raw != L, linearly interpolate along the time axis.
            F.interpolate with mode='linear' operates on (B, C, T) format.
        """
        B, T_raw, D_tok = z_v.shape

        # Resample to L tokens if needed
        if T_raw != self.L:
            z_v = F.interpolate(
                z_v.permute(0, 2, 1),   # (B, D_tok, T_raw)
                size=self.L,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)           # (B, L, D_tok)

        # Project + add positional embedding
        x_v = self.U_v(z_v) + self.pos_emb  # (B, L, D)
        return x_v

    def extra_repr(self) -> str:
        return f"D_tok={self.D_tok}, D={self.D}, L={self.L}"
