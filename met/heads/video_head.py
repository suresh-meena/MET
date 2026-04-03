"""
met/heads/video_head.py
=======================
Thin output head: maps converged video state x_v(T) → visual latent predictions.
Symmetric to AudioHead; used for audio-to-video generation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class VideoHead(nn.Module):
    """
    Maps x_v(T): (B, L, D) → visual_pred: (B, L, D_vis).

    Args:
        D:      state width
        D_vis:  visual latent dimension (e.g., 768 for CLIP-ViT-B patches)
        D_hid:  hidden dimension (default same as D)
    """

    def __init__(self, D: int, D_vis: int = 768, D_hid: int | None = None) -> None:
        super().__init__()
        D_hid = D_hid or D
        self.proj = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D_hid),
            nn.GELU(),
            nn.Linear(D_hid, D_vis),
        )

    def forward(self, x_v: Tensor) -> Tensor:
        """
        Args:
            x_v: (B, L, D)  converged video state
        Returns:
            vis_pred: (B, L, D_vis)
        """
        return self.proj(x_v)
