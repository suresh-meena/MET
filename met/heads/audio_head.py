"""
met/heads/audio_head.py
=======================
Thin output head: maps converged audio state x_a(T) → mel-spectrogram predictions.

This head is OUTSIDE the energy. Its role is to connect the energy's converged
latent state to a measurable reconstruction target. It does not affect inference
dynamics; training gradients flow from J_mel through this head into x_a(T),
and through the (BPTT) solver trajectory into energy parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class AudioHead(nn.Module):
    """
    Maps x_a(T): (B, L, D) → mel_pred: (B, L, n_mels).

    A 2-layer MLP with LayerNorm and GELU activation.
    Thin by design — the energy does the representation work.

    Args:
        D:      state width (must match METEnergy.cfg.D)
        n_mels: number of mel filterbank bins (default 128)
        D_hid:  hidden dimension (default same as D)
    """

    def __init__(self, D: int, n_mels: int = 128, D_hid: int | None = None) -> None:
        super().__init__()
        D_hid = D_hid or D
        self.proj = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D_hid),
            nn.GELU(),
            nn.Linear(D_hid, n_mels),
        )

    def forward(self, x_a: Tensor) -> Tensor:
        """
        Args:
            x_a: (B, L, D)  converged audio state from solver
        Returns:
            mel_pred: (B, L, n_mels)
        """
        return self.proj(x_a)
