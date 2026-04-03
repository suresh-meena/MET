"""
met/core/energy.py
==================
Top-level MET energy function.

    E(x^v, x^a) = E_v^intra + E_a^intra + E_{v↔a}^cross + E_v^HN + E_a^HN

All terms are scalars. The total is a scalar.

Key engineering invariant (§7, Implementation Blueprint):
    freeze_v / freeze_a use .detach() on the STATE, not torch.no_grad().
    This stops state gradients (conditional generation) while STILL
    allowing parameter gradients to flow for training.
    Using no_grad() would block parameter gradients — wrong.

Incremental build order (§7):
    Phase A: attention only, J_gen, BPTT
    Phase B: + Hopfield banks
    Phase C: + JEPA + ranking
    Phase D: switch to EqProp
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from met.core.layernorm import TokenwiseLayerNorm
from met.core.spline import BSplineCache
from met.core.attention import ContinuousAttentionEnergy
from met.core.hopfield import HopfieldMemoryBank


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class METConfig:
    """Hyperparameters for the MET energy model."""
    # Shared
    L: int = 64          # sequence length (tokens per clip)
    D: int = 256         # state width (shared for both modalities)

    # Attention
    D_k: int = 64        # key/query dim per head
    H: int = 4           # number of heads
    beta: float = 1.0    # attention inverse temperature

    # Spline (can differ per modality; default shared)
    N_v: int = 16        # video spline basis size
    N_a: int = 16        # audio spline basis size
    M_v: int = 32        # video quadrature nodes
    M_a: int = 32        # audio quadrature nodes
    lam_spline: float = 1e-3  # ridge regularization

    # Hopfield
    K_v: int = 64        # video prototypes
    K_a: int = 64        # audio prototypes
    beta_HN: float = 1.0 # Hopfield inverse temperature
    lambda_cross: float = 0.05  # cross-modal blending (init small)
    window: int = 3      # temporal smoothing window


# ---------------------------------------------------------------------------
# METEnergy
# ---------------------------------------------------------------------------

class METEnergy(nn.Module):
    """
    Scalar energy function for joint audio-visual state (x^v, x^a).

    The energy is NOT a standard transformer forward pass. It is a scalar
    that inference-time optimization descends along. Modalities interact
    through the cross-modal attention term and the blended Hopfield scores,
    but their state spaces remain distinct.

    Args:
        cfg: METConfig dataclass with all hyperparameters
    """

    def __init__(self, cfg: METConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.D

        # Per-token LayerNorm for each modality
        self.ln_v = TokenwiseLayerNorm(D)
        self.ln_a = TokenwiseLayerNorm(D)

        # Spline caches (modality-specific, all buffers)
        cache_v = BSplineCache(cfg.L, cfg.N_v, cfg.M_v, cfg.lam_spline)
        cache_a = BSplineCache(cfg.L, cfg.N_a, cfg.M_a, cfg.lam_spline)

        # Shared continuous-attention block
        self.attention = ContinuousAttentionEnergy(
            D, cfg.D_k, cfg.H, cache_v, cache_a, cfg.beta
        )

        # Per-modality Hopfield banks
        # D_cross == D since we use a shared state width
        self.hopfield_v = HopfieldMemoryBank(
            D, D, cfg.K_v, cfg.beta_HN, cfg.lambda_cross, cfg.window
        )
        self.hopfield_a = HopfieldMemoryBank(
            D, D, cfg.K_a, cfg.beta_HN, cfg.lambda_cross, cfg.window
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_v: Tensor,
        x_a: Tensor,
        freeze_v: bool = False,
        freeze_a: bool = False,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute total energy and per-component breakdown.

        Args:
            x_v:      (B, L, D)  video state (pre-normalized)
            x_a:      (B, L, D)  audio state (pre-normalized)
            freeze_v: if True, stop gradient w.r.t. x_v (audio-to-video: freeze audio, optimize video)
            freeze_a: if True, stop gradient w.r.t. x_a (video-to-audio Foley: freeze video)

        Returns:
            E_total:    scalar Tensor   (differentiable w.r.t. unfrozen states + all parameters)
            components: dict of float   (for logging; not part of computation graph)

        CRITICAL — freeze semantics:
            .detach() on the STATE stops state gradients only.
            Parameter gradients still flow (needed for training).
            Never replace with torch.no_grad() here.
        """
        if freeze_v:
            x_v = x_v.detach()   # stop grad to x_v state; params still get grad
        if freeze_a:
            x_a = x_a.detach()   # stop grad to x_a state; params still get grad

        # Per-token LayerNorm (Jacobian has null space — see §3.3)
        g_v = self.ln_v(x_v)  # (B, L, D)
        g_a = self.ln_a(x_a)  # (B, L, D)

        # Continuous-attention block (intra + cross)
        E_iv, E_ia, E_cross = self.attention(g_v, g_a)

        # Hopfield memory banks
        E_hv = self.hopfield_v(g_v, g_a)  # video bank, audio as cross context
        E_ha = self.hopfield_a(g_a, g_v)  # audio bank, video as cross context

        E_total = E_iv + E_ia + E_cross + E_hv + E_ha

        components = {
            "E_intra_v": E_iv.item(),
            "E_intra_a": E_ia.item(),
            "E_cross":   E_cross.item(),
            "E_HN_v":    E_hv.item(),
            "E_HN_a":    E_ha.item(),
            "E_total":   E_total.item(),
        }
        return E_total, components

    # ------------------------------------------------------------------
    # Convenience: attention-only energy (Phase A baseline)
    # ------------------------------------------------------------------

    def forward_attention_only(
        self,
        x_v: Tensor,
        x_a: Tensor,
        freeze_v: bool = False,
        freeze_a: bool = False,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute energy without Hopfield banks.
        Used during Phase A (BPTT baseline) before banks are added.
        """
        if freeze_v: x_v = x_v.detach()
        if freeze_a: x_a = x_a.detach()

        g_v = self.ln_v(x_v)
        g_a = self.ln_a(x_a)
        E_iv, E_ia, E_cross = self.attention(g_v, g_a)
        E_total = E_iv + E_ia + E_cross
        return E_total, {"E_intra_v": E_iv.item(), "E_intra_a": E_ia.item(),
                         "E_cross": E_cross.item(), "E_total": E_total.item()}

    def extra_repr(self) -> str:
        return (
            f"L={self.cfg.L}, D={self.cfg.D}, H={self.cfg.H}, "
            f"K_v={self.cfg.K_v}, K_a={self.cfg.K_a}"
        )
