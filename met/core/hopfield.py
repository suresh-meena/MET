"""
met/core/hopfield.py
====================
Modality-specific Hopfield memory bank with cross-modal context blending.

Architecture (§3.2):
    E_m^HN(x^v, x^a) = E_v^HN + E_a^HN

For modality m, with complementary modality m':

    ḡ_{m',ℓ}    = avg_{j∈N(ℓ)} g_{m',j}              (temporal smoothing)
    c_{m'→m,ℓ}  = W_{m'→m} ḡ_{m',ℓ}                 (cross-modal projection)
    s_{ℓμ}^m    = λ <c_{m'→m,ℓ}, ξ_{m,μ}>
                + (1−λ) <g_{m,ℓ}, ξ_{m,μ}>            (blended score)
    E_m^HN      = −(1/β_HN) Σ_ℓ log Σ_μ exp(β_HN s_{ℓμ}^m)

Design decisions:
  - lambda_cross is nn.Parameter (learnable), clamped to [0,1] in forward.
    Init small (0.05); anneal upward only if retrieval quality improves (Exp 1.2).
  - Prototypes Xi are L2-normalized AT USE TIME (not stored normalized).
    This prevents norm collapse during training.
  - Temporal smoothing uses F.avg_pool1d with count_include_pad=False
    to avoid dilution at sequence boundaries.

Gradient note (Appendix A):
    When both modalities are optimized jointly, gradient flows through
    c_{m'→m,ℓ} into the complementary modality. The zero-gradient
    statement (∂E_m^HN/∂g_{m,ℓ} has no cross-modal term) only holds
    under conditional or alternating updates — see §3.2 of the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HopfieldMemoryBank(nn.Module):
    """
    Per-modality Hopfield memory with projected cross-modal context.

    Args:
        D:            state width of the CURRENT modality
        D_cross:      state width of the COMPLEMENTARY modality
        K:            number of stored prototype patterns
        beta_HN:      Hopfield inverse temperature (sharper = stronger attractor)
        lambda_cross: initial cross-modal blending weight (0 = self-only)
        window:       temporal smoothing half-window size w (avg over 2w+1 tokens)
    """

    def __init__(
        self,
        D: int,
        D_cross: int,
        K: int,
        beta_HN: float = 1.0,
        lambda_cross: float = 0.05,
        window: int = 3,
    ) -> None:
        super().__init__()
        self.K = K
        self.beta_HN = beta_HN
        self.window = window

        # lambda_cross: unconstrained parameter, clamped in forward
        # Start small per paper recommendation; Exp 1.2 determines safe range
        self.lambda_cross = nn.Parameter(torch.tensor(float(lambda_cross)))

        # Prototype matrix: (K, D)
        # Normalized at use time via F.normalize — not stored normalized
        self.Xi = nn.Parameter(F.normalize(torch.randn(K, D), dim=-1))

        # Cross-modal projection: D_cross → D
        self.W_cross = nn.Linear(D_cross, D, bias=False)
        nn.init.normal_(self.W_cross.weight, std=0.02)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _temporal_smooth(self, g: Tensor) -> Tensor:
        """
        Causal-symmetric local average over temporal window w.

        ḡ_{m',ℓ} = (1/|N(ℓ)|) Σ_{j∈N(ℓ)} g_{m',j}
        where N(ℓ) = {j : |j−ℓ| ≤ ⌊w/2⌋}

        Implemented via F.avg_pool1d for GPU efficiency. The option
        count_include_pad=False ensures boundary tokens are averaged
        over their actual neighbors, not padded zeros.

        Args:
            g: (B, L, D_cross)
        Returns:
            ḡ: (B, L, D_cross)
        """
        w = self.window
        # avg_pool1d expects (B, C, L)
        g_t = g.permute(0, 2, 1)                    # (B, D_cross, L)
        smoothed = F.avg_pool1d(
            g_t,
            kernel_size=w,
            stride=1,
            padding=w // 2,
            count_include_pad=False,
        )                                            # (B, D_cross, L)
        return smoothed.permute(0, 2, 1)            # (B, L, D_cross)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, g_m: Tensor, g_cross: Tensor) -> Tensor:
        """
        Compute the Hopfield memory energy for one modality.

        Args:
            g_m:     (B, L, D)        current modality (LayerNorm-normalized)
            g_cross: (B, L, D_cross)  complementary modality (LayerNorm-normalized)

        Returns:
            E_m^HN: scalar Tensor

        Gradient scope:
            Under joint optimization (both modalities updating simultaneously),
            gradient flows through g_cross → W_cross → λ_cross path into the
            complementary modality. This is EXPECTED and correct.
            The zero-gradient claim requires conditional or alternating updates.
        """
        # Clamp lambda to valid range [0, 1]
        lam = self.lambda_cross.clamp(0.0, 1.0)

        # Cross-modal context: smooth + project
        g_bar = self._temporal_smooth(g_cross)        # (B, L, D_cross)
        c = self.W_cross(g_bar)                       # (B, L, D)

        # Normalize prototypes at use time (not stored normalized)
        Xi_n = F.normalize(self.Xi, dim=-1)           # (K, D)

        # Blended scores: (B, L, K)
        s_cross = torch.einsum("bld, kd -> blk", c, Xi_n)
        s_self  = torch.einsum("bld, kd -> blk", g_m, Xi_n)
        s = lam * s_cross + (1.0 - lam) * s_self     # (B, L, K)

        # Negative log-partition: -(1/β_HN) Σ_ℓ log Σ_μ exp(β_HN s_{ℓμ})
        log_Z = torch.logsumexp(self.beta_HN * s, dim=-1)  # (B, L)
        return -log_Z.sum() / self.beta_HN

    def extra_repr(self) -> str:
        return (
            f"K={self.K}, beta_HN={self.beta_HN}, "
            f"lambda_cross_init={self.lambda_cross.item():.3f}, window={self.window}"
        )
