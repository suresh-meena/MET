"""
met/core/attention.py
=====================
Continuous-attention energy for intra-modal and cross-modal interaction.

Architecture overview (§3.1):
    E^ATT(x^v, x^a) = E_v^intra + E_a^intra + E_{v↔a}^cross

Each term is a negative log-partition sum over a continuous key trajectory,
evaluated at M Gauss-Legendre quadrature nodes:

    E_m^intra = -(1/β) Σ_h Σ_ℓ log Σ_r ω_r exp(β <Q_{h,ℓ}^m, K̄_h(m,t_r)>)

    E_cross   = E_{v→a} + E_{a→v}
              = -(1/β) Σ_h Σ_ℓ [log Σ_r ω_r exp(β <Q_{h,ℓ}^v, K̄_h(a,t_r)>)
                               +  log Σ_r ω_r exp(β <Q_{h,ℓ}^a, K̄_h(v,t_r)>)]

Gradient notes (Appendix A):
  - Query path: ∂E/∂Q_{h,ℓ} = -E_{t~p}[K̄_h(t)]  (softmax-weighted expectation)
  - Key path:   ∂E/∂K_{h,ℓ} accumulates contributions from ALL query positions
                through the shared spline regression R_m @ K_h.
    → Use autograd. Never hand-code the second sum. See test_attention.py.

Cross-modal semantics:
  - v queries a: Q_v dot K_quad_a  (video tokens attend to audio keys)
  - a queries v: Q_a dot K_quad_v  (symmetric)
  - Modalities interact only through shared head-dim scores, NOT merged states.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from met.core.spline import BSplineCache


class ContinuousAttentionEnergy(nn.Module):
    """
    Continuous-attention energy block.

    Args:
        D:        state width (shared for both modalities)
        D_k:      key/query dimension per head
        H:        number of attention heads
        cache_v:  BSplineCache for visual modality
        cache_a:  BSplineCache for audio modality
        beta:     inverse temperature β (scales scores before softmax)
    """

    def __init__(
        self,
        D: int,
        D_k: int,
        H: int,
        cache_v: BSplineCache,
        cache_a: BSplineCache,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.H = H
        self.D_k = D_k
        self.beta = beta
        self.cache_v = cache_v
        self.cache_a = cache_a

        # Shared Q and K projections across both modalities.
        # Modality separation is maintained by separate spline caches, not
        # separate projection matrices.
        self.W_Q = nn.Linear(D, H * D_k, bias=False)
        self.W_K = nn.Linear(D, H * D_k, bias=False)

        # Weight init: small values keep initial energy landscape flat
        nn.init.normal_(self.W_Q.weight, std=0.02)
        nn.init.normal_(self.W_K.weight, std=0.02)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project_and_encode(
        self, g: Tensor, cache: BSplineCache
    ) -> tuple[Tensor, Tensor]:
        """
        Project g into Q and K, then compress K via spline.

        Args:
            g:     (B, L, D)         normalized token states
            cache: BSplineCache      modality-specific spline cache

        Returns:
            Q:      (B, L, H, D_k)   query vectors
            K_quad: (B, H, M, D_k)   spline-compressed keys at quad nodes
        """
        B, L, D = g.shape

        Q = self.W_Q(g).view(B, L, self.H, self.D_k)   # (B, L, H, D_k)
        K = self.W_K(g).view(B, L, self.H, self.D_k)   # (B, L, H, D_k)

        # Encode: (B, H, L, D_k) → (B, H, N, D_k), (B, H, M, D_k)
        _, K_quad = cache.encode(K.permute(0, 2, 1, 3))

        return Q, K_quad

    def _log_partition(
        self, Q: Tensor, K_quad: Tensor, w_log: Tensor
    ) -> Tensor:
        """
        Compute negative log-partition energy for one attention direction.

        Math:
            -(1/β) Σ_h Σ_ℓ log Σ_r ω_r exp(β <Q_{h,ℓ}, K̄_h(t_r)>)

        Numerically stable via:
            log Σ_r ω_r exp(s_r) = logsumexp(s_r + log(ω_r))
            (valid because Gauss-Legendre weights ω_r > 0 always)

        Optimization: w_log is precomputed and passed in.

        Args:
            Q:      (B, L, H, D_k)
            K_quad: (B, H, M, D_k)
            w_log:  (M,)              log quadrature weights

        Returns:
            E: scalar Tensor
        """
        # Scores: (B, L, H, M)
        scores = torch.einsum("blhd, bhmd -> blhm", Q, K_quad) * self.beta

        # log Σ_r ω_r exp(β s_r)  via logsumexp trick
        log_Z = torch.logsumexp(scores + w_log, dim=-1)  # (B, L, H)

        return -log_Z.sum() / self.beta

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, g_v: Tensor, g_a: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute all three attention energy terms.

        Args:
            g_v: (B, L, D)  LayerNorm-normalized video states
            g_a: (B, L, D)  LayerNorm-normalized audio states

        Returns:
            E_intra_v: scalar  (video self-attention energy)
            E_intra_a: scalar  (audio self-attention energy)
            E_cross:   scalar  (bidirectional cross-modal energy)
        """
        Q_v, Kq_v = self._project_and_encode(g_v, self.cache_v)
        Q_a, Kq_a = self._project_and_encode(g_a, self.cache_a)
        
        # Optimization: use precomputed log weights
        w_log = self.cache_v.log_w_quad

        E_intra_v = self._log_partition(Q_v, Kq_v, w_log)
        E_intra_a = self._log_partition(Q_a, Kq_a, w_log)

        # Bidirectional cross: v→a AND a→v
        E_cross = (
            self._log_partition(Q_v, Kq_a, w_log)  # v queries a's keys
            + self._log_partition(Q_a, Kq_v, w_log)  # a queries v's keys
        )

        return E_intra_v, E_intra_a, E_cross

    def extra_repr(self) -> str:
        return f"H={self.H}, D_k={self.D_k}, beta={self.beta}"
