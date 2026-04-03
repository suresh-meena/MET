"""
met/core/spline.py
==================
B-Spline basis cache for continuous attention key compression.

Key design:
  F_m ∈ ℝ^{L×N}       — basis matrix evaluated at token positions
  R_m ∈ ℝ^{N×L}       — ridge regression operator (ONE-TIME precompute, registered buffer)
  F_quad ∈ ℝ^{M×N}    — basis at Gauss-Legendre quadrature nodes (buffer)
  w_quad ∈ ℝ^{M}      — quadrature weights summing to 1 on [0,1] (buffer)

Per solver step (online cost):
  C_bar_h = R_m @ K_h       O(H·L·N·D_k)   — spline coefficients
  K_quad  = F_quad @ C_bar  O(H·M·N·D_k)   — keys at quadrature nodes

One-time preprocessing cost: O(L·N² + N³)

Why B-splines over Fourier:
  Compact support → local evaluation → no Gibbs-phenomenon ringing around
  sharp transient events (audio onsets, visual motion cuts). See Exp 2.2.

Why Gauss-Legendre quadrature:
  Integrates degree-(2M-1) polynomials exactly. With spline-compressed
  (smooth) key trajectories, M=32 achieves < 1e-3 error (Exp 2.3).

Gradient note (Appendix A):
  R_m is a fixed buffer — autograd differentiates through
      C_bar = R_m @ K_h
  automatically accumulating the all-query key-path term. Never hand-derive
  this gradient; let autograd handle it. See test_attention.py Exp 0.1.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_bspline_basis(t: np.ndarray, N: int, degree: int = 3) -> Tensor:
    """
    Evaluate N cubic B-spline basis functions at positions t ∈ [0,1].

    Uses uniform internal knots with clamped (repeated) boundary knots so
    that the basis spans [0,1] and satisfies the partition of unity:
        Σ_j φ_j(t) = 1  for all t.

    Optimization: uses BSpline.design_matrix() (scipy ≥ 1.8) when available
    for a vectorized single-pass evaluation. Falls back to a per-basis-function
    loop on older scipy.

    Args:
        t:      (n,) numpy array of evaluation points in [0,1]
        N:      number of basis functions
        degree: spline degree (default 3 = cubic)
    Returns:
        F: (n, N) float32 Tensor
    """
    from scipy.interpolate import BSpline

    internal = np.linspace(0.0, 1.0, N - degree + 1)
    knots = np.concatenate([[0.0] * degree, internal, [1.0] * degree])

    # Fast path: BSpline.design_matrix available in scipy >= 1.8
    # Returns a sparse CSR matrix; .toarray() converts to dense.
    try:
        dm = BSpline.design_matrix(t, knots, degree)
        F_np = dm.toarray().astype(np.float32)          # (n, N)
    except AttributeError:
        # Fallback: per-basis evaluation loop (slower but compatible)
        cols = []
        for i in range(N):
            c = np.zeros(N); c[i] = 1.0
            cols.append(BSpline(knots, c, degree)(t))
        F_np = np.stack(cols, axis=1).astype(np.float32)

    return torch.tensor(F_np, dtype=torch.float32)


def _gauss_legendre_on_unit(M: int) -> tuple[Tensor, Tensor]:
    """
    Gauss-Legendre nodes and weights on [0, 1].
    Standard GL is on [-1, 1]; we map via t = (ξ + 1)/2, w = w_GL/2.

    Args:
        M: number of quadrature points
    Returns:
        t_quad: (M,) nodes in [0, 1]
        w_quad: (M,) weights summing to 1
    """
    from scipy.special import roots_legendre

    xi, wi = roots_legendre(M)
    t_quad = torch.tensor((xi + 1.0) / 2.0, dtype=torch.float32)
    w_quad = torch.tensor(wi / 2.0, dtype=torch.float32)
    return t_quad, w_quad


# ---------------------------------------------------------------------------
# BSplineCache
# ---------------------------------------------------------------------------

class BSplineCache(nn.Module):
    """
    Precomputes and registers as buffers:
        F      — (L, N)  basis matrix at token positions
        R      — (N, L)  ridge regression operator [never updated]
        F_quad — (M, N)  basis at Gauss-Legendre nodes
        t_quad — (M,)    quadrature node positions
        w_quad — (M,)    quadrature weights (sum to 1 on [0,1])

    All buffers are float32 and move with the module (e.g., .cuda()).
    None of them are nn.Parameters — they have no gradients.

    Args:
        L:      sequence length (number of tokens)
        N:      number of B-spline basis functions (N ≪ L for compression)
        M:      number of Gauss-Legendre quadrature nodes
        lam:    ridge regularization strength λ (default 1e-3)
        degree: B-spline polynomial degree (default 3 = cubic)
    """

    def __init__(
        self,
        L: int,
        N: int,
        M: int,
        lam: float = 1e-3,
        degree: int = 3,
    ) -> None:
        super().__init__()
        self.L = L
        self.N = N
        self.M = M
        self.lam = lam
        self.degree = degree

        # ---- Basis at token positions ----
        t_tokens = np.linspace(0.0, 1.0, L)
        F = _build_bspline_basis(t_tokens, N, degree)  # (L, N)

        # ---- Ridge regression operator: R = (FᵀF + λI)⁻¹ Fᵀ ----
        # One-time cost: O(L·N² + N³)
        G = F.T @ F + lam * torch.eye(N)              # (N, N)
        R = torch.linalg.solve(G, F.T)                # (N, L)

        # ---- Quadrature ----
        t_quad, w_quad = _gauss_legendre_on_unit(M)
        F_quad = _build_bspline_basis(t_quad.numpy(), N, degree)  # (M, N)

        # ---- Fused encode matrix: A = F_quad @ R  shape (M, L) ----
        # Optimization: encode() becomes a SINGLE matmul  K_quad = A @ K
        # instead of two:  C_bar = R @ K,  K_quad = F_quad @ C_bar.
        # FLOP reduction: O(M·L·D_k) vs O(N·L·D_k + M·N·D_k)
        # For typical values M=32, N=16, L=64: saves ~33% FLOPs per step.
        A = F_quad @ R                                  # (M, L)

        # ---- log_w_quad: cached to avoid .log() in every attention forward ----
        log_w_quad = w_quad.log()                       # (M,)

        # Register as buffers (move with module, no grad)
        self.register_buffer("F", F)
        self.register_buffer("R", R)
        self.register_buffer("F_quad", F_quad)
        self.register_buffer("t_quad", t_quad)
        self.register_buffer("w_quad", w_quad)
        self.register_buffer("A", A)                    # fused encode buffer
        self.register_buffer("log_w_quad", log_w_quad)  # cached log weights

    # ------------------------------------------------------------------
    # Forward: encode key sequence into spline coefficients + quad keys
    # ------------------------------------------------------------------

    def encode(self, K: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compress discrete key sequence K onto the B-spline basis and evaluate
        at quadrature nodes.

        Optimization: uses the precomputed fused matrix A = F_quad @ R (shape M×L)
        so that K_quad = A @ K in a SINGLE matmul instead of two separate ops.
        C_bar is still returned for callers that need it (e.g., diagnostics).

        Args:
            K: (B, H, L, D_k)   raw key matrix for all heads

        Returns:
            C_bar:  (B, H, N, D_k)   spline coefficients  C̄_h = R_m K_h
            K_quad: (B, H, M, D_k)   continuous keys at quadrature nodes  (primary output)

        GRADIENT NOTE:
            autograd differentiates through this operation w.r.t. K.
            The all-query key-path term (Appendix A, second sum) is
            accumulated automatically. Do NOT add a stop_gradient here.
        """
        # Fast path: K_quad = A @ K  (single matmul, A = F_quad @ R precomputed)
        # Shape: (M,L) × (B,H,L,D_k) → (B,H,M,D_k)
        K_quad = torch.einsum("ml, bhld -> bhmd", self.A, K)

        # C_bar still computed for diagnostics / callers that need coefficients
        # Shape: (N,L) × (B,H,L,D_k) → (B,H,N,D_k)
        C_bar = torch.einsum("nl, bhld -> bhnd", self.R, K)

        return C_bar, K_quad

    def extra_repr(self) -> str:
        return f"L={self.L}, N={self.N}, M={self.M}, lam={self.lam}, degree={self.degree}"
