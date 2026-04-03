"""
tests/test_spline.py
====================
Unit tests for BSplineCache (met/core/spline.py).

Tests check:
  - Basis matrix partition of unity (rows sum to ~1)
  - Gram matrix symmetry and positive-definiteness
  - Ridge regression roundtrip: R @ F == I (approx, for small lambda)
  - Quadrature weights sum to 1 (proper integration of constant function)
  - K_quad shape contract: (B, H, M, D_k)
  - encode() is differentiable through K (no detach inside)
"""

import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Minimal stub so tests run without installing the full package.
# Replace with: from met.core.spline import BSplineCache
# ---------------------------------------------------------------------------
def _build_cache(L=32, N=8, M=16, lam=1e-3, degree=3):
    """
    Inline BSplineCache that mirrors the plan implementation.
    Only used until met.core.spline is installed.
    """
    import numpy as np_
    from scipy.interpolate import BSpline
    from scipy.special import roots_legendre

    def build_basis(t_arr, N_, d):
        knots = np_.linspace(0, 1, N_ - d + 1)
        knots = np_.concatenate([[0] * d, knots, [1] * d])
        t_np = t_arr.numpy() if isinstance(t_arr, torch.Tensor) else t_arr
        cols = []
        for i in range(N_):
            c = np_.zeros(N_); c[i] = 1.0
            cols.append(BSpline(knots, c, d)(t_np))
        return torch.tensor(np_.stack(cols, axis=1), dtype=torch.float32)

    t_tok = torch.linspace(0, 1, L)
    F = build_basis(t_tok, N, degree)                     # (L, N)
    G = F.T @ F + lam * torch.eye(N)
    R = torch.linalg.solve(G, F.T)                       # (N, L)

    xi, wi = roots_legendre(M)
    t_quad = torch.tensor((xi + 1) / 2, dtype=torch.float32)
    w_quad = torch.tensor(wi / 2, dtype=torch.float32)
    F_quad = build_basis(t_quad, N, degree)               # (M, N)

    return dict(F=F, R=R, t_quad=t_quad, w_quad=w_quad, F_quad=F_quad)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBSplineBasisMatrix:
    """Structural properties of the basis matrix F."""

    def setup_method(self):
        self.cache = _build_cache(L=32, N=8, M=16)

    def test_shape(self):
        F = self.cache['F']
        assert F.shape == (32, 8), f"Expected (32, 8), got {F.shape}"

    def test_partition_of_unity(self):
        """B-spline rows must sum to 1 (partition of unity property)."""
        F = self.cache['F']
        row_sums = F.sum(dim=1)   # (L,)
        assert torch.allclose(row_sums, torch.ones(32), atol=1e-5), \
            f"Row sums not ~1: max deviation {(row_sums - 1).abs().max().item():.2e}"

    def test_non_negative(self):
        """B-spline basis functions are non-negative."""
        F = self.cache['F']
        assert (F >= -1e-7).all(), "B-spline basis has negative values"

    def test_gram_symmetric(self):
        F = self.cache['F']
        G = F.T @ F
        assert torch.allclose(G, G.T, atol=1e-6), "Gram matrix not symmetric"

    def test_gram_positive_definite(self):
        F = self.cache['F']
        G = F.T @ F + 1e-3 * torch.eye(8)
        eigvals = torch.linalg.eigvalsh(G)
        assert (eigvals > 0).all(), f"Gram + lambda*I not PD; min eigval={eigvals.min().item():.2e}"


class TestRidgeRegression:
    """Properties of the precomputed ridge operator R = (F^T F + lam I)^{-1} F^T."""

    def setup_method(self):
        self.cache = _build_cache(L=32, N=8, M=16, lam=1e-6)

    def test_shape(self):
        R = self.cache['R']
        assert R.shape == (8, 32), f"Expected (8, 32), got {R.shape}"

    def test_roundtrip_recovery(self):
        """F @ R @ F should be close to F when lambda is tiny (near-projection)."""
        F = self.cache['F']
        R = self.cache['R']
        F_reconstructed = F @ (R @ F)   # L x L projection
        # Check a few rows of F (not all — lambda adds regularization bias)
        err = (F_reconstructed - F).abs().mean().item()
        assert err < 0.05, f"Ridge roundtrip mean error too large: {err:.4f}"

    def test_R_F_approx_identity(self):
        """R F should be close to identity N x N for small lambda."""
        F = self.cache['F']
        R = self.cache['R']
        RF = R @ F   # (N, N)
        I = torch.eye(8)
        err = (RF - I).abs().max().item()
        assert err < 0.05, f"R @ F deviates from I by {err:.4f}"


class TestQuadrature:
    """Gauss-Legendre quadrature properties."""

    def setup_method(self):
        self.cache = _build_cache(L=32, N=8, M=16)

    def test_weights_sum_to_one(self):
        """Weights on [0,1] must sum to 1 (integrates constant 1 exactly)."""
        w = self.cache['w_quad']
        assert abs(w.sum().item() - 1.0) < 1e-6, \
            f"Quadrature weights sum to {w.sum().item():.8f}, expected 1.0"

    def test_weights_positive(self):
        """All Gauss-Legendre weights are positive."""
        w = self.cache['w_quad']
        assert (w > 0).all(), "Quadrature has non-positive weights"

    def test_nodes_in_unit_interval(self):
        t = self.cache['t_quad']
        assert (t >= 0).all() and (t <= 1).all(), \
            "Quadrature nodes outside [0,1]"

    def test_integrate_polynomial(self):
        """Gauss-Legendre with M points integrates degree-(2M-1) polynomials exactly.
        Check: integral of t^2 over [0,1] == 1/3."""
        t = self.cache['t_quad']
        w = self.cache['w_quad']
        M = len(t)
        # Exact for M >= 2
        result = (w * t**2).sum().item()
        assert abs(result - 1/3) < 1e-5, \
            f"Integral of t^2 = {result:.6f}, expected {1/3:.6f}"

    def test_F_quad_shape(self):
        F_quad = self.cache['F_quad']
        assert F_quad.shape == (16, 8), f"Expected (16, 8), got {F_quad.shape}"


class TestEncodeContract:
    """Test the encode() output contracts."""

    def setup_method(self):
        self.cache = _build_cache(L=32, N=8, M=16)

    def _encode(self, K):
        """Manual encode as in BSplineCache.encode()."""
        R = self.cache['R']
        F_quad = self.cache['F_quad']
        C_bar  = torch.einsum('nl,bhld->bhnd', R, K)
        K_quad = torch.einsum('mn,bhnd->bhmd', F_quad, C_bar)
        return C_bar, K_quad

    def test_output_shapes(self):
        B, H, L, D_k = 2, 4, 32, 16
        K = torch.randn(B, H, L, D_k)
        C_bar, K_quad = self._encode(K)
        assert C_bar.shape  == (B, H, 8, D_k),  f"C_bar shape {C_bar.shape}"
        assert K_quad.shape == (B, H, 16, D_k), f"K_quad shape {K_quad.shape}"

    def test_differentiable_through_K(self):
        """encode() must be differentiable w.r.t. K for autograd to work."""
        B, H, L, D_k = 1, 2, 32, 8
        K = torch.randn(B, H, L, D_k, requires_grad=True)
        C_bar, K_quad = self._encode(K)
        loss = K_quad.sum()
        loss.backward()
        assert K.grad is not None, "No gradient w.r.t. K — encode() blocked autograd"
        assert not torch.isnan(K.grad).any(), "NaN in K gradient"

    def test_linear_in_K(self):
        """encode() is linear in K: encode(2K) == 2 * encode(K)."""
        B, H, L, D_k = 1, 1, 32, 4
        K = torch.randn(B, H, L, D_k)
        C1, Kq1 = self._encode(K)
        C2, Kq2 = self._encode(2 * K)
        assert torch.allclose(C2, 2 * C1, atol=1e-5), "encode not linear in K (C_bar)"
        assert torch.allclose(Kq2, 2 * Kq1, atol=1e-5), "encode not linear in K (K_quad)"
