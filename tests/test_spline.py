"""
tests/test_spline.py
====================
Robust unit tests for the real BSplineCache implementation.
"""

from __future__ import annotations

import torch

from met.core.spline import BSplineCache


def _make_cache() -> BSplineCache:
    return BSplineCache(L=16, N=5, M=7, lam=1e-3, degree=3)


class TestBSplineCache:
    def test_registered_buffers_exist_and_are_frozen(self):
        cache = _make_cache()
        for name in ["F", "R", "F_quad", "t_quad", "w_quad", "A", "log_w_quad"]:
            tensor = getattr(cache, name)
            assert isinstance(tensor, torch.Tensor), f"{name} is not a Tensor"
            assert tensor.requires_grad is False, f"{name} should be a frozen buffer"

    def test_basis_partition_of_unity_and_nonnegative(self):
        cache = _make_cache()
        F = cache.F
        assert F.shape == (16, 5)
        row_sums = F.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
        assert (F >= -1e-7).all(), "B-spline basis must be non-negative"

    def test_quadrature_weights_and_nodes(self):
        cache = _make_cache()
        assert cache.w_quad.shape == (7,)
        assert torch.all(cache.w_quad > 0)
        assert abs(cache.w_quad.sum().item() - 1.0) < 1e-6
        assert torch.all(cache.t_quad >= 0) and torch.all(cache.t_quad <= 1)
        assert torch.allclose(cache.log_w_quad.exp(), cache.w_quad, atol=1e-7)

    def test_fused_matrix_matches_definition(self):
        cache = _make_cache()
        expected_A = cache.F_quad @ cache.R
        assert torch.allclose(cache.A, expected_A, atol=1e-6), "A must equal F_quad @ R"

    def test_encode_shapes_and_numerical_consistency(self):
        cache = _make_cache()
        B, H, L, D_k = 2, 3, 16, 4
        K = torch.randn(B, H, L, D_k)

        C_bar, K_quad = cache.encode(K)
        assert C_bar.shape == (B, H, 5, D_k)
        assert K_quad.shape == (B, H, 7, D_k)

        # Two equivalent forms:
        # 1) two-step: C_bar = R @ K then K_quad = F_quad @ C_bar
        # 2) fused:    K_quad = A @ K
        K_quad_from_two_step = torch.einsum("mn,bhnd->bhmd", cache.F_quad, C_bar)
        K_quad_from_fused = torch.einsum("ml,bhld->bhmd", cache.A, K)

        assert torch.allclose(K_quad, K_quad_from_two_step, atol=1e-6)
        assert torch.allclose(K_quad, K_quad_from_fused, atol=1e-6)

    def test_encode_is_differentiable_wrt_keys(self):
        cache = _make_cache()
        K = torch.randn(1, 2, 16, 3, requires_grad=True)
        _, K_quad = cache.encode(K)
        loss = (K_quad ** 2).mean()
        loss.backward()

        assert K.grad is not None
        assert torch.isfinite(K.grad).all(), "Gradient through encode() should be finite"

