"""
tests/test_attention.py
=======================
Unit tests for the real ContinuousAttentionEnergy implementation.
"""

from __future__ import annotations

import torch

from met.core.attention import ContinuousAttentionEnergy
from met.core.spline import BSplineCache
from met.utils.grad_check import finite_difference_grad


def _make_model(
    *,
    L: int = 6,
    N: int = 4,
    M: int = 6,
    D: int = 10,
    D_k: int = 5,
    H: int = 1,
    beta: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> ContinuousAttentionEnergy:
    cache_v = BSplineCache(L=L, N=N, M=M, lam=1e-3).to(dtype=dtype)
    cache_a = BSplineCache(L=L, N=N, M=M, lam=1e-3).to(dtype=dtype)
    model = ContinuousAttentionEnergy(
        D=D, D_k=D_k, H=H, cache_v=cache_v, cache_a=cache_a, beta=beta
    ).to(dtype=dtype)
    return model


def _rand_state(B: int = 1, L: int = 6, D: int = 10, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.randn(B, L, D, dtype=dtype)


class TestAttentionForward:
    def test_forward_returns_three_scalar_terms(self):
        model = _make_model(dtype=torch.float32)
        g_v = _rand_state(dtype=torch.float32)
        g_a = _rand_state(dtype=torch.float32)
        E_iv, E_ia, E_cross = model(g_v, g_a)

        for name, val in [("E_iv", E_iv), ("E_ia", E_ia), ("E_cross", E_cross)]:
            assert val.shape == (), f"{name} must be scalar"
            assert torch.isfinite(val), f"{name} is non-finite"

    def test_cross_term_is_bidirectional_sum(self):
        model = _make_model(dtype=torch.float32)
        g_v = _rand_state(dtype=torch.float32)
        g_a = _rand_state(dtype=torch.float32)

        Q_v, Kq_v = model._project_and_encode(g_v, model.cache_v)
        Q_a, Kq_a = model._project_and_encode(g_a, model.cache_a)
        w_log = model.cache_v.log_w_quad

        E_va = model._log_partition(Q_v, Kq_a, w_log)
        E_av = model._log_partition(Q_a, Kq_v, w_log)
        _, _, E_cross = model(g_v, g_a)
        assert torch.allclose(E_cross, E_va + E_av, atol=1e-6)

    def test_weighted_logsumexp_matches_naive(self):
        model = _make_model(dtype=torch.float64)
        B, L, H, M, D_k = 1, 3, model.H, model.cache_v.M, model.D_k

        Q = torch.randn(B, L, H, D_k, dtype=torch.float64)
        Kq = torch.randn(B, H, M, D_k, dtype=torch.float64)
        w = torch.rand(M, dtype=torch.float64)
        w = w / w.sum()
        w_log = w.log()

        ours = model._log_partition(Q, Kq, w_log)
        scores = torch.einsum("blhd,bhmd->blhm", Q, Kq) * model.beta
        naive = -(torch.log((w * torch.exp(scores)).sum(dim=-1)).sum()) / model.beta
        assert torch.allclose(ours, naive, atol=1e-8)


class TestAttentionGradients:
    def test_intra_modal_autodiff_matches_finite_difference(self):
        model = _make_model(L=4, N=3, M=4, D=6, D_k=3, H=1, dtype=torch.float64)
        g_a_fixed = _rand_state(B=1, L=4, D=6, dtype=torch.float64).detach()

        def fn(g_v: torch.Tensor) -> torch.Tensor:
            E_iv, E_ia, E_cross = model(g_v, g_a_fixed)
            return E_iv + E_ia + E_cross

        g_v = _rand_state(B=1, L=4, D=6, dtype=torch.float64)
        g_v_auto = g_v.clone().requires_grad_(True)
        grad_auto = torch.autograd.grad(fn(g_v_auto), g_v_auto)[0]
        grad_fd = finite_difference_grad(fn, g_v, eps=1e-5)

        rel_err = (grad_auto - grad_fd).norm() / (grad_fd.norm() + 1e-12)
        assert rel_err.item() < 2e-3, f"FD mismatch rel_err={rel_err.item():.2e}"

    def test_cross_modal_autodiff_matches_finite_difference(self):
        model = _make_model(L=4, N=3, M=4, D=6, D_k=3, H=1, dtype=torch.float64)
        g_v_fixed = _rand_state(B=1, L=4, D=6, dtype=torch.float64).detach()

        def fn(g_a: torch.Tensor) -> torch.Tensor:
            Q_v, Kq_v = model._project_and_encode(g_v_fixed, model.cache_v)
            Q_a, Kq_a = model._project_and_encode(g_a, model.cache_a)
            return model._log_partition(Q_v, Kq_a, model.cache_v.log_w_quad)

        g_a = _rand_state(B=1, L=4, D=6, dtype=torch.float64)
        g_a_auto = g_a.clone().requires_grad_(True)
        grad_auto = torch.autograd.grad(fn(g_a_auto), g_a_auto)[0]
        grad_fd = finite_difference_grad(fn, g_a, eps=1e-5)

        rel_err = (grad_auto - grad_fd).norm() / (grad_fd.norm() + 1e-12)
        assert rel_err.item() < 2e-3, f"FD mismatch rel_err={rel_err.item():.2e}"


class TestAttentionDynamics:
    def test_single_step_gradient_descent_lowers_energy(self):
        model = _make_model(dtype=torch.float32)
        g_v = _rand_state(dtype=torch.float32).detach()
        g_a = _rand_state(dtype=torch.float32)
        eta = 1e-3

        g_a_r = g_a.clone().requires_grad_(True)
        E0 = sum(model(g_v, g_a_r))
        grad_a = torch.autograd.grad(E0, g_a_r)[0]

        with torch.no_grad():
            g_a_next = g_a_r - eta * grad_a

        E1 = sum(model(g_v, g_a_next))
        assert E1.item() <= E0.item() + 1e-6

    def test_energy_depends_on_both_modalities(self):
        model = _make_model(dtype=torch.float32)
        g_v = _rand_state(dtype=torch.float32)
        g_a1 = _rand_state(dtype=torch.float32)
        g_a2 = _rand_state(dtype=torch.float32)
        E1 = sum(model(g_v, g_a1)).item()
        E2 = sum(model(g_v, g_a2)).item()
        assert abs(E1 - E2) > 1e-7

