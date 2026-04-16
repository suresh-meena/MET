"""
tests/test_hopfield.py
======================
Unit tests for the real HopfieldMemoryBank implementation.
"""

from __future__ import annotations

import copy

import torch

from met.core.hopfield import HopfieldMemoryBank


def _make_bank(
    *,
    D: int = 16,
    K: int = 5,
    beta_HN: float = 4.0,
    lambda_cross: float = 0.2,
    window: int = 3,
) -> HopfieldMemoryBank:
    return HopfieldMemoryBank(
        D=D,
        D_cross=D,
        K=K,
        beta_HN=beta_HN,
        lambda_cross=lambda_cross,
        window=window,
    )


class TestHopfieldForward:
    def test_forward_scalar_and_finite(self):
        bank = _make_bank()
        g_m = torch.randn(2, 8, 16)
        g_cross = torch.randn(2, 8, 16)
        E = bank(g_m, g_cross)
        assert E.shape == ()
        assert torch.isfinite(E)

    def test_temporal_smoothing_constant_sequence_is_identity(self):
        bank = _make_bank(window=5)
        g = torch.ones(1, 12, 16) * 2.5
        smoothed = bank._temporal_smooth(g)
        assert torch.allclose(smoothed, g, atol=1e-6)

    def test_temporal_smoothing_reduces_spike(self):
        bank = _make_bank(window=5)
        g = torch.zeros(1, 12, 16)
        g[:, 6, :] = 10.0
        smoothed = bank._temporal_smooth(g)
        assert smoothed.abs().max().item() < g.abs().max().item()


class TestHopfieldSemantics:
    def test_lambda_cross_clamp_matches_boundary_values(self):
        g_m = torch.randn(1, 8, 16)
        g_cross = torch.randn(1, 8, 16)

        base = _make_bank(lambda_cross=0.3)
        lam0 = copy.deepcopy(base)
        lam1 = copy.deepcopy(base)
        low = copy.deepcopy(base)
        high = copy.deepcopy(base)

        with torch.no_grad():
            lam0.lambda_cross.fill_(0.0)
            lam1.lambda_cross.fill_(1.0)
            low.lambda_cross.fill_(-5.0)
            high.lambda_cross.fill_(5.0)

        E_low = low(g_m, g_cross)
        E_lam0 = lam0(g_m, g_cross)
        E_high = high(g_m, g_cross)
        E_lam1 = lam1(g_m, g_cross)

        assert torch.allclose(E_low, E_lam0, atol=1e-6), "lambda_cross<0 must clamp to 0"
        assert torch.allclose(E_high, E_lam1, atol=1e-6), "lambda_cross>1 must clamp to 1"

    def test_lambda_zero_removes_cross_modal_dependence(self):
        bank = _make_bank(lambda_cross=0.0)
        g_m = torch.randn(1, 8, 16)
        g_cross_1 = torch.randn(1, 8, 16)
        g_cross_2 = torch.randn(1, 8, 16)

        E1 = bank(g_m, g_cross_1)
        E2 = bank(g_m, g_cross_2)
        assert torch.allclose(E1, E2, atol=1e-6)

    def test_lambda_one_uses_cross_modal_context(self):
        bank = _make_bank(lambda_cross=1.0)
        g_m = torch.randn(1, 8, 16)
        g_cross_1 = torch.randn(1, 8, 16)
        g_cross_2 = torch.randn(1, 8, 16)

        E1 = bank(g_m, g_cross_1)
        E2 = bank(g_m, g_cross_2)
        assert not torch.allclose(E1, E2, atol=1e-5)

    def test_prototypes_are_normalized_at_use_time(self):
        bank = _make_bank(lambda_cross=0.0)
        g_m = torch.randn(1, 8, 16)
        g_cross = torch.randn(1, 8, 16)

        E_before = bank(g_m, g_cross).item()
        with torch.no_grad():
            bank.Xi.mul_(7.0)
        E_after = bank(g_m, g_cross).item()
        assert abs(E_before - E_after) < 1e-5


class TestHopfieldGradients:
    def test_gradients_flow_to_state_and_parameters(self):
        bank = _make_bank(lambda_cross=0.3)
        g_m = torch.randn(1, 8, 16, requires_grad=True)
        g_cross = torch.randn(1, 8, 16, requires_grad=True)

        E = bank(g_m, g_cross)
        E.backward()

        assert g_m.grad is not None and torch.isfinite(g_m.grad).all()
        assert g_cross.grad is not None and torch.isfinite(g_cross.grad).all()
        assert bank.Xi.grad is not None and torch.isfinite(bank.Xi.grad).all()
        assert bank.W_cross.weight.grad is not None and torch.isfinite(bank.W_cross.weight.grad).all()

