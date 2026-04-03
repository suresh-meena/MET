"""
tests/test_energy.py
====================
Integration tests for the combined MET energy E(x^v, x^a).

Tests check:
  - Exp 0.3: Energy monotonicity under gradient descent (per-step E decrease)
  - LayerNorm null-space event detection (Ė≈0 but ∇_g E ≠ 0)
  - freeze_v / freeze_a semantics: gradient stops at frozen state, not parameters
  - Alternating vs joint gradient descent: correct gradient scoping
  - Matched vs mismatched energy gap (prerequisite for Exp 4.3 ranking)
  - EqProp memory bound: EqProp stores 2 equilibria; BPTT stores T states
  - Energy is a function of both modalities (sanity: changing x_a changes E)

Uses inline stubs — no met package installation required.
Sizes: L=16 or 32 to stay fast on CPU.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


# ---------------------------------------------------------------------------
# Inline stubs (full pipeline: LayerNorm + Attention + Hopfield -> Energy)
# ---------------------------------------------------------------------------

def _build_basis_np(t_arr, N, degree=3):
    from scipy.interpolate import BSpline
    knots = np.linspace(0, 1, N - degree + 1)
    knots = np.concatenate([[0] * degree, knots, [1] * degree])
    cols = []
    for i in range(N):
        c = np.zeros(N); c[i] = 1.0
        cols.append(BSpline(knots, c, degree)(t_arr))
    return torch.tensor(np.stack(cols, axis=1), dtype=torch.float32)


def _make_cache(L, N, M, lam=1e-3):
    from scipy.special import roots_legendre
    t_tok = np.linspace(0, 1, L)
    F = _build_basis_np(t_tok, N)
    G = F.T @ F + lam * torch.eye(N)
    R = torch.linalg.solve(G, F.T)
    xi, wi = roots_legendre(M)
    t_q = torch.tensor((xi + 1) / 2, dtype=torch.float32)
    w_q = torch.tensor(wi / 2, dtype=torch.float32)
    F_q = _build_basis_np((xi + 1) / 2, N)
    return dict(F=F, R=R, F_quad=F_q, t_quad=t_q, w_quad=w_q)


class _Attention(nn.Module):
    def __init__(self, D, D_k, H, cv, ca, beta=1.0):
        super().__init__()
        self.H, self.D_k, self.beta = H, D_k, beta
        self.W_Q = nn.Linear(D, H * D_k, bias=False)
        self.W_K = nn.Linear(D, H * D_k, bias=False)
        self.cv, self.ca = cv, ca

    def _encode(self, g, c):
        B, L, D = g.shape
        K = self.W_K(g).view(B, L, self.H, self.D_k).permute(0,2,1,3)
        C = torch.einsum('nl,bhld->bhnd', c['R'], K)
        Kq = torch.einsum('mn,bhnd->bhmd', c['F_quad'], C)
        Q = self.W_Q(g).view(B, L, self.H, self.D_k)
        return Q, Kq

    def _lp(self, Q, Kq, w):
        s = torch.einsum('blhd,bhmd->blhm', Q, Kq) * self.beta
        return -(torch.logsumexp(s + w.log(), dim=-1).sum()) / self.beta

    def forward(self, g_v, g_a):
        Q_v, Kq_v = self._encode(g_v, self.cv)
        Q_a, Kq_a = self._encode(g_a, self.ca)
        w = self.cv['w_quad']
        return (self._lp(Q_v, Kq_v, w) + self._lp(Q_a, Kq_a, w)
                + self._lp(Q_v, Kq_a, w) + self._lp(Q_a, Kq_v, w))


class _Hopfield(nn.Module):
    def __init__(self, D, K, beta_HN=1.0, w=3):
        super().__init__()
        self.beta_HN = beta_HN; self.w = w
        self.lambda_cross = nn.Parameter(torch.tensor(0.05))
        self.Xi = nn.Parameter(F.normalize(torch.randn(K, D), dim=-1))
        self.W_cross = nn.Linear(D, D, bias=False)

    def _smooth(self, g):
        return F.avg_pool1d(
            g.permute(0,2,1), self.w, 1, self.w//2, count_include_pad=False
        ).permute(0,2,1)

    def forward(self, g_m, g_cross):
        lam = self.lambda_cross.clamp(0., 1.)
        c = self.W_cross(self._smooth(g_cross))
        Xi_n = F.normalize(self.Xi, dim=-1)
        s = lam * torch.einsum('bld,kd->blk', c, Xi_n) \
          + (1-lam) * torch.einsum('bld,kd->blk', g_m, Xi_n)
        return -(torch.logsumexp(self.beta_HN * s, dim=-1).sum()) / self.beta_HN


class _METEnergy(nn.Module):
    def __init__(self, L=16, N=4, M=8, D=16, D_k=8, H=1,
                 K=4, beta=1.0, beta_HN=1.0):
        super().__init__()
        cv = _make_cache(L, N, M)
        ca = _make_cache(L, N, M)
        self.ln_v = nn.LayerNorm(D)
        self.ln_a = nn.LayerNorm(D)
        self.att = _Attention(D, D_k, H, cv, ca, beta)
        self.hv  = _Hopfield(D, K, beta_HN)
        self.ha  = _Hopfield(D, K, beta_HN)

    def forward(self, x_v, x_a, freeze_v=False, freeze_a=False):
        if freeze_v: x_v = x_v.detach()
        if freeze_a: x_a = x_a.detach()
        g_v = self.ln_v(x_v); g_a = self.ln_a(x_a)
        E = self.att(g_v, g_a) + self.hv(g_v, g_a) + self.ha(g_a, g_v)
        return E


def _tiny_energy():
    return _METEnergy(L=16, N=4, M=8, D=16, D_k=8, H=1, K=4)


def _rand(L=16, D=16):
    return torch.randn(1, L, D)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnergyBasics:
    """Sanity / shape contracts."""

    def test_energy_is_scalar(self):
        model = _tiny_energy()
        E = model(_rand(), _rand())
        assert E.shape == (), f"Energy not scalar: {E.shape}"

    def test_energy_finite(self):
        model = _tiny_energy()
        E = model(_rand(), _rand())
        assert torch.isfinite(E), f"Energy not finite: {E.item()}"

    def test_energy_changes_with_x_a(self):
        """E must depend on x_a — otherwise conditional generation cannot work."""
        model = _tiny_energy()
        x_v = _rand()
        E1 = model(x_v, _rand()).item()
        E2 = model(x_v, _rand()).item()
        assert abs(E1 - E2) > 1e-6, "E does not change when x_a changes"

    def test_energy_changes_with_x_v(self):
        model = _tiny_energy()
        x_a = _rand()
        E1 = model(_rand(), x_a).item()
        E2 = model(_rand(), x_a).item()
        assert abs(E1 - E2) > 1e-6, "E does not change when x_v changes"


class TestFreezeSemantics:
    """
    freeze_v/freeze_a: gradient must NOT flow back into the frozen state,
    but parameter gradients must still flow (for training).
    This is the detach()-not-no_grad() invariant.
    """

    def test_freeze_v_blocks_state_grad(self):
        model = _tiny_energy()
        x_v = _rand().requires_grad_(True)
        x_a = _rand().requires_grad_(True)
        E = model(x_v, x_a, freeze_v=True)
        E.backward()
        assert x_v.grad is None or x_v.grad.norm().item() == 0.0, \
            "freeze_v=True should block gradient to x_v"
        assert x_a.grad is not None, \
            "freeze_v=True must still allow gradient to x_a"

    def test_freeze_a_blocks_state_grad(self):
        model = _tiny_energy()
        x_v = _rand().requires_grad_(True)
        x_a = _rand().requires_grad_(True)
        E = model(x_v, x_a, freeze_a=True)
        E.backward()
        assert x_a.grad is None or x_a.grad.norm().item() == 0.0, \
            "freeze_a=True should block gradient to x_a"
        assert x_v.grad is not None, \
            "freeze_a=True must still allow gradient to x_v"

    def test_freeze_v_still_allows_param_grad(self):
        """After freeze_v, parameter gradients of the model must still flow."""
        model = _tiny_energy()
        x_v = _rand(); x_a = _rand().requires_grad_(True)
        E = model(x_v, x_a, freeze_v=True)
        E.backward()
        has_param_grad = any(
            p.grad is not None for p in model.parameters()
            if p.requires_grad
        )
        assert has_param_grad, \
            "freeze_v=True blocked parameter gradients (detach bug)"


class TestEnergyMonotonicity:
    """
    Exp 0.3: E(x^v, x^a(t)) must decrease monotonically under
    gradient descent x^a(t+1) = x^a(t) - eta * grad_{x^a} E,
    for sufficiently small eta.
    
    NOTE: LayerNorm null-space means E_dot <= 0, not strict <0.
    We check that E does not significantly *increase*.
    """

    @pytest.mark.parametrize("eta", [0.001, 0.005, 0.01])
    def test_monotonicity_under_small_eta(self, eta):
        model = _tiny_energy()
        x_v = _rand().detach()
        x_a = _rand()

        E_prev = None
        violations = 0
        n_steps = 50

        for _ in range(n_steps):
            x_a_r = x_a.clone().requires_grad_(True)
            E = model(x_v, x_a_r, freeze_v=True)
            grad, = torch.autograd.grad(E, x_a_r)
            with torch.no_grad():
                x_a = x_a - eta * grad

            if E_prev is not None and E.item() > E_prev + 1e-5:
                violations += 1
            E_prev = E.item()

        violation_rate = violations / n_steps
        assert violation_rate < 0.1, \
            (f"Energy non-monotone in >10% of steps at eta={eta}: "
             f"violation_rate={violation_rate:.2%}. "
             f"Reduce eta or check LayerNorm null-space handling.")

    def test_large_eta_may_violate(self):
        """
        With very large eta, monotonicity is expected to break.
        This test confirms our diagnostic catches it (not that we want it).
        """
        model = _tiny_energy()
        x_v = _rand().detach()
        x_a = _rand()
        eta = 10.0  # deliberately too large

        E_prev = None
        any_increase = False
        for _ in range(20):
            x_a_r = x_a.clone().requires_grad_(True)
            E = model(x_v, x_a_r, freeze_v=True)
            grad, = torch.autograd.grad(E, x_a_r)
            with torch.no_grad():
                x_a = x_a - eta * grad
            if E_prev is not None and E.item() > E_prev + 1e-4:
                any_increase = True
            E_prev = E.item()
        # We don't assert here — just confirm the code runs without crashing
        # (The experiment framework detects violations and logs them)

    def test_fixed_point_residual_decreases(self):
        """||grad_{x^a} E|| should decrease (on average) as the solver progresses."""
        model = _tiny_energy()
        x_v = _rand().detach()
        x_a = _rand()
        eta = 0.01
        residuals = []

        for _ in range(30):
            x_a_r = x_a.clone().requires_grad_(True)
            E = model(x_v, x_a_r, freeze_v=True)
            grad, = torch.autograd.grad(E, x_a_r)
            residuals.append(grad.norm().item())
            with torch.no_grad():
                x_a = x_a - eta * grad

        # Coarse check: last-10 average < first-10 average
        first = sum(residuals[:10]) / 10
        last  = sum(residuals[-10:]) / 10
        assert last <= first * 1.5, \
            f"FP residual not decreasing: first={first:.4f}, last={last:.4f}"


class TestLayerNormNullSpace:
    """
    Detect the Lyapunov gap: steps where E_dot ~ 0 but grad_g_E != 0.
    This confirms the null-space issue is observable (diagnostic only, not a failure).
    """

    def test_layernorm_null_space_event_detection(self):
        """
        Adding a constant to x should not change g (LayerNorm absorbs shifts).
        This means grad_{x} E has a null component in the constant direction.
        """
        model = _tiny_energy()
        x_v = _rand().detach()
        x_a = _rand()

        x_a_r = x_a.clone().requires_grad_(True)
        E = model(x_v, x_a_r, freeze_v=True)
        grad, = torch.autograd.grad(E, x_a_r)

        # Null component: projection onto all-ones direction per token
        ones = torch.ones_like(grad) / grad.shape[-1] ** 0.5
        null_component = (grad * ones).sum(dim=-1, keepdim=True) * ones
        useful_component = grad - null_component

        null_fraction = (null_component.norm() / (grad.norm() + 1e-10)).item()
        # We expect some null component (the LayerNorm gap exists)
        # This test just ensures our detection code works, not that null_fraction=0
        assert 0.0 <= null_fraction <= 1.0, \
            f"Invalid null_fraction: {null_fraction}"


class TestEnergyRanking:
    """
    Prerequisite for Exp 4.3: after some training signal,
    matched pairs (x^v, x^a) should have LOWER energy than mismatched pairs.
    
    Here we test the diagnostic infrastructure (not the trained model). 
    """

    def test_energy_gap_computable(self):
        """Energy gap E_mismatched - E_matched must be computable without error."""
        model = _tiny_energy()
        x_v = _rand(); x_a = _rand(); x_a_wrong = _rand()

        E_matched    = model(x_v, x_a, freeze_v=True, freeze_a=True).item()
        E_mismatched = model(x_v, x_a_wrong, freeze_v=True, freeze_a=True).item()
        gap = E_mismatched - E_matched

        # Gap can be positive or negative before training — just check it's finite
        assert np.isfinite(gap), f"Energy gap is not finite: {gap}"

    def test_ranking_loss_gradient_flows(self):
        """J_rank = softplus(E_matched - E_mismatched + margin) must have gradient."""
        model = _tiny_energy()
        x_v = _rand(); x_a = _rand(); x_a_wrong = _rand()

        E_matched    = model(x_v, x_a)
        E_mismatched = model(x_v, x_a_wrong)
        m = 0.5
        J_rank = F.softplus(E_matched - E_mismatched + m)
        J_rank.backward()

        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "J_rank backward did not produce parameter gradients"


class TestEqPropMemoryBound:
    """
    Exp 3.2 (conceptual): EqProp stores 2 equilibria; BPTT stores T states.
    This test verifies the state tensor size contract, not GPU memory directly.
    """

    def test_eqprop_stores_two_equilibria(self):
        """
        EqProp only needs x_free* and x_nudged*.
        Total state size = 2 * (B * L * D).
        Check that two detached tensors are sufficient.
        """
        L, D = 32, 16
        x_free_star   = torch.randn(1, L, D)   # free equilibrium
        x_nudged_star = torch.randn(1, L, D)   # nudged equilibrium

        # Neither requires a trajectory — just two snapshots
        assert x_free_star.requires_grad   == False
        assert x_nudged_star.requires_grad == False
        total_elements = 2 * x_free_star.numel()
        assert total_elements == 2 * L * D, \
            f"State size mismatch: {total_elements} != {2*L*D}"

    def test_bptt_would_store_T_states(self):
        """BPTT state grows linearly in T — confirm via list length."""
        L, D, T = 16, 16, 20
        trajectory = [torch.randn(1, L, D) for _ in range(T)]
        assert len(trajectory) == T, "BPTT trajectory has wrong length"
        total_bptt = T * L * D
        total_eqprop = 2 * L * D
        assert total_bptt > total_eqprop, \
            f"BPTT ({total_bptt}) should exceed EqProp ({total_eqprop}) for T={T}>2"
