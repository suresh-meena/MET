"""
tests/test_energy.py
====================
Integration tests for real MET energy + solver + EqProp behavior.
"""

from __future__ import annotations

import torch
import pytest

from met.core.energy import METConfig, METEnergy
from met.solver.eqprop import EqPropEstimator
from met.solver.gradient_descent import run_deterministic_solver
from met.training.losses import J_rank


def _make_tiny_energy() -> tuple[METEnergy, METConfig]:
    cfg = METConfig(
        L=8,
        D=12,
        D_k=4,
        H=2,
        N_v=4,
        N_a=4,
        M_v=6,
        M_a=6,
        K_v=5,
        K_a=5,
        beta=1.0,
        beta_HN=1.0,
        lambda_cross=0.05,
        lam_spline=1e-3,
        window=3,
    )
    return METEnergy(cfg), cfg


def _rand_state(cfg: METConfig, B: int = 2) -> torch.Tensor:
    return torch.randn(B, cfg.L, cfg.D)


def _param_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters() if p.requires_grad])


class TestEnergyForward:
    def test_forward_scalar_and_components_are_finite(self):
        model, cfg = _make_tiny_energy()
        E, comps = model(_rand_state(cfg), _rand_state(cfg))
        assert E.shape == ()
        assert torch.isfinite(E)
        expected = {"E_intra_v", "E_intra_a", "E_cross", "E_HN_v", "E_HN_a", "E_total"}
        assert expected.issubset(set(comps.keys()))
        assert all(torch.isfinite(torch.tensor(v)) for v in comps.values())

    def test_energy_depends_on_both_modalities(self):
        model, cfg = _make_tiny_energy()
        x_v = _rand_state(cfg)
        x_a1 = _rand_state(cfg)
        x_a2 = _rand_state(cfg)
        E1, _ = model(x_v, x_a1)
        E2, _ = model(x_v, x_a2)
        assert abs(E1.item() - E2.item()) > 1e-6


class TestFreezeSemantics:
    def test_freeze_v_blocks_x_v_grad_but_not_params(self):
        model, cfg = _make_tiny_energy()
        x_v = _rand_state(cfg).requires_grad_(True)
        x_a = _rand_state(cfg).requires_grad_(True)

        E, _ = model(x_v, x_a, freeze_v=True, freeze_a=False)
        E.backward()

        assert x_v.grad is None or x_v.grad.norm().item() == 0.0
        assert x_a.grad is not None and x_a.grad.norm().item() > 0
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_freeze_a_blocks_x_a_grad_but_not_params(self):
        model, cfg = _make_tiny_energy()
        x_v = _rand_state(cfg).requires_grad_(True)
        x_a = _rand_state(cfg).requires_grad_(True)

        E, _ = model(x_v, x_a, freeze_v=False, freeze_a=True)
        E.backward()

        assert x_a.grad is None or x_a.grad.norm().item() == 0.0
        assert x_v.grad is not None and x_v.grad.norm().item() > 0
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


class TestLayerNormInvariant:
    def test_layernorm_is_shift_invariant_per_token(self):
        model, cfg = _make_tiny_energy()
        x = _rand_state(cfg)
        shift = torch.randn(x.shape[0], x.shape[1], 1)

        g1 = model.ln_a(x)
        g2 = model.ln_a(x + shift)
        assert torch.allclose(g1, g2, atol=1e-5)


class TestDeterministicSolver:
    def test_solver_updates_only_unfrozen_modality(self):
        model, cfg = _make_tiny_energy()
        x_v0 = _rand_state(cfg)
        x_a0 = _rand_state(cfg)

        x_v1, x_a1, logs = run_deterministic_solver(
            model,
            x_v0,
            x_a0,
            T=8,
            eta=1e-3,
            freeze_v=True,
            attention_only=True,
        )

        assert torch.allclose(x_v1, x_v0), "Frozen modality should remain unchanged"
        assert not torch.allclose(x_a1, x_a0), "Unfrozen modality should be updated"
        assert len(logs) > 0
        assert logs[-1]["E_total"] <= logs[0]["E_total"] + 1e-4

    def test_solver_raises_when_both_modalities_frozen(self):
        model, cfg = _make_tiny_energy()
        with pytest.raises(AssertionError):
            run_deterministic_solver(
                model,
                _rand_state(cfg),
                _rand_state(cfg),
                T=2,
                eta=1e-3,
                freeze_v=True,
                freeze_a=True,
            )

    def test_create_graph_true_preserves_graph_for_bptt(self):
        model, cfg = _make_tiny_energy()
        x_v0 = _rand_state(cfg)
        x_a0 = _rand_state(cfg)

        x_v_out, x_a_out, _ = run_deterministic_solver(
            model,
            x_v0,
            x_a0,
            T=4,
            eta=1e-3,
            freeze_v=True,
            create_graph=True,
            attention_only=True,
        )
        assert x_a_out.requires_grad

        loss = (x_a_out ** 2).mean()
        model.zero_grad()
        loss.backward()
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


class TestRankingSignal:
    def test_rank_loss_can_improve_matched_vs_mismatched_gap(self):
        model, cfg = _make_tiny_energy()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        B = 24
        x_v = _rand_state(cfg, B=B)
        M = torch.randn(cfg.D, cfg.D) * 0.1
        x_a = torch.einsum("ij,blj->bli", M, x_v) + 0.05 * torch.randn_like(x_v)
        x_a_mm = torch.roll(x_a, shifts=1, dims=0)

        with torch.no_grad():
            E_m0, _ = model(x_v, x_a, freeze_v=True, freeze_a=True)
            E_mm0, _ = model(x_v, x_a_mm, freeze_v=True, freeze_a=True)
            gap0 = (E_mm0 - E_m0).item()
            loss0 = J_rank(E_m0, E_mm0, margin=0.5).item()

        for _ in range(20):
            E_m, _ = model(x_v, x_a, freeze_v=True, freeze_a=True)
            E_mm, _ = model(x_v, x_a_mm, freeze_v=True, freeze_a=True)
            loss = J_rank(E_m, E_mm, margin=0.5)
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            E_m1, _ = model(x_v, x_a, freeze_v=True, freeze_a=True)
            E_mm1, _ = model(x_v, x_a_mm, freeze_v=True, freeze_a=True)
            gap1 = (E_mm1 - E_m1).item()
            loss1 = J_rank(E_m1, E_mm1, margin=0.5).item()

        assert loss1 < loss0, f"Expected lower ranking loss, got {loss0:.4f} -> {loss1:.4f}"
        assert gap1 >= gap0 - 1e-5, f"Expected non-worse ranking gap, got {gap0:.4f} -> {gap1:.4f}"


class TestEqPropIntegration:
    def test_estimate_gradient_contract(self):
        model, cfg = _make_tiny_energy()
        eq = EqPropEstimator(model, s=0.01, T=4)

        x_v = _rand_state(cfg, B=1)
        x_a = _rand_state(cfg, B=1)
        x_target = _rand_state(cfg, B=1)

        result = eq.estimate_gradient(x_v, x_a, x_target, eta=1e-3, freeze_v=True)
        grads = result["eqprop_grads"]

        assert set(grads.keys()) == {name for name, _ in model.named_parameters()}
        non_none = [g for g in grads.values() if g is not None]
        assert len(non_none) > 0
        assert all(torch.isfinite(g).all() for g in non_none)
        assert -1.0 <= result["attractor_agreement"] <= 1.0
        assert result["free_steps"] > 0
        assert result["nudged_steps"] > 0

    def test_apply_gradients_respects_agreement_gate(self):
        model, cfg = _make_tiny_energy()
        eq = EqPropEstimator(model, s=0.01, T=3)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x_v = _rand_state(cfg, B=1)
        x_a = _rand_state(cfg, B=1)
        x_target = _rand_state(cfg, B=1)
        result = eq.estimate_gradient(x_v, x_a, x_target, eta=1e-3, freeze_v=True)

        before = _param_vector(model)
        applied = eq.apply_gradients(
            result["eqprop_grads"],
            optimizer=opt,
            min_agreement=2.0,  # impossible threshold -> should skip
            agreement=result["attractor_agreement"],
        )
        after_skip = _param_vector(model)

        assert applied is False
        assert torch.allclose(before, after_skip)

        applied = eq.apply_gradients(
            result["eqprop_grads"],
            optimizer=opt,
            min_agreement=-1.0,  # always pass
            agreement=result["attractor_agreement"],
        )
        after_apply = _param_vector(model)

        assert applied is True
        assert not torch.allclose(after_skip, after_apply)
