import torch

from met.core import BSplineCache, FourierBasisCache, METConfig, METEnergy
from met.heads.audio_head import AudioHead
from met.solver.eqprop import EqPropEstimator


def test_basis_caches_share_encode_contract():
    B, H, L, D_k = 2, 3, 12, 5
    K = torch.randn(B, H, L, D_k)

    spline = BSplineCache(L=L, N=6, M=8, lam=1e-3)
    fourier = FourierBasisCache(L=L, N=6, M=8, lam=1e-3)

    for cache in (spline, fourier):
        coeffs, K_quad = cache.encode(K)
        assert coeffs.shape == (B, H, 6, D_k)
        assert K_quad.shape == (B, H, 8, D_k)


def test_energy_supports_different_basis_families_and_quadrature_sizes():
    cfg = METConfig(
        L=10,
        D=12,
        D_k=4,
        H=2,
        N_v=5,
        N_a=7,
        M_v=6,
        M_a=9,
        K_v=3,
        K_a=3,
        basis_v="fourier",
        basis_a="bspline",
    )
    energy = METEnergy(cfg)
    x_v = torch.randn(2, cfg.L, cfg.D)
    x_a = torch.randn(2, cfg.L, cfg.D)

    E, comps = energy(x_v, x_a)
    assert torch.isfinite(E)
    assert "E_cross" in comps


def test_eqprop_uses_custom_nudge_objective_and_returns_equilibria():
    cfg = METConfig(
        L=8,
        D=10,
        D_k=4,
        H=1,
        N_v=4,
        N_a=4,
        M_v=6,
        M_a=8,
        K_v=2,
        K_a=2,
    )
    energy = METEnergy(cfg)
    head = AudioHead(D=cfg.D, n_mels=6)
    eqprop = EqPropEstimator(energy, s=0.01, T=5)

    x_v = torch.randn(1, cfg.L, cfg.D)
    x_a = torch.randn(1, cfg.L, cfg.D)
    mel_target = torch.randn(1, cfg.L, 6)

    result = eqprop.estimate_gradient(
        x_v,
        x_a,
        eta=0.01,
        freeze_v=True,
        nudge_objective=lambda _x_v, x_a_state: torch.nn.functional.mse_loss(
            head(x_a_state),
            mel_target,
        ),
    )

    assert result["x_a_free"].shape == x_a.shape
    assert result["x_a_nudged"].shape == x_a.shape
    assert 0.0 <= result["attractor_agreement"] <= 1.0
    assert any(g is not None for g in result["eqprop_grads"].values())
