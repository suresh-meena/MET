"""
met/solver/langevin.py
======================
Unadjusted Langevin Algorithm (ULA) for stochastic MET inference.

WARNING — two distinct use cases:
    MAP inference:   use run_deterministic_solver (no noise term)
    Sampling:        use run_langevin (with noise term)

The ULA update (§6):
    x^a(t+1) = x^a(t) − η ∇_{x^a} E(x̂^v, x^a(t)) + sqrt(2η/β) ε(t)
    ε(t) ~ N(0, I)

This samples APPROXIMATELY from exp(−β E(x̂^v, x^a)) in the small-step
long-run limit. In a multi-attractor energy landscape (from Hopfield banks),
practical mixing can be exponentially slow in the inter-basin barrier height.
Finite-run Langevin = local posterior exploration around initialization.

Initialization matters:
    Cold start:  x^a(0) ~ N(0, I)   — broad exploration
    Warm start:  x^a(0) = nearest retrieved memory pattern   — faster convergence

See §6 and §3 (Limitations) of the paper for the mixing time caveat.
"""

from __future__ import annotations

import torch
from torch import Tensor

from met.core.energy import METEnergy


def run_langevin(
    energy: METEnergy,
    x_v: Tensor,
    x_a: Tensor,
    T: int,
    eta: float,
    beta_inv: float,
    freeze_v: bool = True,
    freeze_a: bool = False,
    grad_clip: float = 5.0,
) -> tuple[Tensor, list[dict]]:
    """
    Unadjusted Langevin Algorithm for stochastic inference.

    Typical Foley usage: freeze_v=True, optimize x_a.

    Args:
        energy:    METEnergy module
        x_v:       (B, L, D)  video state (usually frozen conditioning)
        x_a:       (B, L, D)  audio state (initial sample)
        T:         number of Langevin steps
        eta:       step size η (small for stability)
        beta_inv:  1/β — noise temperature (0 = MAP / deterministic)
        freeze_v:  usually True for Foley (video is conditioning)
        freeze_a:  usually False (audio is what we generate)
        grad_clip: clamp gradients before update

    Returns:
        x_a_final: (B, L, D)  final audio sample
        logs:      list of per-step dicts (E, fp_residual)
    """
    assert not (freeze_v and freeze_a), "Both modalities frozen — nothing to sample."

    x_a = x_a.clone()
    noise_scale = (2.0 * eta * beta_inv) ** 0.5  # sqrt(2η/β)

    logs: list[dict] = []

    for t in range(T):
        # Attach grad for this step only
        x_a_r = x_a.detach().requires_grad_(True)
        E, comps = energy(x_v, x_a_r, freeze_v=freeze_v)
        grad_a, = torch.autograd.grad(E, x_a_r)
        grad_a = grad_a.clamp(-grad_clip, grad_clip)

        with torch.no_grad():
            eps = torch.randn_like(x_a)
            if beta_inv == 0.0:
                # MAP mode: no noise
                x_a = x_a - eta * grad_a
            else:
                x_a = x_a - eta * grad_a + noise_scale * eps

        comps.update(step=t, fp_residual=grad_a.norm().item())
        logs.append(comps)

    return x_a.detach(), logs
