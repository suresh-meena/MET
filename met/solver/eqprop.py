"""
met/solver/eqprop.py
====================
Equilibrium Propagation gradient estimator.

Algorithm (§5):
    1. Free phase:   run solver → x_free*  = argmin_x E(x; θ)
    2. Nudged phase: run solver → x_nudged* = argmin_x [E(x; θ) + s·J_nudge(x)]
                    WARM-STARTED from x_free* (same attractor basin)
    3. Estimate:    ∇_θ J ≈ [∇_θ E(x_nudged*; θ) − ∇_θ E(x_free*; θ)] / s
                    → converges to exact gradient as s → 0

Memory advantage:
    Stores 2 equilibrium tensors (O(B·L·D·2)).
    BPTT stores T tensors (O(B·L·D·T)).
    For T=50, B=8, L=64, D=256: EqProp ~25 MB vs BPTT ~1.6 GB.

Validity conditions (§5, Limitations):
    ─ Only J_gen (samplewise reconstruction) enters the nudge.
      J_sem, J_temp, J_JEPA are outer-loop auxiliaries.
    ─ Free and nudged phases MUST converge to the same attractor.
      Monitor attractor_agreement throughout training.
      If agreement drops: reduce s, warm-start nudged, or reduce beta_HN.
    ─ Finite s introduces O(s) bias; s=0.01 is recommended (Exp 3.1).
    ─ Valid only under EqProp symmetry assumptions (see Scellier & Bengio 2017).

Attractor agreement diagnostic:
    Cosine similarity between x_a_free* and x_a_nudged*.
    If < 0.95, the estimator is unreliable for that sample.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from met.core.energy import METEnergy
from met.solver.gradient_descent import run_deterministic_solver


class EqPropEstimator:
    """
    Equilibrium Propagation gradient estimator for METEnergy.

    Args:
        energy: METEnergy module (shared with trainer)
        s:      nudge strength (default 0.01)
        T:      solver iterations per phase (default 50)
    """

    def __init__(self, energy: METEnergy, s: float = 0.01, T: int = 50) -> None:
        self.energy = energy
        self.s = s
        self.T = T

    def estimate_gradient(
        self,
        x_v: Tensor,
        x_a: Tensor,
        x_a_target: Tensor,
        eta: float,
        freeze_v: bool = True,
        grad_clip: float = 1.0,
    ) -> dict:
        """
        Run free and nudged phases, return EqProp gradient estimates.

        Args:
            x_v:        (B, L, D)  video conditioning (frozen for Foley)
            x_a:        (B, L, D)  initial audio state
            x_a_target: (B, L, D)  ground-truth audio for J_nudge = MSE
            eta:        solver step size
            freeze_v:   True for Foley (video is conditioning)
            grad_clip:  gradient clipping in solver

        Returns dict with:
            eqprop_grads:        dict[name → Tensor | None]
            attractor_agreement: float in [0, 1]  (1 = same basin)
            E_free:              float
            E_nudged:            float
            free_steps:          int
            nudged_steps:        int
        """
        params = list(self.energy.parameters())

        # ---- Free phase ----
        x_v_f, x_a_f, free_logs = run_deterministic_solver(
            self.energy, x_v, x_a,
            T=self.T, eta=eta,
            freeze_v=freeze_v,
            grad_clip=grad_clip,
        )
        E_free, _ = self.energy(x_v_f, x_a_f, freeze_v=freeze_v)
        grad_free = torch.autograd.grad(
            E_free, params, allow_unused=True, retain_graph=False
        )

        # ---- Nudged phase ----
        # Warm-start: perturb x_a_free* by s · ∇_{x_a} J_nudge
        # This keeps the nudged phase in the same attractor basin as free.
        x_a_f_detached = x_a_f.detach().requires_grad_(True)
        J_nudge = F.mse_loss(x_a_f_detached, x_a_target.detach())
        nudge_grad, = torch.autograd.grad(J_nudge, x_a_f_detached)
        x_a_nudge_init = (x_a_f + self.s * nudge_grad.detach()).detach()

        x_v_n, x_a_n, nudge_logs = run_deterministic_solver(
            self.energy, x_v_f if freeze_v else x_v, x_a_nudge_init,
            T=self.T, eta=eta,
            freeze_v=freeze_v,
            grad_clip=grad_clip,
        )
        E_nudged, _ = self.energy(x_v_n, x_a_n, freeze_v=freeze_v)
        grad_nudged = torch.autograd.grad(
            E_nudged, params, allow_unused=True, retain_graph=False
        )

        # ---- EqProp gradient estimate ----
        eqprop_grads: dict[str, Tensor | None] = {}
        for (name, _), gn, gf in zip(
            self.energy.named_parameters(), grad_nudged, grad_free
        ):
            if gn is not None and gf is not None:
                eqprop_grads[name] = (gn - gf) / self.s
            else:
                eqprop_grads[name] = None

        # ---- Attractor agreement diagnostic ----
        # Cosine sim between free and nudged equilibria
        agreement = F.cosine_similarity(
            x_a_f.flatten(1), x_a_n.flatten(1), dim=-1
        ).mean().item()

        return dict(
            eqprop_grads=eqprop_grads,
            attractor_agreement=agreement,
            E_free=E_free.item(),
            E_nudged=E_nudged.item(),
            free_steps=len(free_logs),
            nudged_steps=len(nudge_logs),
        )

    def apply_gradients(
        self,
        eqprop_grads: dict[str, Tensor | None],
        optimizer: torch.optim.Optimizer,
        min_agreement: float = 0.90,
        agreement: float = 1.0,
    ) -> bool:
        """
        Manually assign EqProp gradient estimates to parameter .grad fields,
        then call optimizer.step().

        Args:
            eqprop_grads:   output of estimate_gradient['eqprop_grads']
            optimizer:      the model's optimizer
            min_agreement:  skip update if attractor agreement is below this
            agreement:      attractor_agreement value from the same call

        Returns:
            True if update was applied, False if skipped (low agreement).
        """
        if agreement < min_agreement:
            return False

        optimizer.zero_grad()
        for name, param in self.energy.named_parameters():
            g = eqprop_grads.get(name)
            if g is not None:
                param.grad = g.detach()

        optimizer.step()
        return True
