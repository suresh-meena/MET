"""
met/solver/eqprop.py
====================
Equilibrium Propagation gradient estimator.

The key requirement from the writeup is that the nudged phase solves

    E_nudged(x) = E(x; theta) + s * J_nudge(x),

not just a warm-started copy of the free phase. Warm starts are still useful,
but they are not the objective perturbation.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from met.core.energy import METEnergy
from met.solver.gradient_descent import run_deterministic_solver


NudgeObjective = Callable[[Tensor, Tensor], Tensor]


class EqPropEstimator:
    """
    Equilibrium Propagation gradient estimator for METEnergy.

    Args:
        energy: METEnergy module
        s:      nudge strength
        T:      solver iterations per phase
    """

    def __init__(self, energy: METEnergy, s: float = 0.01, T: int = 50) -> None:
        self.energy = energy
        self.s = s
        self.T = T

    def _default_nudge_objective(
        self,
        x_a_target: Optional[Tensor],
        freeze_v: bool,
        freeze_a: bool,
    ) -> NudgeObjective:
        if x_a_target is None:
            raise ValueError(
                "EqProp requires a nudge objective. Pass nudge_objective=... "
                "or provide x_a_target for the default latent-state MSE."
            )
        if freeze_a and not freeze_v:
            raise ValueError(
                "The default x_a_target nudge only covers audio-state updates. "
                "For audio-to-video EqProp, pass a custom nudge_objective."
            )

        target = x_a_target.detach()
        return lambda _x_v, x_a: F.mse_loss(x_a, target)

    @staticmethod
    def _agreement_tensor(
        x_v_free: Tensor,
        x_a_free: Tensor,
        x_v_nudged: Tensor,
        x_a_nudged: Tensor,
        freeze_v: bool,
        freeze_a: bool,
    ) -> tuple[Tensor, Tensor]:
        if freeze_v and not freeze_a:
            return x_a_free, x_a_nudged
        if freeze_a and not freeze_v:
            return x_v_free, x_v_nudged
        free = torch.cat([x_v_free.flatten(1), x_a_free.flatten(1)], dim=-1)
        nudged = torch.cat([x_v_nudged.flatten(1), x_a_nudged.flatten(1)], dim=-1)
        return free, nudged

    def estimate_gradient(
        self,
        x_v: Tensor,
        x_a: Tensor,
        x_a_target: Optional[Tensor] = None,
        eta: float = 0.01,
        freeze_v: bool = True,
        freeze_a: bool = False,
        grad_clip: float = 1.0,
        nudge_objective: Optional[NudgeObjective] = None,
    ) -> dict:
        """
        Run free and nudged phases, return EqProp gradient estimates.

        The default `x_a_target` path is preserved for backward compatibility.
        For the writeup-faithful setup, pass `nudge_objective`, typically a
        task-head reconstruction loss in target space.
        """
        nudge_objective = nudge_objective or self._default_nudge_objective(
            x_a_target=x_a_target,
            freeze_v=freeze_v,
            freeze_a=freeze_a,
        )
        params = list(self.energy.parameters())

        x_v_free, x_a_free, free_logs = run_deterministic_solver(
            self.energy,
            x_v,
            x_a,
            T=self.T,
            eta=eta,
            freeze_v=freeze_v,
            freeze_a=freeze_a,
            grad_clip=grad_clip,
        )
        E_free, _ = self.energy(
            x_v_free,
            x_a_free,
            freeze_v=freeze_v,
            freeze_a=freeze_a,
        )
        grad_free = torch.autograd.grad(
            E_free, params, allow_unused=True, retain_graph=False
        )

        x_v_nudged, x_a_nudged, nudged_logs = run_deterministic_solver(
            self.energy,
            x_v_free,
            x_a_free,
            T=self.T,
            eta=eta,
            freeze_v=freeze_v,
            freeze_a=freeze_a,
            grad_clip=grad_clip,
            extra_objective=nudge_objective,
            extra_scale=self.s,
        )
        E_nudged, _ = self.energy(
            x_v_nudged,
            x_a_nudged,
            freeze_v=freeze_v,
            freeze_a=freeze_a,
        )
        grad_nudged = torch.autograd.grad(
            E_nudged, params, allow_unused=True, retain_graph=False
        )

        eqprop_grads: dict[str, Tensor | None] = {}
        for (name, _), gn, gf in zip(
            self.energy.named_parameters(), grad_nudged, grad_free
        ):
            if gn is not None and gf is not None:
                eqprop_grads[name] = (gn - gf) / self.s
            else:
                eqprop_grads[name] = None

        free_state, nudged_state = self._agreement_tensor(
            x_v_free,
            x_a_free,
            x_v_nudged,
            x_a_nudged,
            freeze_v=freeze_v,
            freeze_a=freeze_a,
        )
        agreement = F.cosine_similarity(
            free_state.flatten(1), nudged_state.flatten(1), dim=-1
        ).mean().item()

        return dict(
            eqprop_grads=eqprop_grads,
            attractor_agreement=agreement,
            E_free=E_free.item(),
            E_nudged=E_nudged.item(),
            free_steps=len(free_logs),
            nudged_steps=len(nudged_logs),
            x_v_free=x_v_free,
            x_a_free=x_a_free,
            x_v_nudged=x_v_nudged,
            x_a_nudged=x_a_nudged,
        )

    def apply_gradients(
        self,
        eqprop_grads: dict[str, Tensor | None],
        optimizer: torch.optim.Optimizer,
        min_agreement: float = 0.90,
        agreement: float = 1.0,
    ) -> bool:
        """Assign EqProp gradient estimates to energy parameters and step."""
        if agreement < min_agreement:
            return False

        optimizer.zero_grad()
        for name, param in self.energy.named_parameters():
            grad = eqprop_grads.get(name)
            if grad is not None:
                param.grad = grad.detach()

        optimizer.step()
        return True
