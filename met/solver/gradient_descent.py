"""
met/solver/gradient_descent.py
===============================
Deterministic T-step gradient descent solver.

This is the PRIMARY inference and BPTT-training engine for MET.
Build order (§7 Implementation Blueprint):
    Phase A — run this with create_graph=True and autograd BPTT
    Phase D — replace with EqPropEstimator; keep this as fallback / comparison

Usage:
    # Inference (video-to-audio Foley):
    x_v_final, x_a_final, logs = run_deterministic_solver(
        energy, x_v_cond, x_a_init, T=50, eta=1e-2, freeze_v=True
    )

    # BPTT training (Phase A):
    x_v_out, x_a_out, _ = run_deterministic_solver(
        energy, x_v, x_a, T=20, eta=1e-2, create_graph=True
    )
    loss = J_gen(audio_head(x_a_out), mel_target)
    loss.backward()

Gradient notes:
    create_graph=False (inference): each step detaches states for memory efficiency.
    create_graph=True  (BPTT):     graph is kept alive through all T steps.
        Memory = O(T · B · L · D) — the primary memory cost of BPTT.
        Use EqProp (eqprop.py) to reduce to O(B · L · D).

Stability:
    grad_clip prevents exploding gradients in non-convex E landscapes.
    early_stop_eps stops early when ‖∇_x E‖ < eps (fixed-point reached).
    Monotone check (logged but not enforced) detects LayerNorm null-space events.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from met.core.energy import METEnergy


def run_deterministic_solver(
    energy: METEnergy,
    x_v: Tensor,
    x_a: Tensor,
    T: int,
    eta: float,
    freeze_v: bool = False,
    freeze_a: bool = False,
    grad_clip: float = 1.0,
    early_stop_eps: float = 1e-6,
    create_graph: bool = False,
    attention_only: bool = False,
) -> tuple[Tensor, Tensor, list[dict]]:
    """
    Run T steps of gradient descent on the unfrozen state variable(s).

    Update rule (§4):
        x^(t+1) = x^(t) − η ∇_x E(x^(t))

    This is standard gradient descent (the "Lipschitz gate" in earlier drafts
    reduces algebraically to this; see §4 and Appendix D of the paper).

    Args:
        energy:         METEnergy module
        x_v:            (B, L, D)  initial video state
        x_a:            (B, L, D)  initial audio state
        T:              number of gradient descent steps
        eta:            step size η
        freeze_v:       if True, x_v is conditioning; only x_a is updated
        freeze_a:       if True, x_a is conditioning; only x_v is updated
        grad_clip:      gradient clipping threshold (per-element)
        early_stop_eps: stop if ‖∇_x E‖ < eps
        create_graph:   if True, keep computation graph (BPTT training mode)
        attention_only: use energy.forward_attention_only (Phase A)

    Returns:
        x_v_final:  (B, L, D) final video state.
            - detached when create_graph=False (inference mode)
            - graph-connected when create_graph=True (BPTT mode)
        x_a_final:  (B, L, D) final audio state.
            - detached when create_graph=False
            - graph-connected when create_graph=True
        logs: list of dicts per-step diagnostics
    """
    x_v = x_v.clone()
    x_a = x_a.clone()

    # Requires_grad only on unfrozen states
    if not freeze_v:
        x_v.requires_grad_(True)
    if not freeze_a:
        x_a.requires_grad_(True)

    active = [x for x in [x_v, x_a] if x.requires_grad]
    assert len(active) > 0, "Both modalities are frozen — nothing to update."

    logs: list[dict] = []
    E_prev: Optional[float] = None
    forward_fn = energy.forward_attention_only if attention_only else energy.forward

    for t in range(T):
        E, comps = forward_fn(x_v, x_a, freeze_v=freeze_v, freeze_a=freeze_a)

        # Compute gradients w.r.t. unfrozen states
        grads = torch.autograd.grad(
            E, active, create_graph=create_graph, allow_unused=False
        )

        # Diagnostics
        fp_residual = torch.stack([g.norm() for g in grads]).sum().item()
        monotone = E_prev is None or E.item() <= E_prev + 1e-8
        
        # Don't spend time pulling 5 scalars from GPU if not strictly needed
        comps.update(step=t, fp_residual=fp_residual, monotone=monotone)
        logs.append(comps)
        E_prev = E.item()

        # State update
        if not create_graph:
            with torch.no_grad():
                i = 0
                if not freeze_v:
                    g = grads[i].clamp(-grad_clip, grad_clip); i += 1
                    x_v = (x_v - eta * g).detach().requires_grad_(True)
                if not freeze_a:
                    g = grads[i].clamp(-grad_clip, grad_clip)
                    x_a = (x_a - eta * g).detach().requires_grad_(True)
        else:
            # BPTT: keep graph — update without detaching
            i = 0
            if not freeze_v:
                x_v = x_v - eta * grads[i].clamp(-grad_clip, grad_clip); i += 1
            if not freeze_a:
                x_a = x_a - eta * grads[i].clamp(-grad_clip, grad_clip)

        if fp_residual < early_stop_eps:
            break

    if create_graph:
        return x_v, x_a, logs
    return x_v.detach(), x_a.detach(), logs
