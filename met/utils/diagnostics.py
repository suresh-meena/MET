"""
met/utils/diagnostics.py
=========================
Diagnostic logging utilities for MET training and inference.

Tracks the quantities specified in experiments.md §7 (Diagnostic Logging Template):
    E_free, E_nudged, E_matched, E_mismatched, E_gap
    fp_residual, attractor_agreement, energy_monotone_rate
    grad_cosine_eqprop_bptt, layernorm_null_events
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


class SolverDiagnostics:
    """
    Accumulates per-step solver diagnostics and computes summary statistics.

    Usage:
        diag = SolverDiagnostics()
        for step_log in solver_logs:
            diag.update(step_log)
        summary = diag.summary()
    """

    def __init__(self) -> None:
        self._logs: list[dict] = []

    def update(self, step_log: dict) -> None:
        """Append one solver step log."""
        self._logs.append(step_log)

    def summary(self) -> dict:
        """Compute summary statistics over all recorded steps."""
        if not self._logs:
            return {}

        E_vals = [l.get("E_total", float("nan")) for l in self._logs]
        fp_vals = [l.get("fp_residual", float("nan")) for l in self._logs]
        mono_vals = [l.get("monotone", True) for l in self._logs]

        n = len(E_vals)
        monotone_rate = sum(mono_vals) / n if n > 0 else float("nan")

        return {
            "E_final":            E_vals[-1],
            "E_initial":          E_vals[0],
            "E_delta":            E_vals[-1] - E_vals[0],
            "fp_residual_final":  fp_vals[-1],
            "fp_residual_max":    max(fp_vals),
            "energy_monotone_rate": monotone_rate,
            "n_steps":            n,
        }

    def reset(self) -> None:
        self._logs.clear()


class EpochLogger:
    """
    Accumulates batch-level metrics over an epoch and computes mean / std.

    Usage:
        logger = EpochLogger()
        logger.log(E_free=0.5, E_nudged=0.4, attractor_agreement=0.97)
        ...
        epoch_summary = logger.epoch_summary()
        logger.reset()
    """

    def __init__(self) -> None:
        self._data: dict[str, list[float]] = defaultdict(list)

    def log(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            self._data[k].append(float(v))

    def epoch_summary(self) -> dict[str, dict[str, float]]:
        """Returns {metric: {mean, std, min, max}} for each tracked metric."""
        summary = {}
        for k, vals in self._data.items():
            t = torch.tensor(vals)
            summary[k] = {
                "mean": t.mean().item(),
                "std":  t.std().item() if len(vals) > 1 else 0.0,
                "min":  t.min().item(),
                "max":  t.max().item(),
            }
        return summary

    def reset(self) -> None:
        self._data.clear()


def layernorm_null_events(
    grad_x: Tensor,  # (B, L, D)  gradient w.r.t. pre-norm state x
    E_dot: float,    # Ė at this step (negative means descent)
    eps_edot: float = 1e-6,
    eps_grad: float = 1e-4,
) -> bool:
    """
    Detect a LayerNorm null-space event:
        Ė ≈ 0  BUT  ‖∇_g E‖ ≠ 0

    This means gradient descent is stalled in the LayerNorm null space
    (the mean-shift direction) — the Lyapunov gap from §3.3.

    Returns True if a null-space event is detected.
    """
    edot_zero = abs(E_dot) < eps_edot
    # Useful component = grad_x - null_component
    # Null component per token = per-token mean projection
    null_comp = grad_x.mean(dim=-1, keepdim=True)
    useful_grad = grad_x - null_comp
    grad_norm = useful_grad.norm().item()
    return edot_zero and (grad_norm > eps_grad)


def attractor_agreement(
    x_free: Tensor,  # (B, L, D)
    x_nudged: Tensor,  # (B, L, D)
) -> float:
    """Cosine similarity between free and nudged equilibria (per batch; averaged)."""
    return F.cosine_similarity(
        x_free.flatten(1), x_nudged.flatten(1), dim=-1
    ).mean().item()


def grad_cosine_similarity(
    grads_a: list[Tensor | None],
    grads_b: list[Tensor | None],
) -> float:
    """
    Compute cosine similarity between two gradient vectors (e.g., EqProp vs BPTT).
    Flattens all parameter gradients into a single vector.
    """
    flat_a = torch.cat([g.flatten() for g in grads_a if g is not None])
    flat_b = torch.cat([g.flatten() for g in grads_b if g is not None])
    return F.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0)).item()
