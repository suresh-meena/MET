"""
met/utils/grad_check.py
=======================
Finite-difference gradient verifier for MET components.

Used in Tier 0 experiments (Exp 0.1, 0.2) as correctness gates.
Must pass before any training begins.

Common failure mode:
    The all-query key-path term (Appendix A, second sum) is missing.
    This happens when only the same-token key contribution is computed
    manually instead of using autograd through R_m @ K_h.
    The autograd path is always correct; hand-derived paths are not.
"""

from __future__ import annotations

import torch
from torch import Tensor


def finite_difference_grad(
    fn,
    x: Tensor,
    eps: float = 1e-4,
) -> Tensor:
    """
    Central finite-difference gradient of scalar fn(x) w.r.t. x.

    grad_fd[i] = (fn(x + eps*e_i) - fn(x - eps*e_i)) / (2*eps)

    Args:
        fn:  callable, maps x (any shape) → scalar Tensor or float
        x:   input Tensor (any shape); NOT modified in place
        eps: perturbation magnitude (1e-4 is standard for float32)

    Returns:
        grad: Tensor of same shape as x

    Usage:
        def energy_fn(g_v):
            E, _ = energy(g_v, g_a.detach())
            return E

        grad_fd = finite_difference_grad(energy_fn, g_v, eps=1e-4)
        g_v_r = g_v.clone().requires_grad_(True)
        grad_auto = torch.autograd.grad(energy_fn(g_v_r), g_v_r)[0]
        rel_err = (grad_auto - grad_fd).norm() / grad_fd.norm()
        assert rel_err < 1e-3
    """
    x = x.detach().clone()
    grad = torch.zeros_like(x)
    flat_x = x.reshape(-1)

    for i in range(flat_x.numel()):
        x_p = x.clone(); x_p.reshape(-1)[i] += eps
        x_m = x.clone(); x_m.reshape(-1)[i] -= eps

        fp = fn(x_p)
        fm = fn(x_m)

        # Handle scalar Tensor or Python float
        val_p = fp.item() if isinstance(fp, Tensor) else float(fp)
        val_m = fm.item() if isinstance(fm, Tensor) else float(fm)
        grad.reshape(-1)[i] = (val_p - val_m) / (2.0 * eps)

    return grad


def check_grad(
    fn,
    x: Tensor,
    eps: float = 1e-4,
    threshold: float = 1e-3,
    label: str = "",
) -> tuple[float, bool]:
    """
    Compare autograd gradient of fn(x) against central FD.
    Returns (relative_error, passed).

    Args:
        fn:        callable: x → scalar (x must be first positional arg)
        x:         input Tensor
        eps:       FD perturbation
        threshold: relative error threshold (default 1e-3, from experiments.md)
        label:     name for error messages

    Returns:
        rel_err: relative L2 error
        passed:  True if rel_err < threshold
    """
    # Autograd gradient
    x_r = x.detach().clone().requires_grad_(True)
    E = fn(x_r)
    grad_auto, = torch.autograd.grad(E, x_r)

    # FD gradient
    grad_fd = finite_difference_grad(fn, x, eps=eps)

    rel_err = (grad_auto - grad_fd).norm() / (grad_fd.norm() + 1e-10)
    rel_err = rel_err.item()
    passed = rel_err < threshold

    if not passed:
        tag = f" [{label}]" if label else ""
        print(
            f"GRAD CHECK FAILED{tag}: rel_err={rel_err:.2e} > {threshold:.2e}\n"
            f"  Most likely cause: cross-token key-path not accumulated via autograd.\n"
            f"  See Appendix A, second sum in the paper."
        )

    return rel_err, passed


def check_all_tokens(
    fn,
    x: Tensor,  # (B, L, D)
    eps: float = 1e-4,
    threshold: float = 1e-3,
    label: str = "",
) -> dict[str, float | bool]:
    """
    Run check_grad at every token position and return per-token errors.
    Used for Exp 0.1 / 0.2 where the requirement is:
        relative error < 1e-3 at ALL token positions.

    Returns dict with:
        max_rel_err:     max over all tokens
        mean_rel_err:    mean over all tokens
        all_passed:      True iff all tokens pass
        n_failed:        number of token positions failing
    """
    B, L, D = x.shape
    errors = []

    for b in range(B):
        for ell in range(L):
            # FD on a single token
            x_b = x[b]  # (L, D)

            def fn_token(x_ell_vec):
                # Reconstruct full (1, L, D) from perturbed token ell
                x_copy = x_b.clone()
                x_copy[ell] = x_ell_vec
                return fn(x_copy.unsqueeze(0))

            x_ell = x_b[ell].detach()
            rel_err, passed = check_grad(fn_token, x_ell, eps=eps,
                                         threshold=threshold,
                                         label=f"{label}[b={b},ℓ={ell}]")
            errors.append(rel_err)

    return {
        "max_rel_err":  max(errors),
        "mean_rel_err": sum(errors) / len(errors),
        "all_passed":   all(e < threshold for e in errors),
        "n_failed":     sum(1 for e in errors if e >= threshold),
    }
