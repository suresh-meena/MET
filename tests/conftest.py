"""
tests/conftest.py
=================
Shared pytest fixtures and utilities for the MET test suite.

Contents:
  - finite_difference_grad(): central FD gradient (used in Exp 0.1 / 0.2)
  - TinyConfig: dataclass with small hyperparameters for fast CPU tests
  - check_relative_error(): assert helper for FD checks
"""

import pytest
import torch
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Finite-difference gradient utility
# ---------------------------------------------------------------------------

def finite_difference_grad(fn, x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Central finite-difference gradient of scalar fn(x) w.r.t. x.
    
    Args:
        fn:  callable, maps x (any shape) -> scalar Tensor
        x:   input Tensor (any shape), will not be modified
        eps: perturbation magnitude
    
    Returns:
        grad: Tensor of same shape as x
    
    Usage:
        grad_fd = finite_difference_grad(lambda g: energy(g, g_other), g_v)
        grad_auto = torch.autograd.grad(energy(g_v_r, g_other), g_v_r)[0]
        rel_err = (grad_auto - grad_fd).norm() / grad_fd.norm()
        assert rel_err < 1e-3
    """
    x = x.detach().clone()
    grad = torch.zeros_like(x)
    flat = x.view(-1)
    for i in range(flat.numel()):
        x_p = x.clone(); x_p.view(-1)[i] += eps
        x_m = x.clone(); x_m.view(-1)[i] -= eps
        fp = fn(x_p)
        fm = fn(x_m)
        # Handle both scalar tensors and Python floats
        grad.view(-1)[i] = (fp.item() - fm.item()) / (2 * eps)
    return grad


def check_relative_error(
    grad_auto: torch.Tensor,
    grad_fd: torch.Tensor,
    threshold: float = 1e-3,
    label: str = "",
) -> float:
    """
    Asserts relative L2 error between autodiff and FD gradients is below threshold.
    Returns the relative error for logging.
    """
    rel_err = (grad_auto - grad_fd).norm() / (grad_fd.norm() + 1e-10)
    rel_err = rel_err.item()
    assert rel_err < threshold, (
        f"Gradient check FAILED{' (' + label + ')' if label else ''}: "
        f"relative error = {rel_err:.2e} > threshold {threshold:.2e}. "
        f"Most likely cause: cross-token key-path not accumulated via autograd "
        f"(see Appendix A of the paper)."
    )
    return rel_err


# ---------------------------------------------------------------------------
# Tiny configuration dataclass for consistent test sizing
# ---------------------------------------------------------------------------

@dataclass
class TinyConfig:
    """Minimal hyperparameters for fast CPU unit tests.
    Matches the sizes specified in experiments.md Tier 0."""
    L: int = 16       # sequence length
    N: int = 4        # spline basis size
    M: int = 8        # quadrature nodes
    D: int = 16       # state width
    D_k: int = 8      # key dimension
    H: int = 1        # attention heads
    K: int = 4        # Hopfield patterns per modality
    beta: float = 1.0
    beta_HN: float = 1.0
    lambda_cross: float = 0.05
    lam_spline: float = 1e-3
    window: int = 3


@pytest.fixture
def tiny_cfg():
    return TinyConfig()


@pytest.fixture
def fd_grad():
    """Fixture exposing finite_difference_grad as a test helper."""
    return finite_difference_grad


@pytest.fixture
def check_grad():
    """Fixture exposing check_relative_error as a test helper."""
    return check_relative_error
