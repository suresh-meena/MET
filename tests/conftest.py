"""
tests/conftest.py
=================
Shared fixtures for robust, deterministic tests against real `met.*` modules.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from met.core.energy import METConfig, METEnergy
from met.utils.grad_check import finite_difference_grad


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Seed every test for reproducible, non-flaky assertions."""
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TinyConfig:
    L: int = 8
    D: int = 12
    D_k: int = 4
    H: int = 2
    N: int = 4
    M: int = 6
    K: int = 5
    beta: float = 1.0
    beta_HN: float = 1.0
    lambda_cross: float = 0.05
    lam_spline: float = 1e-3
    window: int = 3


@pytest.fixture
def tiny_cfg() -> TinyConfig:
    return TinyConfig()


@pytest.fixture
def tiny_met_cfg(tiny_cfg: TinyConfig) -> METConfig:
    return METConfig(
        L=tiny_cfg.L,
        D=tiny_cfg.D,
        D_k=tiny_cfg.D_k,
        H=tiny_cfg.H,
        N_v=tiny_cfg.N,
        N_a=tiny_cfg.N,
        M_v=tiny_cfg.M,
        M_a=tiny_cfg.M,
        K_v=tiny_cfg.K,
        K_a=tiny_cfg.K,
        beta=tiny_cfg.beta,
        beta_HN=tiny_cfg.beta_HN,
        lambda_cross=tiny_cfg.lambda_cross,
        lam_spline=tiny_cfg.lam_spline,
        window=tiny_cfg.window,
    )


@pytest.fixture
def tiny_energy(tiny_met_cfg: METConfig) -> METEnergy:
    return METEnergy(tiny_met_cfg)


@pytest.fixture
def fd_grad():
    return finite_difference_grad


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm().item() / (b.norm().item() + 1e-12))
