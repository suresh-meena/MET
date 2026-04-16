"""
met/core/basis.py
=================
Fixed continuous bases for compressed attention.

The writeup treats the basis family as an implementation choice. This module
keeps the downstream attention code agnostic to that choice by exposing one
cache interface:

    encode(K: B,H,L,D_k) -> C_bar: B,H,N,D_k, K_quad: B,H,M,D_k

Any fixed basis family that can build a token-position matrix F and a
quadrature-position matrix F_quad can plug into the same ridge-regression
projection:

    R = (F^T F + lambda I)^(-1) F^T
    C_bar = R K
    K_quad = F_quad C_bar
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class BasisSpec:
    """Configuration for a fixed continuous basis."""

    family: str = "bspline"
    num_basis: int = 16
    num_quadrature: int = 32
    ridge_lambda: float = 1e-3
    degree: int = 3


def _build_bspline_basis(t: np.ndarray, spec: BasisSpec) -> Tensor:
    """Evaluate a clamped B-spline basis at positions t in [0, 1]."""
    from scipy.interpolate import BSpline

    N = spec.num_basis
    degree = spec.degree
    internal = np.linspace(0.0, 1.0, N - degree + 1)
    knots = np.concatenate([[0.0] * degree, internal, [1.0] * degree])

    try:
        dm = BSpline.design_matrix(t, knots, degree)
        values = dm.toarray().astype(np.float32)
    except AttributeError:
        cols = []
        for i in range(N):
            c = np.zeros(N)
            c[i] = 1.0
            cols.append(BSpline(knots, c, degree)(t))
        values = np.stack(cols, axis=1).astype(np.float32)

    return torch.tensor(values, dtype=torch.float32)


def _build_fourier_basis(t: np.ndarray, spec: BasisSpec) -> Tensor:
    """
    Evaluate a truncated real Fourier basis at positions t in [0, 1].

    Basis ordering:
        1,
        sqrt(2) cos(2 pi t), sqrt(2) sin(2 pi t),
        sqrt(2) cos(4 pi t), sqrt(2) sin(4 pi t), ...
    """
    cols = [np.ones_like(t, dtype=np.float32)]
    freq = 1
    while len(cols) < spec.num_basis:
        angle = 2.0 * math.pi * freq * t
        cols.append((math.sqrt(2.0) * np.cos(angle)).astype(np.float32))
        if len(cols) < spec.num_basis:
            cols.append((math.sqrt(2.0) * np.sin(angle)).astype(np.float32))
        freq += 1

    values = np.stack(cols[: spec.num_basis], axis=1).astype(np.float32)
    return torch.tensor(values, dtype=torch.float32)


def _resolve_builder(family: str):
    family = family.lower()
    if family in {"bspline", "b_spline", "spline"}:
        return _build_bspline_basis
    if family in {"fourier", "fft"}:
        return _build_fourier_basis
    raise ValueError(
        f"Unsupported basis family '{family}'. "
        "Expected one of: bspline, fourier."
    )


def _gauss_legendre_on_unit(M: int) -> tuple[Tensor, Tensor]:
    """Gauss-Legendre nodes and weights mapped from [-1, 1] to [0, 1]."""
    from scipy.special import roots_legendre

    xi, wi = roots_legendre(M)
    t_quad = torch.tensor((xi + 1.0) / 2.0, dtype=torch.float32)
    w_quad = torch.tensor(wi / 2.0, dtype=torch.float32)
    return t_quad, w_quad


class FixedBasisCache(nn.Module):
    """
    Precompute projection operators for a fixed continuous basis family.

    Registered buffers:
        F:         (L, N) basis at token positions
        R:         (N, L) ridge projection operator
        F_quad:    (M, N) basis at quadrature positions
        t_quad:    (M,)   quadrature nodes
        w_quad:    (M,)   quadrature weights
        A:         (M, L) fused encode operator = F_quad @ R
        log_w_quad:(M,)   cached log quadrature weights
    """

    def __init__(self, L: int, spec: BasisSpec) -> None:
        super().__init__()
        self.L = L
        self.spec = spec

        build_basis = _resolve_builder(spec.family)

        t_tokens = np.linspace(0.0, 1.0, L)
        F = build_basis(t_tokens, spec)

        G = F.T @ F + spec.ridge_lambda * torch.eye(
            spec.num_basis, dtype=F.dtype
        )
        R = torch.linalg.solve(G, F.T)

        t_quad, w_quad = _gauss_legendre_on_unit(spec.num_quadrature)
        F_quad = build_basis(t_quad.numpy(), spec)
        A = F_quad @ R

        self.register_buffer("F", F)
        self.register_buffer("R", R)
        self.register_buffer("F_quad", F_quad)
        self.register_buffer("t_quad", t_quad)
        self.register_buffer("w_quad", w_quad)
        self.register_buffer("A", A)
        self.register_buffer("log_w_quad", w_quad.log())

    def encode(self, K: Tensor) -> tuple[Tensor, Tensor]:
        """Project keys onto the fixed basis and evaluate them at quadrature nodes."""
        K_quad = torch.einsum("ml,bhld->bhmd", self.A, K)
        C_bar = torch.einsum("nl,bhld->bhnd", self.R, K)
        return C_bar, K_quad

    def extra_repr(self) -> str:
        return (
            f"family={self.spec.family}, L={self.L}, "
            f"N={self.spec.num_basis}, M={self.spec.num_quadrature}, "
            f"lam={self.spec.ridge_lambda}"
        )


class BSplineCache(FixedBasisCache):
    """Backward-compatible B-spline cache wrapper."""

    def __init__(
        self,
        L: int,
        N: int,
        M: int,
        lam: float = 1e-3,
        degree: int = 3,
    ) -> None:
        super().__init__(
            L,
            BasisSpec(
                family="bspline",
                num_basis=N,
                num_quadrature=M,
                ridge_lambda=lam,
                degree=degree,
            ),
        )


class FourierBasisCache(FixedBasisCache):
    """Fixed Fourier basis cache using the same projection interface as B-splines."""

    def __init__(
        self,
        L: int,
        N: int,
        M: int,
        lam: float = 1e-3,
    ) -> None:
        super().__init__(
            L,
            BasisSpec(
                family="fourier",
                num_basis=N,
                num_quadrature=M,
                ridge_lambda=lam,
                degree=0,
            ),
        )


def build_basis_cache(
    L: int,
    family: str,
    N: int,
    M: int,
    lam: float = 1e-3,
    degree: int = 3,
) -> FixedBasisCache:
    """Factory for basis caches used by the MET energy core."""
    family = family.lower()
    if family in {"bspline", "b_spline", "spline"}:
        return BSplineCache(L=L, N=N, M=M, lam=lam, degree=degree)
    if family in {"fourier", "fft"}:
        return FourierBasisCache(L=L, N=N, M=M, lam=lam)
    raise ValueError(
        f"Unsupported basis family '{family}'. "
        "Expected one of: bspline, fourier."
    )


__all__ = [
    "BasisSpec",
    "FixedBasisCache",
    "BSplineCache",
    "FourierBasisCache",
    "build_basis_cache",
]
