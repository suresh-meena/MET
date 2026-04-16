"""
Backward-compatible exports for fixed basis caches.

Historically MET only exposed B-spline compression from this module. The
implementation now supports multiple fixed basis families through
`met.core.basis`, while keeping the old import path stable.
"""

from met.core.basis import (
    BasisSpec,
    FixedBasisCache,
    BSplineCache,
    FourierBasisCache,
    build_basis_cache,
)

__all__ = [
    "BasisSpec",
    "FixedBasisCache",
    "BSplineCache",
    "FourierBasisCache",
    "build_basis_cache",
]
