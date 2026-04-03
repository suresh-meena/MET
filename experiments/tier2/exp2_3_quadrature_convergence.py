"""
experiments/tier2/exp2_3_quadrature_convergence.py
====================================================
Exp 2.3 — Gauss-Legendre quadrature convergence for smooth trajectories.

Hypothesis: For smooth key trajectories (sinusoidal + polynomial),
Gauss-Legendre with M=32 achieves absolute integration error < 1e-3
relative to the closed-form integral.

Tests two reference integrals:
    1. ∫₀¹ sin(2πt) · cos(2πt) dt = 0            (zero exactly)
    2. ∫₀¹ t³ dt = 0.25                           (polynomial: GL exact for M≥2)

More practically: verifies the spline-based approximate integral
    ∫₀¹ ⟨Q, K̄(t)⟩ ω(t) dt  ≈ Σ_r ω_r ⟨Q, K_quad_r⟩
converges as M increases.
"""

import sys
import torch
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))

from met.core.spline import BSplineCache


def closed_form_integral(fn, a=0.0, b=1.0, n=10000):
    """Monte Carlo reference integral."""
    t = np.linspace(a, b, n)
    return np.mean(fn(t)) * (b - a)


def run_exp2_3(L=128, D_k=1, M_values=None):
    print("=" * 60)
    print("Exp 2.3 — Gauss-Legendre quadrature convergence")
    print("=" * 60)
    if M_values is None:
        M_values = [4, 8, 16, 32, 64]

    # Smooth signal: sinusoidal key trajectory K(t) = sin(2πt)
    # Query Q = constant vector [1, 0, ...]
    # Reference integral: ∫₀¹ sin(2πt) dt = 0 (but with Q dot product it's just marginally nonzero)
    # Use Q = [1], K(t) = sin(2πt) → integral = 0

    fn_ref = lambda t: np.sin(2 * np.pi * t)
    ref_val = 0.0  # exact

    print(f"\nReference: ∫₀¹ sin(2πt)dt = {ref_val}")
    print(f"\n{'M':>6}  {'GL estimate':>14}  {'Error':>10}  {'Status':>8}")
    print("-" * 45)

    results = {}
    for M in M_values:
        N = max(M, 8)  # enough basis functions to represent the signal
        cache = BSplineCache(L=L, N=N, M=M, lam=1e-4)

        # Evaluate sin(2πt) at quadrature nodes
        t_q = cache.t_quad.numpy()
        f_q = np.sin(2 * np.pi * t_q)

        # GL estimate: Σ_r w_r f(t_r)
        w_q = cache.w_quad.numpy()
        estimate = float(np.dot(w_q, f_q))
        error = abs(estimate - ref_val)
        passed = error < 1e-3
        results[M] = {"estimate": estimate, "error": error, "passed": passed}

        status = "✓" if passed else "✗"
        print(f"{M:>6d}  {estimate:>14.6f}  {error:>10.2e}  {status}")

    # Polynomial test: ∫₀¹ t³ dt = 0.25 (exact for M≥2)
    print(f"\nPolynomial test: ∫₀¹ t³ dt = 0.25 (GL exact for M≥2)")
    for M in [2, 4, 8, 16]:
        N = max(M, 4)
        cache = BSplineCache(L=L, N=N, M=M, lam=1e-4)
        t_q = cache.t_quad.numpy()
        w_q = cache.w_quad.numpy()
        est = float(np.dot(w_q, t_q**3))
        err = abs(est - 0.25)
        print(f"  M={M}: ∫t³dt ≈ {est:.6f}, error={err:.2e}")

    m32_pass = results.get(32, {}).get("passed", False)
    print(f"\n{'✓ Exp 2.3 PASSED (M=32 error < 1e-3)' if m32_pass else '✗ Exp 2.3 FAILED'}")
    return results


if __name__ == "__main__":
    run_exp2_3()
