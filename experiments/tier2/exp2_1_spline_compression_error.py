"""
experiments/tier2/exp2_1_spline_compression_error.py
=====================================================
Exp 2.1 — B-Spline compression error for localized signals.

Hypothesis: A localized spike at L=128 is compressed with relative
error < 5% at N=16 B-spline basis functions.

Also measures: how compression error scales with N (should decrease).
"""

import sys
import torch
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))

from met.core.spline import BSplineCache


def compression_error(cache: BSplineCache, signal: torch.Tensor) -> float:
    """Relative L2 reconstruction error via ridge regression."""
    # signal: (L, D_k) — treat as single-head single-batch key
    K = signal.unsqueeze(0).unsqueeze(0)        # (1, 1, L, D_k)
    C_bar, K_quad = cache.encode(K)

    # Reconstruct at token positions using F @ R @ K = F @ C_bar
    # Reconstruction = F @ C_bar: (L, N) x (N, D_k)
    C = C_bar[0, 0]                              # (N, D_k)
    K_recon = cache.F @ C                        # (L, D_k)

    err = (K_recon - signal).norm() / signal.norm()
    return err.item()


def run_exp2_1(L=128, D_k=1, n_basis_values=None, spike_pos=None):
    print("=" * 60)
    print("Exp 2.1 — B-Spline compression error (localized spike)")
    print("=" * 60)
    if n_basis_values is None:
        n_basis_values = [4, 8, 16, 32, 64]
    if spike_pos is None:
        spike_pos = L // 2

    # Localized spike signal
    signal = torch.zeros(L, D_k)
    signal[spike_pos, 0] = 1.0
    print(f"Signal: spike at position {spike_pos}/{L}")

    print(f"\n{'N (basis)':>12}  {'Rel L2 error':>14}  {'Status':>8}")
    print("-" * 40)

    results = {}
    for N in n_basis_values:
        if N >= L:
            continue  # skip degenerate case
        try:
            cache = BSplineCache(L=L, N=N, M=N, lam=1e-3)
            err = compression_error(cache, signal)
            results[N] = err
            threshold = 0.05
            passed = err < threshold
            status = "✓" if passed else "✗"
            print(f"{N:>12d}  {err:>14.4f}  {status}")
        except Exception as e:
            print(f"{N:>12d}  ERROR: {e}")

    print()
    n16_err = results.get(16, float("inf"))
    passed = n16_err < 0.05
    print(f"N=16 error: {n16_err:.4f}  (threshold: < 5%)")
    print(f"\n{'✓ Exp 2.1 PASSED' if passed else '✗ Exp 2.1 FAILED — increase N or use local basis'}")
    return results


if __name__ == "__main__":
    run_exp2_1()
