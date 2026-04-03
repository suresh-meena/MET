"""
experiments/tier2/exp2_2_spline_vs_fourier_ringing.py
======================================================
Exp 2.2 — B-spline vs Fourier ringing near transient events.

Hypothesis: B-spline error within ±3 tokens of a spike is < 50%
of the error of a truncated Fourier basis at the same N.

This validates the paper's claim that B-splines are preferable for
audio transients (onset clicks, percussive events).
"""

import sys
import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))

from met.core.spline import BSplineCache


def fourier_reconstruction(signal_np: np.ndarray, N: int) -> np.ndarray:
    """Truncated Fourier series reconstruction at N/2 harmonics."""
    L = len(signal_np)
    freqs = np.fft.rfft(signal_np)
    n_keep = min(N // 2 + 1, len(freqs))
    truncated = np.zeros_like(freqs)
    truncated[:n_keep] = freqs[:n_keep]
    return np.fft.irfft(truncated, n=L)


def run_exp2_2(L=128, N=16, spike_pos=None, window=3):
    print("=" * 60)
    print("Exp 2.2 — B-spline vs Fourier: ringing near spike")
    print("=" * 60)
    spike_pos = spike_pos or L // 2

    # Localized spike
    signal = np.zeros(L)
    signal[spike_pos] = 1.0

    # Fourier reconstruction
    fourier_recon = fourier_reconstruction(signal, N)

    # B-spline reconstruction
    cache = BSplineCache(L=L, N=N, M=N, lam=1e-3)
    sig_t = torch.tensor(signal, dtype=torch.float32).unsqueeze(-1)  # (L, 1)
    K = sig_t.unsqueeze(0).unsqueeze(0)   # (1, 1, L, 1)
    C_bar, _ = cache.encode(K)
    C = C_bar[0, 0]                         # (N, 1)
    spline_recon = (cache.F @ C).squeeze().detach().numpy()  # (L,)

    # Local error ±window tokens around spike
    lo = max(0, spike_pos - window)
    hi = min(L, spike_pos + window + 1)

    spline_local_err = np.abs(spline_recon[lo:hi] - signal[lo:hi]).mean()
    fourier_local_err = np.abs(fourier_recon[lo:hi] - signal[lo:hi]).mean()

    ratio = spline_local_err / (fourier_local_err + 1e-10)
    passed = ratio < 0.50

    print(f"Spike at position {spike_pos}/{L}, N={N}, window=±{window}")
    print(f"\n  B-spline local MAE:  {spline_local_err:.4f}")
    print(f"  Fourier local MAE:   {fourier_local_err:.4f}")
    print(f"  Ratio (spline/Fourier): {ratio:.3f}  (threshold: < 0.50)")

    # Global error
    spline_global = np.abs(spline_recon - signal).mean()
    fourier_global = np.abs(fourier_recon - signal).mean()
    print(f"\n  B-spline global MAE: {spline_global:.4f}")
    print(f"  Fourier global MAE:  {fourier_global:.4f}")

    print(f"\n{'✓ Exp 2.2 PASSED — B-spline has less ringing than Fourier' if passed else '✗ Exp 2.2 FAILED'}")
    print("  (If failed: try lower lambda or higher N for B-spline)")

    return {"ratio": ratio, "spline_local": spline_local_err,
            "fourier_local": fourier_local_err, "passed": passed}


if __name__ == "__main__":
    run_exp2_2()
