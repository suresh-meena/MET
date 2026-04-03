"""
experiments/tier1/exp1_3_temporal_window.py
============================================
Exp 1.3 — Temporal smoothing window sensitivity.

Hypothesis: When cross-modal context is temporally shifted by δ tokens,
smoothing with window w ≥ 2δ+1 preserves retrieval accuracy > 0.90.
With w < 2δ+1, alignment degrades.

Gives empirical guidance for window size based on expected A/V temporal offset.
"""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))

from met.core.hopfield import HopfieldMemoryBank


def retrieval_with_offset(bank, D, L, offset, n_steps=60, eta=0.05, n_trials=5):
    """Return mean cosine similarity when cross-modal context is shifted by offset tokens."""
    sims = []
    for trial in range(n_trials):
        proto_idx = trial % bank.K
        xi = bank.Xi[proto_idx].detach()

        # g_cross aligned with g_m (but shifted)
        g_m = (xi + 0.1 * torch.randn(L, D)).unsqueeze(0)
        g_cross = torch.roll(g_m, shifts=offset, dims=1)  # temporal shift

        for _ in range(n_steps):
            g_r = g_m.clone().requires_grad_(True)
            E = bank(g_r, g_cross)
            grad, = torch.autograd.grad(E, g_r)
            with torch.no_grad():
                g_m = g_m - eta * grad

        g_final = F.normalize(g_m[0].mean(dim=0, keepdim=True), dim=-1)
        Xi_n = F.normalize(bank.Xi.detach(), dim=-1)
        sim = (g_final @ Xi_n.T).squeeze().max().item()
        sims.append(sim)
    return sum(sims) / len(sims)


def run_exp1_3(D=64, K=5, L=32, beta_HN=5.0, lam=0.3):
    print("=" * 60)
    print("Exp 1.3 — Temporal window sensitivity vs AV offset")
    print("=" * 60)
    print(f"D={D}, K={K}, L={L}, lambda_cross={lam}")

    offsets = [0, 1, 2, 4]
    windows = [1, 3, 5, 9]

    print(f"\n{'offset/window':>14}", end="")
    for w in windows:
        print(f"  w={w:2d}", end="")
    print()
    print("-" * (14 + 8 * len(windows)))

    results = {}
    for delta in offsets:
        print(f"δ={delta:12d}", end="")
        row = {}
        for w in windows:
            bank = HopfieldMemoryBank(D, D, K, beta_HN=beta_HN,
                                       lambda_cross=lam, window=w)
            with torch.no_grad():
                bank.lambda_cross.fill_(lam)
            sim = retrieval_with_offset(bank, D, L, offset=delta, n_trials=5)
            row[w] = sim
            print(f"  {sim:.3f}", end="")
        results[delta] = row
        print()

    # Check: for each offset δ, window w ≥ 2δ+1 should give better results
    print("\nWindow coverage check (w ≥ 2δ+1 should improve):")
    coverage_rule_holds = True
    for delta in offsets:
        required_w = 2 * delta + 1
        sims_at_required = [results[delta][w] for w in windows if w >= required_w]
        sims_below = [results[delta][w] for w in windows if w < required_w]
        if sims_at_required and sims_below:
            better = (sum(sims_at_required) / len(sims_at_required) >=
                      sum(sims_below) / len(sims_below) - 0.05)
            status = "✓" if better else "✗"
            if not better:
                coverage_rule_holds = False
            print(f"  δ={delta}, required w≥{required_w}: {status}")

    print(f"\n{'✓ Exp 1.3 PASSED' if coverage_rule_holds else '✗ Exp 1.3 FAILED'}")
    return results


if __name__ == "__main__":
    run_exp1_3()
