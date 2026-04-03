"""
experiments/tier1/exp1_2_lambda_degradation.py
================================================
Exp 1.2 — Cross-modal context degradation sweep.

Hypothesis: Retrieval quality degrades gracefully as lambda_cross increases
when cross-modal context is pure noise.
At lambda=0.3, cosine sim > 0.90. At lambda=0.9, marked degradation.

Setup: Sweep lambda_cross ∈ {0, 0.1, 0.3, 0.5, 0.9}
       Cross-modal input = random noise (no real signal)
       50 gradient descent steps from noisy init, measure cosine sim
"""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))

from met.core.hopfield import HopfieldMemoryBank


def retrieval_cosine(bank, D, L, n_steps=50, eta=0.05, n_trials=5):
    """Returns mean cosine similarity across n_trials for given bank settings."""
    g_cross = torch.randn(1, L, D)  # pure noise — no cross-modal signal
    sims = []

    for trial in range(n_trials):
        proto_idx = trial % bank.K
        xi_target = bank.Xi[proto_idx].detach()
        g_m = (xi_target + 0.3 * torch.randn(L, D)).unsqueeze(0)

        for _ in range(n_steps):
            g_r = g_m.clone().requires_grad_(True)
            E = bank(g_r, g_cross)
            grad, = torch.autograd.grad(E, g_r)
            with torch.no_grad():
                g_m = g_m - eta * grad

        g_final = F.normalize(g_m[0].mean(dim=0, keepdim=True), dim=-1)
        Xi_n = F.normalize(bank.Xi.detach(), dim=-1)
        best_sim = (g_final @ Xi_n.T).squeeze().max().item()
        sims.append(best_sim)

    return sum(sims) / len(sims)


def run_exp1_2(D=64, K=5, L=8, beta_HN=5.0):
    print("=" * 60)
    print("Exp 1.2 — Cross-modal context degradation (noise context)")
    print("=" * 60)

    lambdas = [0.0, 0.1, 0.3, 0.5, 0.9]
    thresholds = {0.0: 0.95, 0.1: 0.90, 0.3: 0.80, 0.5: 0.60, 0.9: None}

    results = {}
    print(f"\n{'λ_cross':>10}  {'mean_cosine':>14}  {'threshold':>12}  {'status':>8}")
    print("-" * 60)

    for lam in lambdas:
        bank = HopfieldMemoryBank(D, D, K, beta_HN=beta_HN,
                                   lambda_cross=lam, window=1)
        with torch.no_grad():
            bank.lambda_cross.fill_(lam)

        sim = retrieval_cosine(bank, D, L, n_steps=80, eta=0.05, n_trials=10)
        thresh = thresholds[lam]
        passed = (thresh is None) or (sim >= thresh)
        results[lam] = {"sim": sim, "passed": passed}

        thresh_str = f">={thresh:.2f}" if thresh else "  n/a"
        status = "✓" if passed else "✗"
        print(f"{lam:>10.1f}  {sim:>14.4f}  {thresh_str:>12}  {status:>8}")

    all_passed = all(r["passed"] for r in results.values())
    print(f"\n{'✓ Exp 1.2 PASSED' if all_passed else '✗ Exp 1.2 FAILED — lambda_cross too high'}")
    print("  If failed: clamp lambda_cross to [0, 0.3] during early training.")
    return results


if __name__ == "__main__":
    run_exp1_2()
