"""
experiments/tier1/exp1_1_hopfield_completion.py
================================================
Exp 1.1 — Unimodal pattern completion.

Hypothesis: With lambda_cross=0 and beta_HN=5.0,
Hopfield gradient descent from a noisy initial state converges
to the nearest stored prototype with cosine similarity > 0.95.

Setup: D=64, K=5 random unit-norm prototypes, L=8, lambda_cross=0
Success: cosine similarity of final state to nearest prototype > 0.95
"""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))

from met.core.hopfield import HopfieldMemoryBank


def run_exp1_1(D=64, K=5, L=8, beta_HN=5.0, n_steps=200, eta=0.05, n_trials=10):
    print("=" * 60)
    print("Exp 1.1 — Unimodal Hopfield pattern completion")
    print("=" * 60)
    print(f"D={D}, K={K}, L={L}, beta_HN={beta_HN}, eta={eta}, n_steps={n_steps}")

    bank = HopfieldMemoryBank(D, D, K, beta_HN=beta_HN, lambda_cross=0.0, window=1)
    # Force lambda=0 (no cross-modal blending)
    with torch.no_grad():
        bank.lambda_cross.fill_(0.0)

    g_cross_dummy = torch.zeros(1, L, D)  # unused when lambda=0
    successes = []

    for trial in range(n_trials):
        # Pick a random prototype as target
        proto_idx = trial % K
        xi_target = bank.Xi[proto_idx].detach()

        # Noisy initialization: prototype + Gaussian noise (SNR ~ 0.3)
        g_m = (xi_target + 0.3 * torch.randn(L, D)).unsqueeze(0)  # (1, L, D)

        # Gradient descent
        for _ in range(n_steps):
            g_r = g_m.clone().requires_grad_(True)
            E = bank(g_r, g_cross_dummy)
            grad, = torch.autograd.grad(E, g_r)
            with torch.no_grad():
                g_m = g_m - eta * grad

        # Cosine similarity between mean-pooled final state and target prototype
        g_final = F.normalize(g_m[0].mean(dim=0, keepdim=True), dim=-1)  # (1, D)
        Xi_n = F.normalize(bank.Xi.detach(), dim=-1)  # (K, D)
        sims = (g_final @ Xi_n.T).squeeze()  # (K,)
        best_sim = sims.max().item()
        nearest = sims.argmax().item()

        successes.append(best_sim)
        status = "✓" if best_sim > 0.95 else "✗"
        print(f"  Trial {trial+1:2d}: target=ξ_{proto_idx}, "
              f"nearest=ξ_{nearest}, cosine={best_sim:.4f}  {status}")

    mean_sim = sum(successes) / len(successes)
    n_pass = sum(1 for s in successes if s > 0.95)
    print(f"\nMean cosine similarity: {mean_sim:.4f}")
    print(f"Pass rate (>0.95): {n_pass}/{n_trials}")

    passed = n_pass >= int(0.9 * n_trials)  # 90% of trials must pass
    print(f"\n{'✓ Exp 1.1 PASSED' if passed else '✗ Exp 1.1 FAILED'}")
    return passed


if __name__ == "__main__":
    run_exp1_1()
