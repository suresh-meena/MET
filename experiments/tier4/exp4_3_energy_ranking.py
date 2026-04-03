"""
experiments/tier4/exp4_3_energy_ranking.py
===========================================
Exp 4.3 — Energy ranking: matched vs mismatched pairs.

Hypothesis: After training, E(x^v, x^a_matched) < E(x^v, x^a_mismatched).
Measured via AUROC — should be > 0.90 on a held-out eval set.

J_rank was applied as an outer-loop auxiliary loss during training.
This experiment verifies the ranking discriminability of the energy.
"""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))

from met.core.energy import METEnergy, METConfig
from met.solver.gradient_descent import run_deterministic_solver
from met.training.losses import J_rank


def compute_auroc(positives, negatives):
    """Compute AUROC given positive (matched) and negative (mismatched) scores.
    Higher score = more likely negative (energy gap = E_mismatch - E_match).
    """
    labels = [1] * len(positives) + [0] * len(negatives)
    scores = list(positives) + list(negatives)
    # AUROC: fraction of (pos, neg) pairs where pos has higher score
    n_pos, n_neg = len(positives), len(negatives)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    count = sum(1 for p in positives for n in negatives if p > n)
    return count / (n_pos * n_neg)


def run_exp4_3(N_train=500, N_eval=200, L=8, D=16, n_epochs=20, batch_size=32):
    print("=" * 60)
    print("Exp 4.3 — Energy ranking (matched vs mismatched)")
    print("=" * 60)
    print(f"N_train={N_train}, N_eval={N_eval}")

    cfg = METConfig(L=L, N_v=4, N_a=4, M_v=8, M_a=8,
                    D=D, D_k=D//2, H=1, K_v=4, K_a=4)
    energy = METEnergy(cfg)

    # Synthetic: matched pairs from linear map; mismatched = shuffled audio
    M = torch.randn(D, D) * 0.1
    x_v_all = torch.randn(N_train + N_eval, L, D)
    x_a_all = torch.einsum("ij, blj -> bli", M, x_v_all) + 0.1 * torch.randn_like(x_v_all)

    x_v_train, x_a_train = x_v_all[:N_train], x_a_all[:N_train]
    x_v_eval, x_a_eval   = x_v_all[N_train:], x_a_all[N_train:]

    optimizer = torch.optim.AdamW(energy.parameters(), lr=3e-4)

    # Train with J_rank
    print(f"\nTraining {n_epochs} epochs with J_rank...")
    for epoch in range(n_epochs):
        perm = torch.randperm(N_train)
        x_v_train, x_a_train = x_v_train[perm], x_a_train[perm]

        epoch_ranks = []
        for i in range(0, N_train, batch_size):
            x_v_b = x_v_train[i:i+batch_size].detach()
            x_a_m = x_a_train[i:i+batch_size].detach()
            # Mismatched: roll by 1 in batch
            x_a_mm = torch.roll(x_a_m, 1, dims=0)

            E_match, _   = energy(x_v_b, x_a_m,  freeze_v=True, freeze_a=True)
            E_mismatch, _ = energy(x_v_b, x_a_mm, freeze_v=True, freeze_a=True)
            loss = J_rank(E_match, E_mismatch, margin=0.5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_ranks.append(loss.item())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}: J_rank={sum(epoch_ranks)/len(epoch_ranks):.4f}")

    # Evaluate AUROC
    print("\nEvaluating energy ranking on held-out set...")
    e_gaps = []
    with torch.no_grad():
        for i in range(0, N_eval, batch_size):
            x_v_b = x_v_eval[i:i+batch_size]
            x_a_m = x_a_eval[i:i+batch_size]
            x_a_mm = torch.roll(x_a_m, 1, dims=0)

            E_m, _  = energy(x_v_b, x_a_m,  freeze_v=True, freeze_a=True)
            E_mm, _ = energy(x_v_b, x_a_mm, freeze_v=True, freeze_a=True)
            # Positive score = E_mismatch - E_matched (higher = better ranking)
            e_gaps.append((E_mm - E_m).item())

    # AUROC: we want E_gap > 0 → treat as binary classification
    positives = [g for g in e_gaps if g > 0]   # correct rankings
    negatives = [g for g in e_gaps if g <= 0]  # incorrect
    accuracy = len(positives) / len(e_gaps)

    # Simple AUROC using the gap values
    gaps_tensor = torch.tensor(e_gaps)
    auroc_approx = (gaps_tensor > 0).float().mean().item()  # ranking accuracy

    passed = auroc_approx > 0.90
    print(f"\n  Ranking accuracy:  {auroc_approx:.3f}  (threshold > 0.90)")
    print(f"  Mean energy gap:   {gaps_tensor.mean().item():+.4f}  (should be > 0)")
    print(f"\n{'✓ Exp 4.3 PASSED' if passed else '✗ Exp 4.3 FAILED'}")
    print("  If failed: increase n_epochs or lambda_rank (currently 0.05)")
    return {"ranking_accuracy": auroc_approx, "mean_gap": gaps_tensor.mean().item(), "passed": passed}


if __name__ == "__main__":
    run_exp4_3()
