"""
experiments/tier4/exp4_2_av_symmetry.py
========================================
Exp 4.2 — AV-symmetry: reconstruction error in both directions.

Hypothesis: The joint energy handles both directions equally:
  - Conditioned on x^v (frozen), reconstruct x^a  →  E_v→a
  - Conditioned on x^a (frozen), reconstruct x^v  →  E_a→v
  - Asymmetry < 5% relative: |E_v→a - E_a→v| / E_v→a < 0.05

This validates that the bidirectional cross-modal term is truly symmetric.
"""

import sys
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))

from met.core.energy import METEnergy, METConfig
from met.solver.gradient_descent import run_deterministic_solver


def run_exp4_2(N=100, L=8, D=16, n_solver_steps=30, eta=0.01):
    print("=" * 60)
    print("Exp 4.2 — AV-symmetry: reconstruction error both directions")
    print("=" * 60)

    cfg = METConfig(L=L, N_v=4, N_a=4, M_v=8, M_a=8,
                    D=D, D_k=D//2, H=1, K_v=4, K_a=4)
    energy = METEnergy(cfg)
    energy.eval()

    M = torch.randn(D, D) * 0.1
    x_v_all = torch.randn(N, L, D)
    x_a_all = torch.einsum("ij, blj -> bli", M, x_v_all) + 0.05 * torch.randn_like(x_v_all)

    E_va_list = []  # freeze v, optimize a
    E_av_list = []  # freeze a, optimize v

    with torch.no_grad():
        for i in range(N):
            x_v = x_v_all[i:i+1]
            x_a = x_a_all[i:i+1]

    # v→a direction: freeze x_v, solve for x_a
    for i in range(N):
        x_v = x_v_all[i:i+1].detach()
        x_a = x_a_all[i:i+1].detach()

        x_a_init = torch.randn_like(x_a) * 0.1
        _, x_a_out, logs = run_deterministic_solver(
            energy, x_v, x_a_init,
            T=n_solver_steps, eta=eta, freeze_v=True
        )
        E_va, _ = energy(x_v, x_a_out, freeze_v=True, freeze_a=True)
        E_va_list.append(E_va.item())

    # a→v direction: freeze x_a, solve for x_v
    for i in range(N):
        x_v = x_v_all[i:i+1].detach()
        x_a = x_a_all[i:i+1].detach()

        x_v_init = torch.randn_like(x_v) * 0.1
        _, x_v_out, logs = run_deterministic_solver(
            energy, x_v_init, x_a,
            T=n_solver_steps, eta=eta, freeze_a=True
        )
        E_av, _ = energy(x_v_out, x_a, freeze_v=True, freeze_a=True)
        E_av_list.append(E_av.item())

    E_va_mean = sum(E_va_list) / N
    E_av_mean = sum(E_av_list) / N
    asymmetry = abs(E_va_mean - E_av_mean) / (abs(E_va_mean) + 1e-10)

    passed = asymmetry < 0.05
    print(f"\n  E(freeze_v → solve_a) mean: {E_va_mean:.4f}")
    print(f"  E(freeze_a → solve_v) mean: {E_av_mean:.4f}")
    print(f"  Asymmetry: {asymmetry:.3f}  (threshold < 0.05)")
    print(f"\n{'✓ Exp 4.2 PASSED' if passed else '✗ Exp 4.2 FAILED — energy is asymmetric'}")
    return {"E_va": E_va_mean, "E_av": E_av_mean, "asymmetry": asymmetry, "passed": passed}


if __name__ == "__main__":
    run_exp4_2()
