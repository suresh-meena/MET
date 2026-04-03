"""
experiments/tier0/exp0_1_grad_check_intra.py
============================================
Exp 0.1 — Finite-difference intra-modal gradient check.

From experiments.md:
    Hypothesis: Autodiff gradients through the shared key-path in B-spline
    attention are correct, specifically including the all-query accumulation term.
    Setup: L=16, N=4, H=1, D_k=8, D=16, single modality
    Success: relative error < 1e-3 at all token positions

This is a MANDATORY pass/fail gate. If it fails, stop.
"""

import sys
import torch

# Allow running from MET/ root: python experiments/tier0/exp0_1_grad_check_intra.py
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))

from met.core.energy import METEnergy, METConfig
from met.utils.grad_check import check_grad, finite_difference_grad


def run_exp0_1():
    print("=" * 60)
    print("Exp 0.1 — Intra-modal gradient check (shared key-path)")
    print("=" * 60)

    cfg = METConfig(L=16, N_v=4, N_a=4, M_v=8, M_a=8,
                    D=16, D_k=8, H=1, K_v=4, K_a=4)
    energy = METEnergy(cfg).double()
    energy.eval()

    # Fix g_a (frozen) — test gradient w.r.t. g_v only
    g_a_fixed = torch.randn(1, 16, 16, dtype=torch.float64).detach()

    def energy_fn(g_v):
        """Energy as a function of g_v only (g_a fixed)."""
        E, _ = energy(g_v, g_a_fixed)
        return E

    g_v = torch.randn(1, 16, 16, dtype=torch.float64)

    print(f"\nTesting at all L={cfg.L} token positions...")
    errors = []
    for ell in range(cfg.L):
        def fn_ell(x_ell):
            g_v_copy = g_v.clone()
            g_v_copy[0, ell] = x_ell
            return energy_fn(g_v_copy)

        x_ell = g_v[0, ell].detach()
        rel_err, passed = check_grad(fn_ell, x_ell, eps=1e-4, threshold=1e-3,
                                      label=f"ℓ={ell}")
        errors.append((ell, rel_err, passed))
        status = "✓" if passed else "✗ FAILED"
        print(f"  ℓ={ell:2d}: rel_err={rel_err:.2e}  {status}")

    all_passed = all(p for _, _, p in errors)
    max_err = max(e for _, e, _ in errors)
    mean_err = sum(e for _, e, _ in errors) / len(errors)

    print(f"\n{'All tokens PASSED' if all_passed else 'SOME TOKENS FAILED'}")
    print(f"  Max relative error:  {max_err:.2e}  (threshold: 1e-3)")
    print(f"  Mean relative error: {mean_err:.2e}")

    if all_passed:
        print("\n✓ Exp 0.1 PASSED — shared key-path gradient is correct.")
        print("  Proceed to Exp 0.2 (cross-modal gradient check).")
    else:
        n_failed = sum(1 for _, _, p in errors if not p)
        print(f"\n✗ Exp 0.1 FAILED — {n_failed}/{cfg.L} token positions failed.")
        print("  STOP. Fix before proceeding.")
        print("  Most likely cause: cross-token key-path not accumulated.")
        print("  Use autograd through R_m @ K_h — never hand-derive.")
        sys.exit(1)

    return all_passed


if __name__ == "__main__":
    run_exp0_1()
