"""
experiments/tier0/exp0_2_grad_check_cross.py
=============================================
Exp 0.2 — Cross-modal gradient check (both directions).

From experiments.md:
    Two modalities, L=16, N=4, H=1, D_v=D_a=16
    Freeze modality a; FD grad_v E_cross vs autodiff
    Repeat with modality v frozen
    Success: relative error < 1e-3 in both directions
"""

import sys
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))

from met.core.energy import METEnergy, METConfig
from met.utils.grad_check import check_grad


def _cross_only_energy(energy, g_v, g_a):
    """Extract only the cross-modal attention energy."""
    g_v_n = energy.ln_v(g_v)
    g_a_n = energy.ln_a(g_a)
    _, _, E_cross = energy.attention(g_v_n, g_a_n)
    return E_cross


def run_exp0_2():
    print("=" * 60)
    print("Exp 0.2 — Cross-modal gradient check (both directions)")
    print("=" * 60)

    cfg = METConfig(L=8, N_v=4, N_a=4, M_v=8, M_a=8, D=8, D_k=4, H=1, K_v=2, K_a=2)
    energy = METEnergy(cfg).double()
    energy.eval()

    g_v = torch.randn(1, cfg.L, cfg.D, dtype=torch.float64).detach()
    g_a = torch.randn(1, cfg.L, cfg.D, dtype=torch.float64).detach()

    results = {}

    # ---- Direction 1: freeze g_a, FD g_v -------------------------
    print("\n[Direction 1] Freeze g_a, check ∇_{g_v} E_cross")
    errors_v = []
    for ell in range(cfg.L):
        def fn_v(x_ell):
            g_v_copy = g_v.clone()
            g_v_copy[0, ell] = x_ell
            return _cross_only_energy(energy, g_v_copy, g_a)

        rel_err, passed = check_grad(fn_v, g_v[0, ell].detach(),
                                      threshold=1e-3, label=f"v ℓ={ell}")
        errors_v.append((ell, rel_err, passed))

    all_v = all(p for _, _, p in errors_v)
    max_v = max(e for _, e, _ in errors_v)
    print(f"  Max rel error (∇_v): {max_v:.2e}  {'PASS ✓' if all_v else 'FAIL ✗'}")
    results["direction_v"] = all_v

    # ---- Direction 2: freeze g_v, FD g_a -------------------------
    print("\n[Direction 2] Freeze g_v, check ∇_{g_a} E_cross")
    errors_a = []
    for ell in range(cfg.L):
        def fn_a(x_ell):
            g_a_copy = g_a.clone()
            g_a_copy[0, ell] = x_ell
            return _cross_only_energy(energy, g_v, g_a_copy)

        rel_err, passed = check_grad(fn_a, g_a[0, ell].detach(),
                                      threshold=1e-3, label=f"a ℓ={ell}")
        errors_a.append((ell, rel_err, passed))

    all_a = all(p for _, _, p in errors_a)
    max_a = max(e for _, e, _ in errors_a)
    print(f"  Max rel error (∇_a): {max_a:.2e}  {'PASS ✓' if all_a else 'FAIL ✗'}")
    results["direction_a"] = all_a

    # ---- Summary ----
    overall = all_v and all_a
    print(f"\n{'✓ Exp 0.2 PASSED' if overall else '✗ Exp 0.2 FAILED'}")
    if not overall:
        print("  Stop. Fix cross-modal key-path gradient before proceeding.")
        print("  Common issue: reverse-direction key-path double-counted")
        print("  or stop-gradient boundary misplaced.")
        sys.exit(1)
    else:
        print("  Proceed to Exp 0.3 (energy monotonicity).")

    return overall


if __name__ == "__main__":
    run_exp0_2()
