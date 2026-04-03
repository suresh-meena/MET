"""
experiments/tier0/exp0_3_energy_monotonicity.py
===============================================
Exp 0.3 — Energy monotonicity under gradient descent.

From experiments.md:
    Single modality, L=32, N=8, random init
    Run 200 gradient descent steps, log E at each step
    Repeat for eta ∈ {0.1, 0.01, 0.001}
    Success: strict decrease at all steps for smallest eta

Also logs LayerNorm null-space events (Ė≈0 but ∇_g E ≠ 0)
to empirically demonstrate the Lyapunov gap from §3.3.
"""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))

from met.core.energy import METEnergy, METConfig
from met.utils.diagnostics import layernorm_null_events


def run_exp0_3_for_eta(energy, x_v, x_a, eta, n_steps=200):
    """Run deterministic descent for n_steps; return diagnostics."""
    x_a = x_a.clone()
    E_trace = []
    null_events = 0
    violations = 0

    E_prev = None
    for t in range(n_steps):
        x_a_r = x_a.clone().requires_grad_(True)
        E, comps = energy(x_v, x_a_r, freeze_v=True)
        grad_a, = torch.autograd.grad(E, x_a_r)

        E_val = E.item()
        E_trace.append(E_val)

        # Monotonicity check
        if E_prev is not None and E_val > E_prev + 1e-8:
            violations += 1

        # LayerNorm null-space event detection
        # Ė ≈ (E_t - E_{t-1}) / eta
        if E_prev is not None:
            E_dot_approx = (E_val - E_prev) / eta
            if layernorm_null_events(grad_a, E_dot_approx):
                null_events += 1

        E_prev = E_val

        with torch.no_grad():
            x_a = x_a - eta * grad_a.clamp(-5.0, 5.0)

    n = len(E_trace)
    monotone_rate = 1.0 - violations / max(n - 1, 1)
    return {
        "E_initial":         E_trace[0],
        "E_final":           E_trace[-1],
        "E_delta":           E_trace[-1] - E_trace[0],
        "violations":        violations,
        "monotone_rate":     monotone_rate,
        "null_space_events": null_events,
        "fp_residual_final": grad_a.norm().item(),
        "E_trace":           E_trace,
    }


def run_exp0_3():
    print("=" * 60)
    print("Exp 0.3 — Energy monotonicity under gradient descent")
    print("=" * 60)

    cfg = METConfig(L=32, N_v=8, N_a=8, M_v=16, M_a=16,
                    D=32, D_k=16, H=1, K_v=8, K_a=8)
    energy = METEnergy(cfg)
    energy.eval()

    x_v = torch.randn(1, 32, 32).detach()  # frozen conditioning
    x_a_init = torch.randn(1, 32, 32)

    etas = [0.1, 0.01, 0.001]
    results = {}

    for eta in etas:
        print(f"\n--- eta = {eta} ---")
        res = run_exp0_3_for_eta(energy, x_v, x_a_init.clone(), eta, n_steps=200)
        results[eta] = res

        print(f"  E_initial:         {res['E_initial']:.4f}")
        print(f"  E_final:           {res['E_final']:.4f}")
        print(f"  E_delta:           {res['E_delta']:+.4f}  (negative = descended)")
        print(f"  Monotone rate:     {res['monotone_rate']:.1%}")
        print(f"  Violations:        {res['violations']}")
        print(f"  Null-space events: {res['null_space_events']}  (LayerNorm Lyapunov gap)")
        print(f"  FP residual final: {res['fp_residual_final']:.4e}")

    # Success criterion
    smallest_eta = min(etas)
    passed = results[smallest_eta]["violations"] == 0

    print(f"\n{'✓ Exp 0.3 PASSED' if passed else '✗ Exp 0.3 FAILED'}")
    print(f"  At eta={smallest_eta}: monotone rate = {results[smallest_eta]['monotone_rate']:.1%}")

    if not passed:
        print("\n  The LayerNorm null-space issue is active in the joint system.")
        print("  Isolate which component (attention vs Hopfield) breaks monotonicity")
        print("  by disabling each in turn (use energy.forward_attention_only).")
        # Don't exit — this is a diagnostic experiment, not always a hard blocker
        # But the paper says: if non-monotone at ALL step sizes, it's a live problem
        all_eta_fail = all(results[e]["violations"] > 0 for e in etas)
        if all_eta_fail:
            print("  ALL step sizes show violations. Stop and investigate.")
    else:
        print("  Proceed to Tier 1 (Hopfield unit tests).")

    return results


if __name__ == "__main__":
    run_exp0_3()
