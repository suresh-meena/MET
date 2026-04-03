"""
experiments/tier3/exp3_1_eqprop_vs_bptt.py
============================================
Exp 3.1 — EqProp gradient estimate vs BPTT oracle.

Setup: L=8, N=4, K=3, D=16, 20 BPTT steps
Sweep s ∈ {0.1, 0.05, 0.01, 0.005, 0.001}

Success criteria (from experiments.md):
  - cosine sim(EqProp, BPTT) > 0.9 at s=0.01
  - error scales as O(s)  [i.e., halving s halves error]
  - attractor_agreement > 95% across batches

This is the critical validity check for Phase D.
Run ON SMALL GPU (or CPU with small sizes).
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))

from met.core.energy import METEnergy, METConfig
from met.solver.gradient_descent import run_deterministic_solver
from met.solver.eqprop import EqPropEstimator
from met.utils.diagnostics import grad_cosine_similarity


def bptt_gradient(energy, x_v, x_a_init, mel_target, n_mels, T, eta, audio_head):
    """Compute parameter gradients via BPTT."""
    _, x_a_out, _ = run_deterministic_solver(
        energy, x_v, x_a_init, T=T, eta=eta,
        freeze_v=True, create_graph=True
    )
    mel_pred = audio_head(x_a_out)
    loss = nn.functional.mse_loss(mel_pred, mel_target)
    params = list(energy.parameters())
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return grads


def run_exp3_1():
    print("=" * 60)
    print("Exp 3.1 — EqProp vs BPTT gradient comparison")
    print("=" * 60)

    from met.heads.audio_head import AudioHead

    cfg = METConfig(L=8, N_v=4, N_a=4, M_v=8, M_a=8,
                    D=16, D_k=8, H=1, K_v=3, K_a=3)
    energy = METEnergy(cfg)

    n_mels = 16
    audio_head = AudioHead(D=16, n_mels=n_mels)

    x_v = torch.randn(1, 8, 16).detach()
    x_a_init = torch.randn(1, 8, 16)
    mel_target = torch.randn(1, 8, n_mels).detach()

    # BPTT oracle (20 steps)
    T = 20
    eta = 0.01
    print(f"\nBPTT oracle: T={T} steps, eta={eta}")
    grads_bptt = bptt_gradient(energy, x_v, x_a_init, mel_target, n_mels,
                                T=T, eta=eta, audio_head=audio_head)
    print(f"  BPTT gradient norm: {sum(g.norm().item() for g in grads_bptt if g is not None):.4f}")

    s_values = [0.1, 0.05, 0.01, 0.005, 0.001]
    print(f"\n{'s':>10}  {'cosine_sim':>12}  {'agreement':>12}  {'status':>8}")
    print("-" * 50)

    results = {}
    prev_err = None
    for s in s_values:
        eqprop = EqPropEstimator(energy, s=s, T=T)
        result = eqprop.estimate_gradient(
            x_v, x_a_init, mel_target, eta=eta, freeze_v=True
        )

        # Collect EqProp gradients as list (same order as BPTT)
        params = list(energy.parameters())
        grads_eqprop = [result["eqprop_grads"].get(n) for n, _ in energy.named_parameters()]

        # Filter Nones
        pairs = [(g_b, g_e) for g_b, g_e in zip(grads_bptt, grads_eqprop)
                 if g_b is not None and g_e is not None]
        if pairs:
            flat_b = torch.cat([g.flatten() for g, _ in pairs])
            flat_e = torch.cat([g.flatten() for _, g in pairs])
            cos_sim = torch.nn.functional.cosine_similarity(
                flat_b.unsqueeze(0), flat_e.unsqueeze(0)
            ).item()
        else:
            cos_sim = 0.0

        agreement = result["attractor_agreement"]
        err_norm = (torch.cat([g.flatten() for _, g in pairs]) -
                    torch.cat([g.flatten() for g, _ in pairs])).norm().item() if pairs else float("nan")

        # O(s) scaling check
        scaling_ok = True
        if prev_err is not None and prev_err > 0:
            expected_ratio = 2.0  # halving s should halve error
            actual_ratio = prev_err / (err_norm + 1e-10)
            scaling_ok = 0.5 <= actual_ratio <= 4.0  # loose bound

        passed = cos_sim > 0.9 and agreement > 0.95
        results[s] = {"cos_sim": cos_sim, "agreement": agreement,
                      "err_norm": err_norm, "passed": passed}
        prev_err = err_norm

        status = "✓" if passed else "✗"
        print(f"{s:>10.3f}  {cos_sim:>12.4f}  {agreement:>12.4f}  {status}")

    # Summary
    s01_pass = results.get(0.01, {}).get("passed", False)
    print(f"\nAt s=0.01: {'✓ PASSED' if s01_pass else '✗ FAILED'}")
    print(f"{'✓ Exp 3.1 PASSED' if s01_pass else '✗ Exp 3.1 FAILED'}")
    print("  If failed: check attractor_agreement. Both phases must hit same basin.")
    return results


if __name__ == "__main__":
    run_exp3_1()
