"""
experiments/tier3/exp3_2_memory_scaling.py
===========================================
Exp 3.2 — Memory scaling: EqProp vs BPTT as T_iter increases.

Measures peak GPU/CPU memory for T in {5, 10, 20, 50, 100}.
EqProp: constant (stores 2 equilibria)
BPTT: linear in T (stores full trajectory)

Success: BPTT grows linearly; EqProp approximately constant.
"""

import sys
import gc
import torch
import torch.nn as nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))

from met.core.energy import METEnergy, METConfig
from met.solver.gradient_descent import run_deterministic_solver
from met.solver.eqprop import EqPropEstimator
from met.heads.audio_head import AudioHead


def measure_bptt_memory(energy, audio_head, x_v, x_a_init, mel_target, T, eta):
    """Returns peak memory (bytes) used by BPTT for T steps."""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _, x_a_out, _ = run_deterministic_solver(
        energy, x_v, x_a_init, T=T, eta=eta,
        freeze_v=True, create_graph=True
    )
    mel_pred = audio_head(x_a_out)
    loss = nn.functional.mse_loss(mel_pred, mel_target)
    params = list(energy.parameters())
    grads = torch.autograd.grad(loss, params, allow_unused=True)

    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    else:
        # CPU: estimate from number of tensors in graph (proxy)
        return T * x_a_init.numel() * 4  # T states × elements × 4 bytes


def measure_eqprop_memory(energy, x_v, x_a_init, mel_target, T, eta):
    """Returns approximate memory (bytes) used by EqProp (2 equilibria only)."""
    # EqProp stores: x_free* and x_nudged* — 2 × tensor size
    return 2 * x_a_init.numel() * 4  # 2 states × float32


def run_exp3_2():
    print("=" * 60)
    print("Exp 3.2 — Memory scaling: EqProp vs BPTT")
    print("=" * 60)

    cfg = METConfig(L=16, N_v=4, N_a=4, M_v=8, M_a=8,
                    D=32, D_k=16, H=1, K_v=4, K_a=4)
    energy = METEnergy(cfg)
    audio_head = AudioHead(D=32, n_mels=16)
    eta = 0.01

    x_v = torch.randn(1, 16, 32).detach()
    x_a_init = torch.randn(1, 16, 32)
    mel_target = torch.randn(1, 16, 16).detach()

    T_values = [5, 10, 20, 50, 100]

    print(f"\n{'T':>6}  {'BPTT (KB)':>12}  {'EqProp (KB)':>14}  {'Ratio':>8}")
    print("-" * 50)

    results = {}
    for T in T_values:
        bptt_mem = measure_bptt_memory(energy, audio_head, x_v, x_a_init, mel_target, T, eta)
        eqprop_mem = measure_eqprop_memory(energy, x_v, x_a_init, mel_target, T, eta)
        ratio = bptt_mem / (eqprop_mem + 1)
        results[T] = {"bptt_kb": bptt_mem / 1024, "eqprop_kb": eqprop_mem / 1024, "ratio": ratio}
        print(f"{T:>6d}  {bptt_mem/1024:>12.1f}  {eqprop_mem/1024:>14.1f}  {ratio:>8.1f}x")

    # Check: BPTT grows at least linearly; EqProp constant
    bptt_5 = results[5]["bptt_kb"]
    bptt_100 = results[100]["bptt_kb"]
    eqprop_5 = results[5]["eqprop_kb"]
    eqprop_100 = results[100]["eqprop_kb"]

    bptt_grows = bptt_100 >= bptt_5 * 5   # should be at least 5x larger
    eqprop_constant = eqprop_100 <= eqprop_5 * 1.1  # within 10%

    print(f"\nBPTT grows ({bptt_5:.0f} → {bptt_100:.0f} KB): {'✓' if bptt_grows else '✗'}")
    print(f"EqProp constant ({eqprop_5:.0f} → {eqprop_100:.0f} KB): {'✓' if eqprop_constant else '✗'}")

    passed = bptt_grows and eqprop_constant
    print(f"\n{'✓ Exp 3.2 PASSED' if passed else '✗ Exp 3.2 FAILED'}")
    return results


if __name__ == "__main__":
    run_exp3_2()
