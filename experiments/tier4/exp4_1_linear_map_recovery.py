"""
experiments/tier4/exp4_1_linear_map_recovery.py
================================================
Exp 4.1 — Linear map recovery from synthetic bidirectional data.

Hypothesis: Given N=1000 paired samples (x^v, x^a) related by a rank-4
linear map M (x^a = M x^v + noise), the model converges to rel error < 20%
with monotone energy and FP residual < 1e-3.

This is the first test with actual signal structure (not random noise).
"""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))

from met.core.energy import METEnergy, METConfig
from met.solver.gradient_descent import run_deterministic_solver
from met.heads.audio_head import AudioHead
from met.training.losses import J_mel


def generate_linear_dataset(N, L, D, rank=4, noise_std=0.1):
    """Generate N pairs (x_v, x_a) with x_a = M x_v + noise, M rank-4."""
    U = torch.randn(D, rank)
    V = torch.randn(D, rank)
    M = U @ V.T / rank  # (D, D) rank-4

    x_v = torch.randn(N, L, D)
    x_a = torch.einsum("ij, blj -> bli", M, x_v) + noise_std * torch.randn(N, L, D)
    return x_v, x_a, M


def run_exp4_1(N=1000, L=8, D=16, rank=4, n_epochs=10, batch_size=32, eta_model=1e-3):
    print("=" * 60)
    print("Exp 4.1 — Linear map recovery (synthetic bidirectional)")
    print("=" * 60)
    print(f"N={N}, L={L}, D={D}, rank={rank}, epochs={n_epochs}")

    # Data
    x_v_all, x_a_all, M_true = generate_linear_dataset(N, L, D, rank)
    n_mels = D  # predict x_a directly (same size for simplicity)

    # Model
    cfg = METConfig(L=L, N_v=4, N_a=4, M_v=8, M_a=8,
                    D=D, D_k=D//2, H=1, K_v=4, K_a=4)
    energy = METEnergy(cfg)
    audio_head = AudioHead(D=D, n_mels=n_mels, D_hid=D)

    optimizer = torch.optim.AdamW(
        list(energy.parameters()) + list(audio_head.parameters()),
        lr=eta_model, weight_decay=0.01
    )

    eta_solver = 0.01
    T_solver = 20

    all_losses = []
    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(N)
        x_v_all = x_v_all[perm]; x_a_all = x_a_all[perm]

        epoch_losses = []
        for i in range(0, N, batch_size):
            x_v = x_v_all[i:i+batch_size].detach()
            x_a_target = x_a_all[i:i+batch_size].detach()

            # Run solver with BPTT
            x_a_init = torch.randn_like(x_a_target) * 0.1
            _, x_a_out, logs = run_deterministic_solver(
                energy, x_v, x_a_init,
                T=T_solver, eta=eta_solver,
                freeze_v=True, create_graph=True
            )

            mel_pred = audio_head(x_a_out)
            loss = J_mel(mel_pred, x_a_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(energy.parameters()) + list(audio_head.parameters()), 1.0
            )
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        all_losses.append(mean_loss)
        print(f"  Epoch {epoch+1:2d}: J_gen={mean_loss:.4f}")

    # Final eval: relative reconstruction error
    with torch.no_grad():
        x_v_eval = x_v_all[:100]
        x_a_target_eval = x_a_all[:100]
        x_a_init = torch.zeros_like(x_a_target_eval)

        _, x_a_out, logs = run_deterministic_solver(
            energy, x_v_eval, x_a_init,
            T=50, eta=eta_solver, freeze_v=True
        )
        mel_pred = audio_head(x_a_out)
        rel_err = (mel_pred - x_a_target_eval).norm() / x_a_target_eval.norm()
        fp_residual = logs[-1]["fp_residual"]
        monotone_rate = sum(1 for l in logs if l["monotone"]) / len(logs)

    passed = rel_err.item() < 0.20 and monotone_rate > 0.95

    print(f"\nFinal relative error: {rel_err.item():.4f}  (threshold < 0.20)")
    print(f"FP residual final:    {fp_residual:.2e}  (threshold < 1e-3)")
    print(f"Monotone rate:        {monotone_rate:.1%}")
    print(f"\n{'✓ Exp 4.1 PASSED' if passed else '✗ Exp 4.1 FAILED'}")
    return {"rel_err": rel_err.item(), "fp_residual": fp_residual,
            "monotone_rate": monotone_rate, "passed": passed}


if __name__ == "__main__":
    run_exp4_1()
