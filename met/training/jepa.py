"""
met/training/jepa.py
====================
EMA teacher and predictor heads for the JEPA auxiliary objective.

Design (§5):
    θ̄ ← μ θ̄ + (1-μ) θ            (exponential moving average teacher)
    Teacher equilibrium: deterministic descent on unmasked inputs, stop-grad
    Student equilibrium: deterministic descent on masked inputs (M_v, M_a)
    Predictor heads: P_v, P_a map student equilibria → teacher latent space

Key invariants:
    - Teacher parameters require_grad = False (never backpropd through)
    - Stop-gradient applied to teacher equilibria BEFORE loss computation
    - Predictor heads are OUTSIDE the energy — thin MLP only
    - Collapse prevention: stop-gradient on teacher targets is the primary guard
      (no contrastive loss needed for JEPA-style architecture)

Masking strategy (Exp 5.4):
    Contiguous temporal masks M_v and M_a.
    Designed to force cross-modal infilling:
    e.g., mask audio during high-motion video segments and vice versa.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from met.core.energy import METEnergy
from met.solver.gradient_descent import run_deterministic_solver


class PredictorHead(nn.Module):
    """
    Lightweight MLP predictor: maps student equilibrium → teacher latent space.
    Per-token projection (no pooling here; pooling happens in the loss computation).
    """

    def __init__(self, D_in: int, D_out: int, D_hidden: int | None = None) -> None:
        super().__init__()
        D_hidden = D_hidden or D_in
        self.net = nn.Sequential(
            nn.LayerNorm(D_in),
            nn.Linear(D_in, D_hidden),
            nn.GELU(),
            nn.Linear(D_hidden, D_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L, D_in) → (B, L, D_out)"""
        return self.net(x)


class JEPATeacher(nn.Module):
    """
    EMA teacher for cross-modal masked prediction.

    Args:
        energy:   student METEnergy (teacher is a deepcopy)
        D:        state width (for predictor heads)
        D_pred:   predictor output dimension (default same as D)
        mu:       EMA decay (default 0.99)
    """

    def __init__(
        self,
        energy: METEnergy,
        D: int,
        D_pred: int | None = None,
        mu: float = 0.99,
    ) -> None:
        super().__init__()
        D_pred = D_pred or D
        self.mu = mu

        # Teacher: frozen deepcopy — NO grad
        self.teacher = copy.deepcopy(energy)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Thin predictor heads (outside the energy)
        self.pred_v = PredictorHead(D, D_pred)
        self.pred_a = PredictorHead(D, D_pred)

    @torch.no_grad()
    def update_ema(self, student: METEnergy) -> None:
        """
        Update teacher parameters: θ̄ ← μ θ̄ + (1−μ) θ
        Must be called after every optimizer step.
        """
        for tp, sp in zip(self.teacher.parameters(), student.parameters()):
            tp.data.mul_(self.mu).add_((1.0 - self.mu) * sp.data)

    def get_teacher_equilibrium(
        self,
        x_v: Tensor,
        x_a: Tensor,
        T: int,
        eta: float,
        grad_clip: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        """
        Run teacher deterministic solver on UNMASKED inputs.
        Returns stop-gradiented teacher equilibria.

        Args:
            x_v: (B, L, D)  unmasked video tokens
            x_a: (B, L, D)  unmasked audio tokens
        Returns:
            g_v_T: (B, L, D)  teacher video equilibrium (no grad)
            g_a_T: (B, L, D)  teacher audio equilibrium (no grad)
        """
        with torch.no_grad():
            x_v_T, x_a_T, _ = run_deterministic_solver(
                self.teacher, x_v, x_a, T=T, eta=eta, grad_clip=grad_clip
            )
        # Already detached (run_deterministic_solver returns detached)
        return x_v_T, x_a_T

    def compute_jepa_loss(
        self,
        student_eq_v: Tensor,  # (B, L, D)  student equilibrium on masked video
        student_eq_a: Tensor,  # (B, L, D)  student equilibrium on masked audio
        teacher_eq_v: Tensor,  # (B, L, D)  teacher equilibrium (stop-grad; unmasked)
        teacher_eq_a: Tensor,  # (B, L, D)  teacher equilibrium (stop-grad; unmasked)
        mask_v: Tensor,        # (B, L) bool  True at masked video positions
        mask_a: Tensor,        # (B, L) bool  True at masked audio positions
    ) -> Tensor:
        """
        Compute JEPA masked-latent loss on the masked token positions.

            J_JEPA = (1/|M_v|) Σ_{ℓ∈M_v} ‖P_v(g̃*_v,ℓ) − g*_T_v,ℓ‖²
                   + (1/|M_a|) Σ_{ℓ∈M_a} ‖P_a(g̃*_a,ℓ) − g*_T_a,ℓ‖²

        Teacher targets are stop-gradiented inside get_teacher_equilibrium.
        This method asserts that invariant.

        Args:
            student_eq_v, student_eq_a: student equilibria on masked inputs
            teacher_eq_v, teacher_eq_a: teacher equilibria on unmasked inputs (sg)
            mask_v, mask_a:             boolean masks (True = masked position)
        Returns:
            scalar JEPA loss
        """
        assert not teacher_eq_v.requires_grad, "teacher_eq_v must be stop-grad"
        assert not teacher_eq_a.requires_grad, "teacher_eq_a must be stop-grad"

        # Predict masked positions
        pred_v = self.pred_v(student_eq_v)  # (B, L, D_pred)
        pred_a = self.pred_a(student_eq_a)  # (B, L, D_pred)

        # Slice masked positions
        # mask: (B, L) → use as index
        def masked_mse(pred, target, mask):
            # pred, target: (B, L, D_pred); mask: (B, L)
            p = pred[mask]   # (N_masked, D_pred)
            t = target[mask] # (N_masked, D_pred)
            if p.numel() == 0:
                return pred.new_zeros(())
            return F.mse_loss(p, t)

        loss_v = masked_mse(pred_v, teacher_eq_v, mask_v)
        loss_a = masked_mse(pred_a, teacher_eq_a, mask_a)
        return loss_v + loss_a

    @staticmethod
    def make_contiguous_mask(
        B: int,
        L: int,
        mask_ratio: float = 0.4,
        device: torch.device | None = None,
    ) -> Tensor:
        """
        Generate contiguous temporal masks per batch item.
        Each mask covers a random contiguous block of mask_ratio*L tokens.

        Args:
            B, L:        batch size, sequence length
            mask_ratio:  fraction of tokens to mask (default 0.4)
        Returns:
            mask: (B, L) bool tensor, True at masked positions
        """
        n_mask = max(1, int(mask_ratio * L))
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            start = torch.randint(0, L - n_mask + 1, (1,)).item()
            mask[b, start: start + n_mask] = True
        return mask
