"""
met/core/layernorm.py
=====================
Tokenwise LayerNorm — per-token normalization over feature dimension D.

nn.LayerNorm(D) with input (B, L, D) is exactly this operation.
We wrap it to make the Jacobian null-space issue explicit in comments.

Stability note (§3.3 of the paper):
    J_ℓ = ∂g_ℓ/∂x_ℓ  (LayerNorm Jacobian)
    J_ℓ @ 1 = 0        (mean subtraction kills the all-ones direction)
    ⟹ J_ℓ J_ℓᵀ is PSD, NOT PD
    ⟹ Energy descent gives Ė ≤ 0 only, NOT Ė < 0.

This is the Lyapunov gap described in the manuscript. The implementation
logs norm of the null component during solver runs (see met/utils/diagnostics.py).
"""

import torch
import torch.nn as nn
from torch import Tensor


class TokenwiseLayerNorm(nn.Module):
    """
    Per-token LayerNorm: normalizes each token x_ℓ ∈ ℝ^D independently.

    Equation (§3):
        g_{ℓi} = γ_i · (x_{ℓi} − x̄_ℓ) / sqrt(σ_ℓ + ε) + δ_i

    Args:
        D:   feature dimension to normalize over
        eps: numerical stability constant
    """

    def __init__(self, D: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(D, eps=eps)
        self.D = D

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, D)  pre-normalized states
        Returns:
            g: (B, L, D)  normalized states (used in energy computation)
        """
        return self.ln(x)

    def null_component(self, grad_x: Tensor) -> Tensor:
        """
        Returns the component of grad_x that lies in the LayerNorm null space
        (the per-token mean direction). Useful for the null-space event diagnostic.

        Null direction per token ℓ: e_ℓ = 1/sqrt(D) · 1_D
        Null component: (grad_x_ℓ · e_ℓ) e_ℓ  for each ℓ

        Args:
            grad_x: (B, L, D)  gradient w.r.t. x
        Returns:
            null_part: (B, L, D)
        """
        # Mean projection: (B, L, 1) → broadcast
        mean_proj = grad_x.mean(dim=-1, keepdim=True)  # scalar per token
        return mean_proj.expand_as(grad_x)
