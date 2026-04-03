"""
met/training/losses.py
======================
All loss terms for MET training.

Loss hierarchy (§5 / §7):
    J_gen:   PRIMARY — samplewise reconstruction (only term in EqProp nudge)
    J_jepa:  auxiliary — masked cross-modal latent prediction
    J_rank:  auxiliary — energy landscape shaping
    J_sem:   Foley auxiliary — clip-level AV alignment (InfoNCE; batch-coupled)
    J_temp:  Foley auxiliary — window-level AV timing (InfoNCE; same-clip negatives)
    J_sync:  Foley auxiliary — AV-onset vs visual-motion MSE

    The EqProp nudge uses ONLY J_gen. All other terms are outer-loop:
    applied via standard backward() on top of EqProp for J_gen.

CRITICAL: J_sem and J_temp are batch-coupled (InfoNCE over a batch).
    They are NOT covered by the samplewise EqProp theorem.
    Apply them only as outer-loop auxiliaries.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# J_gen: Mel-spectrogram reconstruction (primary for Foley)
# ---------------------------------------------------------------------------

def J_mel(mel_pred: Tensor, mel_target: Tensor) -> Tensor:
    """
    L2 reconstruction loss in mel-spectrogram space.
    Primary generative loss; dominant weight lambda_gen = 1.0.

    Args:
        mel_pred:   (B, L, n_mels)  predicted mel from audio head
        mel_target: (B, L, n_mels)  ground-truth mel
    Returns:
        scalar MSE loss
    """
    return F.mse_loss(mel_pred, mel_target)


# ---------------------------------------------------------------------------
# J_sync: AV synchronization penalty (§8, Foley)
# ---------------------------------------------------------------------------

def J_sync(
    S_a: Tensor,  # (B, W, D_feat)  audio-onset features per window
    S_v: Tensor,  # (B, W, D_feat)  visual-motion features per window
) -> Tensor:
    """
    AV synchronization penalty: aligns audio events with visual motion.

        J_sync = (1/W) Σ_w ‖S_a(â_w) − S_v(v_w)‖²

    Features S_a and S_v are fixed (frozen) front-end representations,
    not model parameters. This keeps the synthesis claim anchored to
    a real signal rather than to representation geometry alone.

    Args:
        S_a: (B, W, D_feat)
        S_v: (B, W, D_feat)
    Returns:
        scalar loss
    """
    return F.mse_loss(S_a, S_v)


# ---------------------------------------------------------------------------
# J_JEPA: Masked cross-modal latent prediction (§5)
# ---------------------------------------------------------------------------

def J_jepa(
    student_pred_v: Tensor,  # (B, |M_v|, D_p)  predictor P_v output on masked video tokens
    teacher_target_v: Tensor,  # (B, |M_v|, D_p)  teacher equilibrium (stop-grad)
    student_pred_a: Tensor,  # (B, |M_a|, D_p)
    teacher_target_a: Tensor,  # (B, |M_a|, D_p)
) -> Tensor:
    """
    JEPA masked-latent loss (§5):
        J_JEPA = (1/|M_v|) Σ_{ℓ∈M_v} ‖P_v(g̃*_v,ℓ) − g*_T_v,ℓ‖²
               + (1/|M_a|) Σ_{ℓ∈M_a} ‖P_a(g̃*_a,ℓ) − g*_T_a,ℓ‖²

    Teacher targets must be stop-gradiented BEFORE being passed in.
    This is enforced here with an assertion check.

    Args:
        student_pred_v:   predictor output on masked video positions
        teacher_target_v: teacher equilibrium at those positions (sg applied)
        student_pred_a:   predictor output on masked audio positions
        teacher_target_a: teacher equilibrium at those positions (sg applied)
    Returns:
        scalar loss
    """
    assert not teacher_target_v.requires_grad, \
        "teacher_target_v must be stop-gradiented before passing to J_jepa"
    assert not teacher_target_a.requires_grad, \
        "teacher_target_a must be stop-gradiented before passing to J_jepa"

    loss_v = F.mse_loss(student_pred_v, teacher_target_v)
    loss_a = F.mse_loss(student_pred_a, teacher_target_a)
    return loss_v + loss_a


# ---------------------------------------------------------------------------
# J_rank: Energy ranking regularizer (§5)
# ---------------------------------------------------------------------------

def J_rank(
    E_matched: Tensor,     # scalar — energy of matched (x^v, x^a) pair
    E_mismatched: Tensor,  # scalar — energy of mismatched pair (x̄^v, x^a)
    margin: float = 0.5,
) -> Tensor:
    """
    Ranking regularizer: matched pairs should have lower energy.

        J_rank = softplus(E_matched − E_mismatched + m)

    Symmetric audio-negative can be added analogously.
    x̄^v is a mismatched or temporally shifted clip.

    Args:
        E_matched:    energy of correct pair
        E_mismatched: energy of wrong pair (same x^a, different x^v)
        margin:       margin m > 0
    Returns:
        scalar loss
    """
    return F.softplus(E_matched - E_mismatched + margin)


# ---------------------------------------------------------------------------
# J_sem: Semantic (clip-level) contrastive loss (§8, Foley)
# ---------------------------------------------------------------------------

def J_sem(
    h_v: Tensor,    # (B, D_p)  clip-level pooled video equilibrium features
    h_a: Tensor,    # (B, D_p)  clip-level pooled audio equilibrium features
    tau: float = 0.07,
) -> Tensor:
    """
    Clip-level semantic contrastive loss (InfoNCE, symmetric):
    Aligns global audio-visual semantics; uses other videos in batch as negatives.

    BATCH-COUPLED: Not covered by samplewise EqProp theorem.
    Apply as outer-loop auxiliary only.

    Args:
        h_v: (B, D_p)  video features (L2-normalized inside)
        h_a: (B, D_p)  audio features (L2-normalized inside)
        tau: temperature
    Returns:
        scalar loss
    """
    h_v = F.normalize(h_v, dim=-1)  # (B, D_p)
    h_a = F.normalize(h_a, dim=-1)  # (B, D_p)

    # Similarity matrix: (B, B)
    sim = torch.einsum("id, jd -> ij", h_a, h_v) / tau

    # Symmetric InfoNCE
    labels = torch.arange(sim.size(0), device=sim.device)
    loss_a2v = F.cross_entropy(sim, labels)
    loss_v2a = F.cross_entropy(sim.T, labels)
    return (loss_a2v + loss_v2a) / 2.0


# ---------------------------------------------------------------------------
# J_temp: Temporal (window-level) contrastive loss (§8, Foley)
# ---------------------------------------------------------------------------

def J_temp(
    h_v_win: Tensor,  # (B, W, D_p)  per-window video features
    h_a_win: Tensor,  # (B, W, D_p)  per-window audio features
    tau: float = 0.07,
) -> Tensor:
    """
    Window-level temporal contrastive loss:
    Synchronized windows are positives; other windows IN THE SAME CLIP are negatives.

    This is the loss that enforces Foley timing quality.
    Hard negatives: nearby non-overlapping windows within the same clip,
    NOT random clips (per §8 of the paper).

    BATCH-COUPLED: Apply as outer-loop auxiliary only.

    Args:
        h_v_win: (B, W, D_p)
        h_a_win: (B, W, D_p)
        tau: temperature
    Returns:
        scalar loss
    """
    B, W, D_p = h_v_win.shape
    h_v_win = F.normalize(h_v_win, dim=-1)  # (B, W, D_p)
    h_a_win = F.normalize(h_a_win, dim=-1)  # (B, W, D_p)

    total = 0.0
    for b in range(B):
        # (W, D_p) for clip b
        hv = h_v_win[b]  # (W, D_p)
        ha = h_a_win[b]  # (W, D_p)

        # Within-clip similarity: (W, W)
        sim = torch.einsum("id, jd -> ij", ha, hv) / tau

        labels = torch.arange(W, device=sim.device)
        total = total + (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2.0

    return total / B


# ---------------------------------------------------------------------------
# Combined Foley objective
# ---------------------------------------------------------------------------

def foley_total_loss(
    mel_pred: Tensor,
    mel_target: Tensor,
    E_matched: Tensor,
    E_mismatched: Tensor,
    S_a: Tensor | None = None,
    S_v: Tensor | None = None,
    student_pred_v: Tensor | None = None,
    teacher_target_v: Tensor | None = None,
    student_pred_a: Tensor | None = None,
    teacher_target_a: Tensor | None = None,
    h_v_clip: Tensor | None = None,
    h_a_clip: Tensor | None = None,
    h_v_win: Tensor | None = None,
    h_a_win: Tensor | None = None,
    lam_gen: float = 1.0,
    lam_sync: float = 0.1,
    lam_jepa: float = 0.1,
    lam_rank: float = 0.05,
    lam_sem: float = 0.1,
    lam_temp: float = 0.1,
    margin: float = 0.5,
    tau_s: float = 0.07,
    tau_t: float = 0.07,
) -> tuple[Tensor, dict[str, float]]:
    """
    Combined Foley training objective (§8):

        J_train = λ_gen·J_mel
                + λ_sync·J_sync     (if S_a, S_v provided)
                + λ_jepa·J_JEPA    (if teacher targets provided)
                + λ_rank·J_rank
                + λ_sem·J_sem      (if clip features provided)
                + λ_temp·J_temp    (if window features provided)

    λ_gen is dominant (=1.0 default). All others are auxiliary.

    Returns:
        total loss (scalar), component dict for logging
    """
    comps: dict[str, float] = {}

    loss = lam_gen * J_mel(mel_pred, mel_target)
    comps["J_mel"] = J_mel(mel_pred, mel_target).item()

    loss = loss + lam_rank * J_rank(E_matched, E_mismatched, margin)
    comps["J_rank"] = J_rank(E_matched, E_mismatched, margin).item()

    if S_a is not None and S_v is not None:
        js = J_sync(S_a, S_v)
        loss = loss + lam_sync * js
        comps["J_sync"] = js.item()

    if all(x is not None for x in [student_pred_v, teacher_target_v,
                                     student_pred_a, teacher_target_a]):
        jj = J_jepa(student_pred_v, teacher_target_v, student_pred_a, teacher_target_a)
        loss = loss + lam_jepa * jj
        comps["J_jepa"] = jj.item()

    if h_v_clip is not None and h_a_clip is not None:
        js2 = J_sem(h_v_clip, h_a_clip, tau_s)
        loss = loss + lam_sem * js2
        comps["J_sem"] = js2.item()

    if h_v_win is not None and h_a_win is not None:
        jt = J_temp(h_v_win, h_a_win, tau_t)
        loss = loss + lam_temp * jt
        comps["J_temp"] = jt.item()

    comps["J_total"] = loss.item()
    return loss, comps
