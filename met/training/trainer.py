"""
met/training/trainer.py
========================
MET training loop — incremental phase-based, as described in §7.

Build order (must respect; each phase requires the prior to be stable):

    Phase A — BPTT, J_gen only, attention-only energy
              Verify: Tier 0 passes, monotone energy, decreasing FP residual
    Phase B — Add Hopfield banks to energy
              Verify: Tier 1 unit tests pass
    Phase C — Add J_JEPA + J_rank as outer-loop auxiliaries
              Verify: Tier 2 + 3 experiments pass
    Phase D — Switch primary param update to EqProp; keep BPTT as comparison
              Verify: attractor_agreement > 0.95 consistently
    Phase E — Add Foley losses: J_sync + J_sem + J_temp
              Verify: FAD improves vs Phase D baseline

The trainer is deliberately NOT a monolithic class. Phases A-E each have
their own `train_epoch_*` function to keep logic readable and allow selective
use of each training mode.

Usage (Phase A):
    trainer = METTrainer(energy, audio_head, optimizer, cfg)
    for epoch in range(n_epochs):
        metrics = trainer.train_epoch_phaseA(dataloader)

Usage (Phase D):
    eqprop = EqPropEstimator(energy, s=cfg.s, T=cfg.T_iter)
    trainer = METTrainer(energy, audio_head, optimizer, cfg, eqprop=eqprop)
    for epoch in range(n_epochs):
        metrics = trainer.train_epoch_phaseD(dataloader)
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from met.core.energy import METEnergy
from met.heads.audio_head import AudioHead
from met.solver.gradient_descent import run_deterministic_solver
from met.solver.eqprop import EqPropEstimator
from met.training.losses import J_mel, J_rank
from met.training.jepa import JEPATeacher
from met.utils.diagnostics import EpochLogger


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    # Solver
    T_iter: int = 50
    eta: float = 0.01
    grad_clip: float = 1.0
    early_stop_eps: float = 1e-6

    # Loss weights
    lambda_gen: float = 1.0
    lambda_rank: float = 0.05
    lambda_jepa: float = 0.1
    lambda_sem: float = 0.1
    lambda_temp: float = 0.1
    lambda_sync: float = 0.1
    margin: float = 0.5

    # EqProp
    s: float = 0.01
    min_attractor_agreement: float = 0.90

    # JEPA
    mu: float = 0.99
    mask_ratio: float = 0.4

    # BPTT
    bptt_steps: int = 20    # BPTT unrolls only this many steps (< T_iter for memory)

    # Optimizer
    grad_clip_global: float = 1.0

    # AMP (applies to BPTT-heavy phases; EqProp path remains FP32)
    use_amp: bool = False
    amp_dtype: str = "bfloat16"  # "float16" | "bfloat16"


# ---------------------------------------------------------------------------
# METTrainer
# ---------------------------------------------------------------------------

class METTrainer:
    """
    MET training orchestrator.

    Args:
        energy:     METEnergy (student model)
        audio_head: AudioHead output projection
        optimizer:  torch optimizer (AdamW recommended)
        cfg:        TrainerConfig
        eqprop:     optional EqPropEstimator (required for Phase D)
        jepa:       optional JEPATeacher (required for Phase C+)
        device:     torch device
    """

    def __init__(
        self,
        energy: METEnergy,
        audio_head: AudioHead,
        optimizer: torch.optim.Optimizer,
        cfg: TrainerConfig,
        eqprop: Optional[EqPropEstimator] = None,
        jepa: Optional[JEPATeacher] = None,
        device: Optional[torch.device] = None,
        head_optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        self.energy = energy
        self.audio_head = audio_head
        self.optimizer = optimizer
        self.head_optimizer = head_optimizer or optimizer
        self.cfg = cfg
        self.eqprop = eqprop
        self.jepa = jepa
        self.device = device or torch.device("cpu")
        self.logger = EpochLogger()
        self.global_step = 0
        self.amp_enabled = bool(
            self.cfg.use_amp and self.device.type == "cuda"
        )
        self.amp_dtype = (
            torch.bfloat16 if self.cfg.amp_dtype.lower() == "bfloat16"
            else torch.float16
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    def _autocast_ctx(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.cuda.amp.autocast(dtype=self.amp_dtype)

    def _audio_reconstruction_loss(
        self,
        x_a_state: Tensor,
        mel_target: Tensor,
        detach_state: bool = False,
    ) -> Tensor:
        if detach_state:
            x_a_state = x_a_state.detach()
        mel_pred = self.audio_head(x_a_state)
        return self.cfg.lambda_gen * J_mel(mel_pred, mel_target)

    def _update_audio_head(self, x_a_state: Tensor, mel_target: Tensor) -> Tensor:
        """
        Update the readout head on a detached equilibrium state.

        Under EqProp the energy gradients come from the free/nudged equilibrium
        estimator, but the head sits outside the energy and still trains with
        ordinary backprop on the free equilibrium output.
        """
        self.head_optimizer.zero_grad()
        loss = self._audio_reconstruction_loss(
            x_a_state,
            mel_target,
            detach_state=True,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.audio_head.parameters()),
            self.cfg.grad_clip_global,
        )
        self.head_optimizer.step()
        return loss

    # ------------------------------------------------------------------
    # Phase A — BPTT with J_gen only (attention-only energy)
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        dataloader: DataLoader,
        method: str = "bptt",
        attention_only: bool = False,
    ) -> dict:
        """
        Minimal dispatcher for the two primary update rules.

        This keeps the common case simple:
            method="bptt"   -> unrolled solver training
            method="eqprop" -> equilibrium propagation training
        """
        method = method.lower()
        if method == "bptt":
            return self.train_epoch_phaseA(
                dataloader,
                attention_only=attention_only,
            )
        if method == "eqprop":
            if attention_only:
                raise ValueError(
                    "EqProp is only wired for the full energy path. "
                    "Use method='bptt' for attention-only training."
                )
            return self.train_epoch_phaseD(dataloader)
        raise ValueError(f"Unknown training method '{method}'. Expected 'bptt' or 'eqprop'.")

    def train_epoch_phaseA(
        self,
        dataloader: DataLoader,
        attention_only: bool = True,
    ) -> dict:
        """
        Phase A: BPTT through T_bptt unrolled steps, J_gen only.
        Uses attention-only energy by default (add Hopfield in Phase B).

        Batch format expected: dict with keys "x_v" (B,L,D_tok) and
        "mel_target" (B,L,n_mels) already tokenized and projected.
        """
        self.energy.train()
        self.audio_head.train()
        self.logger.reset()

        for batch in dataloader:
            x_v = batch["x_v"].to(self.device)          # (B, L, D)
            x_a_init = batch["x_a_init"].to(self.device) # (B, L, D)  noisy init
            mel_target = batch["mel_target"].to(self.device)

            with self._autocast_ctx():
                # BPTT: run T_bptt steps with create_graph=True
                x_v_out, x_a_out, logs = run_deterministic_solver(
                    self.energy,
                    x_v, x_a_init,
                    T=self.cfg.bptt_steps,
                    eta=self.cfg.eta,
                    freeze_v=True,
                    grad_clip=self.cfg.grad_clip,
                    create_graph=True,
                    attention_only=attention_only,
                )

                # Reconstruction loss
                loss = self._audio_reconstruction_loss(x_a_out, mel_target)

            self.optimizer.zero_grad()
            if self.amp_enabled:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.energy.parameters()) + list(self.audio_head.parameters()),
                self.cfg.grad_clip_global,
            )
            if self.amp_enabled:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Diagnostics
            E_final = logs[-1]["E_total"] if logs else float("nan")
            mono_rate = sum(1 for l in logs if l["monotone"]) / max(len(logs), 1)
            self.logger.log(
                J_gen=loss.item(),
                E_final=E_final,
                monotone_rate=mono_rate,
                fp_residual=logs[-1]["fp_residual"] if logs else float("nan"),
            )
            self.global_step += 1

        return self.logger.epoch_summary()

    # ------------------------------------------------------------------
    # Phase B — Like Phase A but with full energy (Hopfield enabled)
    # ------------------------------------------------------------------

    def train_epoch_phaseB(self, dataloader: DataLoader) -> dict:
        """Phase B: same as A but with full energy (attention + Hopfield)."""
        return self.train_epoch_phaseA(dataloader, attention_only=False)

    # ------------------------------------------------------------------
    # Phase C — J_gen (BPTT) + J_rank + J_JEPA outer-loop
    # ------------------------------------------------------------------

    def train_epoch_phaseC(
        self,
        dataloader: DataLoader,
    ) -> dict:
        """
        Phase C: BPTT for J_gen + outer-loop J_rank and J_JEPA.

        Batch must additionally contain:
            "x_a_neg": (B, L, D) mismatched audio (for J_rank)
            Masked inputs for JEPA (generated on the fly via JEPATeacher.make_contiguous_mask)
        """
        assert self.jepa is not None, "Phase C requires a JEPATeacher (pass jepa= to METTrainer)"
        self.energy.train()
        self.audio_head.train()
        self.logger.reset()

        for batch in dataloader:
            x_v = batch["x_v"].to(self.device)
            x_a_init = batch["x_a_init"].to(self.device)
            x_a_neg = batch["x_a_neg"].to(self.device)
            mel_target = batch["mel_target"].to(self.device)
            B, L, D = x_a_init.shape

            with self._autocast_ctx():
                # --- J_gen (BPTT) ---
                _, x_a_out, logs = run_deterministic_solver(
                    self.energy, x_v, x_a_init,
                    T=self.cfg.bptt_steps, eta=self.cfg.eta,
                    freeze_v=True, create_graph=True,
                )
                loss_gen = self._audio_reconstruction_loss(x_a_out, mel_target)

                # --- J_rank: compute matched and mismatched energies ---
                with torch.no_grad():
                    x_v_pair = torch.cat([x_v, x_v], dim=0)
                    x_a_pair = torch.cat([x_a_init, x_a_neg], dim=0)
                    _, x_a_pair_eq, _ = run_deterministic_solver(
                        self.energy,
                        x_v_pair,
                        x_a_pair,
                        T=self.cfg.T_iter,
                        eta=self.cfg.eta,
                        freeze_v=True,
                        grad_clip=self.cfg.grad_clip,
                    )
                    x_a_eq, x_a_neg_eq = x_a_pair_eq.chunk(2, dim=0)
                E_pair, _ = self.energy(
                    torch.cat([x_v, x_v], dim=0),
                    torch.cat([x_a_eq, x_a_neg_eq], dim=0),
                    freeze_v=True,
                    freeze_a=True,
                )
                E_match, E_mismatch = E_pair.chunk(2, dim=0)
                loss_rank = self.cfg.lambda_rank * J_rank(E_match, E_mismatch, self.cfg.margin)

                # --- J_JEPA ---
                mask_a = JEPATeacher.make_contiguous_mask(B, L, self.cfg.mask_ratio, self.device)
                x_a_masked = x_a_init.clone()
                x_a_masked[mask_a] = 0.0  # simple zero-masking

                # Teacher equilibrium (unmasked, stop-grad)
                teacher_eq_v, teacher_eq_a = self.jepa.get_teacher_equilibrium(
                    x_v, x_a_init, T=self.cfg.T_iter, eta=self.cfg.eta
                )
                # Student equilibrium (masked input)
                _, x_a_student, _ = run_deterministic_solver(
                    self.energy,
                    x_v,
                    x_a_masked,
                    T=self.cfg.bptt_steps,
                    eta=self.cfg.eta,
                    freeze_v=True,
                    create_graph=True,
                )
                loss_jepa = self.cfg.lambda_jepa * self.jepa.compute_jepa_loss(
                    x_v, x_a_student,
                    teacher_eq_v, teacher_eq_a,
                    torch.zeros(B, L, dtype=torch.bool, device=self.device),  # v mask: none
                    mask_a,
                )

                loss = loss_gen + loss_rank + loss_jepa

            self.optimizer.zero_grad()
            if self.amp_enabled:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.energy.parameters()) + list(self.audio_head.parameters())
                + list(self.jepa.pred_v.parameters()) + list(self.jepa.pred_a.parameters()),
                self.cfg.grad_clip_global,
            )
            if self.amp_enabled:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.jepa.update_ema(self.energy)

            self.logger.log(
                J_total=loss.item(), J_gen=loss_gen.item(),
                J_rank=loss_rank.item(), J_jepa=loss_jepa.item(),
            )
            self.global_step += 1

        return self.logger.epoch_summary()

    # ------------------------------------------------------------------
    # Phase D — EqProp for J_gen + outer-loop auxiliaries
    # ------------------------------------------------------------------

    def train_epoch_phaseD(self, dataloader: DataLoader) -> dict:
        """
        Phase D: Use EqPropEstimator for primary J_gen gradient.
        J_rank applied as outer-loop on top (standard backward).

        Critical monitoring: log attractor_agreement every batch.
        If < 0.90 consistently, revert to Phase C (BPTT) and debug.
        """
        assert self.eqprop is not None, "Phase D requires EqPropEstimator"
        self.energy.train()
        self.audio_head.train()
        self.logger.reset()

        for batch in dataloader:
            x_v = batch["x_v"].to(self.device)
            x_a_init = batch["x_a_init"].to(self.device)
            x_a_neg = batch.get("x_a_neg", torch.randn_like(x_a_init)).to(self.device)
            mel_target = batch["mel_target"].to(self.device)

            # --- EqProp for J_gen ---
            nudge_objective = lambda _x_v, x_a_state: self._audio_reconstruction_loss(
                x_a_state,
                mel_target,
            )
            eqprop_result = self.eqprop.estimate_gradient(
                x_v,
                x_a_init,
                eta=self.cfg.eta,
                freeze_v=True,
                grad_clip=self.cfg.grad_clip,
                nudge_objective=nudge_objective,
            )
            agreement = eqprop_result["attractor_agreement"]

            applied = self.eqprop.apply_gradients(
                eqprop_result["eqprop_grads"],
                self.optimizer,
                min_agreement=self.cfg.min_attractor_agreement,
                agreement=agreement,
            )

            # The readout head is outside the energy and still trains by
            # ordinary backprop on the free equilibrium.
            loss_head = self._update_audio_head(
                eqprop_result["x_a_free"],
                mel_target,
            )

            # --- J_rank outer loop on top ---
            with torch.no_grad():
                _, x_a_neg_eq, _ = run_deterministic_solver(
                    self.energy,
                    x_v,
                    x_a_neg,
                    T=self.cfg.T_iter,
                    eta=self.cfg.eta,
                    freeze_v=True,
                    grad_clip=self.cfg.grad_clip,
                )
            E_pair, _ = self.energy(
                torch.cat([x_v, x_v], dim=0),
                torch.cat([eqprop_result["x_a_free"], x_a_neg_eq], dim=0),
                freeze_v=True,
                freeze_a=True,
            )
            E_match, E_mismatch = E_pair.chunk(2, dim=0)
            loss_rank = self.cfg.lambda_rank * J_rank(E_match, E_mismatch, self.cfg.margin)
            self.optimizer.zero_grad()
            loss_rank.backward()
            self.optimizer.step()

            if self.jepa is not None:
                self.jepa.update_ema(self.energy)

            self.logger.log(
                attractor_agreement=agreement,
                eqprop_applied=float(applied),
                E_free=eqprop_result["E_free"],
                E_nudged=eqprop_result["E_nudged"],
                J_gen_head=loss_head.item(),
                J_rank=loss_rank.item(),
            )
            self.global_step += 1

        return self.logger.epoch_summary()
