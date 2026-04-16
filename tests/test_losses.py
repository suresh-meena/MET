"""
tests/test_losses.py
====================
Unit tests for training losses used by MET.
"""

from __future__ import annotations

import torch
import pytest

from met.training.losses import (
    J_jepa,
    J_mel,
    J_rank,
    J_sem,
    J_sync,
    J_temp,
    foley_total_loss,
)


class TestBasicLosses:
    def test_j_mel_zero_on_exact_match(self):
        x = torch.randn(2, 8, 16)
        assert J_mel(x, x).item() == pytest.approx(0.0, abs=1e-8)

    def test_j_sync_zero_on_exact_match(self):
        x = torch.randn(2, 4, 10)
        assert J_sync(x, x).item() == pytest.approx(0.0, abs=1e-8)

    def test_j_rank_lower_when_mismatch_energy_is_higher(self):
        e_match = torch.tensor(-2.0)
        e_mismatch_easy = torch.tensor(1.0)
        e_mismatch_hard = torch.tensor(-1.8)

        loss_easy = J_rank(e_match, e_mismatch_easy, margin=0.5).item()
        loss_hard = J_rank(e_match, e_mismatch_hard, margin=0.5).item()
        assert loss_easy < loss_hard


class TestContrastiveLosses:
    def test_j_sem_prefers_aligned_pairs_over_shuffled_pairs(self):
        B, D = 8, 12
        h_v = torch.randn(B, D)
        h_a_aligned = h_v + 0.05 * torch.randn(B, D)
        h_a_shuffled = h_a_aligned.flip(0)

        aligned = J_sem(h_v, h_a_aligned, tau=0.1).item()
        shuffled = J_sem(h_v, h_a_shuffled, tau=0.1).item()
        assert aligned < shuffled

    def test_j_temp_prefers_temporal_alignment(self):
        B, W, D = 4, 6, 10
        h_v = torch.randn(B, W, D)
        h_a_aligned = h_v + 0.05 * torch.randn(B, W, D)
        h_a_shifted = torch.roll(h_a_aligned, shifts=1, dims=1)

        aligned = J_temp(h_v, h_a_aligned, tau=0.1).item()
        shifted = J_temp(h_v, h_a_shifted, tau=0.1).item()
        assert aligned < shifted


class TestJEPAInvariants:
    def test_j_jepa_requires_stopgrad_teacher_targets(self):
        B, L, D = 2, 4, 6
        student_v = torch.randn(B, L, D, requires_grad=True)
        student_a = torch.randn(B, L, D, requires_grad=True)
        teacher_v = torch.randn(B, L, D, requires_grad=True)
        teacher_a = torch.randn(B, L, D, requires_grad=False)

        with pytest.raises(AssertionError):
            _ = J_jepa(student_v, teacher_v, student_a, teacher_a)


class TestFoleyTotal:
    def test_total_loss_contains_expected_components(self):
        B, L, D = 2, 5, 8
        mel_pred = torch.randn(B, L, D, requires_grad=True)
        mel_target = torch.randn(B, L, D)

        E_match = torch.tensor(-1.0, requires_grad=True)
        E_mismatch = torch.tensor(0.5, requires_grad=True)

        S_a = torch.randn(B, 3, D)
        S_v = torch.randn(B, 3, D)
        h_v_clip = torch.randn(B, D)
        h_a_clip = h_v_clip + 0.1 * torch.randn(B, D)
        h_v_win = torch.randn(B, 4, D)
        h_a_win = h_v_win + 0.1 * torch.randn(B, 4, D)

        loss, comps = foley_total_loss(
            mel_pred=mel_pred,
            mel_target=mel_target,
            E_matched=E_match,
            E_mismatched=E_mismatch,
            S_a=S_a,
            S_v=S_v,
            h_v_clip=h_v_clip,
            h_a_clip=h_a_clip,
            h_v_win=h_v_win,
            h_a_win=h_a_win,
        )
        assert torch.isfinite(loss)
        expected = {"J_mel", "J_rank", "J_sync", "J_sem", "J_temp", "J_total"}
        assert expected.issubset(comps.keys())
