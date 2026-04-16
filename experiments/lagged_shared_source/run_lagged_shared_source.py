"""
Lagged shared-source audio-video retrieval synthetic.

Goal:
Recover a shared latent event across audio/video while ignoring modality-private
nuisance and handling temporal lag.

This script implements:
1) Synthetic generator with shared latent events, per-modality nuisance, and lag.
2) Trainable baselines:
   - Dual-encoder pooled cosine (InfoNCE)
   - Lag-aware bilinear energy model (InfoNCE + optional hard-negative ranking)
3) Non-trainable scoring baselines:
   - Framewise min-distance on raw features
   - Oracle lag cross-correlation on raw features
4) Metrics:
   - Bidirectional retrieval (R@1, R@5, MRR)
   - Optional lag-robustness curve
   - Optional hard-negative robustness benchmark
   - Optional component ablations
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class GeneratorConfig:
    K: int = 8
    T: int = 8
    d_obs: int = 32
    d_event: int = 16
    d_nuisance: int = 8
    n_speakers: int = 32
    n_backgrounds: int = 32
    noise_std: float = 0.08
    nuisance_strength: float = 1.0
    event_stay_prob: float = 0.65
    lag_values: Tuple[int, ...] = (-2, -1, 0, 1, 2)


@dataclass
class GeneratorParams:
    event_embed: torch.Tensor
    U_a: torch.Tensor
    U_v: torch.Tensor
    S_a: torch.Tensor
    B_v: torch.Tensor
    speaker_bank: torch.Tensor
    background_bank: torch.Tensor


@dataclass
class SplitData:
    audio: torch.Tensor
    video: torch.Tensor
    z: torch.Tensor
    tau: torch.Tensor
    s_id: torch.Tensor
    b_id: torch.Tensor

    def size(self) -> int:
        return int(self.audio.shape[0])


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    hard_negative_rate: float = 0.0
    hard_negative_weight: float = 0.25
    hard_margin: float = 0.2
    overlap_threshold: float = 0.5


class AVPairDataset(Dataset):
    def __init__(self, split: SplitData):
        self.split = split

    def __len__(self) -> int:
        return self.split.size()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "audio": self.split.audio[idx],
            "video": self.split.video[idx],
            "z": self.split.z[idx],
            "tau": self.split.tau[idx],
            "s_id": self.split.s_id[idx],
            "b_id": self.split.b_id[idx],
        }


class FrameMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 64, d_out: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PooledCosineDualEncoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 64):
        super().__init__()
        self.audio_encoder = FrameMLP(d_in, d_hidden, d_hidden)
        self.video_encoder = FrameMLP(d_in, d_hidden, d_hidden)

    def similarity_matrix(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        a = F.normalize(self.audio_encoder(audio).mean(dim=1), dim=-1)
        v = F.normalize(self.video_encoder(video).mean(dim=1), dim=-1)
        return a @ v.T


class LagAwareEnergyModel(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 64, max_lag: int = 2):
        super().__init__()
        self.audio_encoder = FrameMLP(d_in, d_hidden, d_hidden)
        self.video_encoder = FrameMLP(d_in, d_hidden, d_hidden)
        self.W = nn.Parameter(torch.eye(d_hidden))
        self.max_lag = max_lag

    def similarity_matrix(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        a = self.audio_encoder(audio)
        v = self.video_encoder(video)
        return lagged_bilinear_scores(a, v, self.W, self.max_lag)


def make_cpu_generator(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def build_generator_params(cfg: GeneratorConfig, seed: int) -> GeneratorParams:
    g = make_cpu_generator(seed)

    event_embed = F.normalize(torch.randn(cfg.K, cfg.d_event, generator=g), dim=-1)
    speaker_bank = F.normalize(torch.randn(cfg.n_speakers, cfg.d_nuisance, generator=g), dim=-1)
    background_bank = F.normalize(torch.randn(cfg.n_backgrounds, cfg.d_nuisance, generator=g), dim=-1)

    scale_event = 1.0 / (cfg.d_event ** 0.5)
    scale_nuis = 1.0 / (cfg.d_nuisance ** 0.5)

    U_a = torch.randn(cfg.d_obs, cfg.d_event, generator=g) * scale_event
    U_v = torch.randn(cfg.d_obs, cfg.d_event, generator=g) * scale_event
    S_a = torch.randn(cfg.d_obs, cfg.d_nuisance, generator=g) * scale_nuis
    B_v = torch.randn(cfg.d_obs, cfg.d_nuisance, generator=g) * scale_nuis

    return GeneratorParams(
        event_embed=event_embed,
        U_a=U_a,
        U_v=U_v,
        S_a=S_a,
        B_v=B_v,
        speaker_bank=speaker_bank,
        background_bank=background_bank,
    )


def sample_latent_events(num_samples: int, cfg: GeneratorConfig, g: torch.Generator) -> torch.Tensor:
    z = torch.empty(num_samples, cfg.T, dtype=torch.long)
    z[:, 0] = torch.randint(0, cfg.K, (num_samples,), generator=g)

    for t in range(1, cfg.T):
        stay = torch.rand(num_samples, generator=g) < cfg.event_stay_prob
        new_vals = torch.randint(0, cfg.K, (num_samples,), generator=g)
        z[:, t] = torch.where(stay, z[:, t - 1], new_vals)

    return z


def shift_events_by_tau(events: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    # events: (N, T, D), tau: (N,), and v_t = e(z_{t - tau})
    n, t, d = events.shape
    base = torch.arange(t).unsqueeze(0).expand(n, t)  # (N, T)
    indices = (base - tau.unsqueeze(1)) % t
    gather_idx = indices.unsqueeze(-1).expand(n, t, d)
    return events.gather(dim=1, index=gather_idx)


def make_split(
    num_samples: int,
    cfg: GeneratorConfig,
    params: GeneratorParams,
    seed: int,
    fixed_tau: int | None = None,
    lag_values: Sequence[int] | None = None,
    nuisance_strength: float | None = None,
) -> SplitData:
    g = make_cpu_generator(seed)

    z = sample_latent_events(num_samples, cfg, g)

    if fixed_tau is None:
        candidate_lags = torch.tensor(list(lag_values if lag_values is not None else cfg.lag_values), dtype=torch.long)
        lag_idx = torch.randint(0, candidate_lags.numel(), (num_samples,), generator=g)
        tau = candidate_lags[lag_idx]
    else:
        tau = torch.full((num_samples,), int(fixed_tau), dtype=torch.long)

    s_id = torch.randint(0, cfg.n_speakers, (num_samples,), generator=g)
    b_id = torch.randint(0, cfg.n_backgrounds, (num_samples,), generator=g)

    event_seq = params.event_embed[z]  # (N, T, d_event)
    shifted_event_seq = shift_events_by_tau(event_seq, tau)

    audio_shared = torch.einsum("ntd,od->nto", event_seq, params.U_a)
    video_shared = torch.einsum("ntd,od->nto", shifted_event_seq, params.U_v)

    s_vec = params.speaker_bank[s_id]
    b_vec = params.background_bank[b_id]
    audio_nuis = torch.einsum("nd,od->no", s_vec, params.S_a).unsqueeze(1).expand(-1, cfg.T, -1)
    video_nuis = torch.einsum("nd,od->no", b_vec, params.B_v).unsqueeze(1).expand(-1, cfg.T, -1)

    n_strength = cfg.nuisance_strength if nuisance_strength is None else float(nuisance_strength)
    noise_a = cfg.noise_std * torch.randn(num_samples, cfg.T, cfg.d_obs, generator=g)
    noise_v = cfg.noise_std * torch.randn(num_samples, cfg.T, cfg.d_obs, generator=g)

    audio = audio_shared + n_strength * audio_nuis + noise_a
    video = video_shared + n_strength * video_nuis + noise_v

    return SplitData(
        audio=audio.contiguous(),
        video=video.contiguous(),
        z=z.contiguous(),
        tau=tau.contiguous(),
        s_id=s_id.contiguous(),
        b_id=b_id.contiguous(),
    )


def lagged_bilinear_scores(
    audio_feats: torch.Tensor,
    video_feats: torch.Tensor,
    W: torch.Tensor,
    max_lag: int,
) -> torch.Tensor:
    # audio_feats: (Qa, T, D), video_feats: (Qv, T, D)
    best = None
    for delta in range(-max_lag, max_lag + 1):
        shifted_v = torch.roll(video_feats, shifts=delta, dims=1)
        proj_v = torch.einsum("ctd,df->ctf", shifted_v, W)
        score = torch.einsum("qtd,ctd->qc", audio_feats, proj_v)
        best = score if best is None else torch.maximum(best, score)
    return best


def symmetric_infonce_loss(sim: torch.Tensor, temperature: float) -> torch.Tensor:
    labels = torch.arange(sim.shape[0], device=sim.device)
    logits = sim / temperature
    loss_a2v = F.cross_entropy(logits, labels)
    loss_v2a = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_a2v + loss_v2a)


def batch_hard_negative_loss(
    sim: torch.Tensor,
    z: torch.Tensor,
    s_id: torch.Tensor,
    b_id: torch.Tensor,
    hard_negative_rate: float,
    margin: float,
    overlap_threshold: float,
) -> torch.Tensor:
    bsz = sim.shape[0]
    if hard_negative_rate <= 0.0 or bsz < 2:
        return sim.new_tensor(0.0)

    overlap = (z[:, None, :] == z[None, :, :]).float().mean(dim=-1)
    same_nuisance = (s_id[:, None] == s_id[None, :]) | (b_id[:, None] == b_id[None, :])

    candidates = (overlap >= overlap_threshold) | same_nuisance
    candidates.fill_diagonal_(False)

    pick_mask = torch.rand(bsz, device=sim.device) < hard_negative_rate
    picked_rows = pick_mask.nonzero(as_tuple=False).flatten()
    if picked_rows.numel() == 0:
        return sim.new_tensor(0.0)

    all_idx = torch.arange(bsz, device=sim.device)
    neg_idx = torch.zeros(bsz, dtype=torch.long, device=sim.device)

    for i in picked_rows.tolist():
        valid = all_idx[candidates[i]]
        if valid.numel() == 0:
            valid = all_idx[all_idx != i]
        choice = valid[torch.randint(0, valid.numel(), (1,), device=sim.device)]
        neg_idx[i] = choice

    pos = sim[picked_rows, picked_rows]
    neg = sim[picked_rows, neg_idx[picked_rows]]
    return F.relu(margin - (pos - neg)).mean()


def train_model(
    model: nn.Module,
    split_train: SplitData,
    cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    model.to(device)
    model.train()

    loader = DataLoader(
        AVPairDataset(split_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    last_loss = 0.0
    for epoch in range(cfg.epochs):
        epoch_total = 0.0
        epoch_infonce = 0.0
        epoch_hard = 0.0
        n_steps = 0

        for batch in loader:
            audio = batch["audio"].to(device)
            video = batch["video"].to(device)
            z = batch["z"].to(device)
            s_id = batch["s_id"].to(device)
            b_id = batch["b_id"].to(device)

            sim = model.similarity_matrix(audio, video)
            loss_infonce = symmetric_infonce_loss(sim, cfg.temperature)
            loss_hard = batch_hard_negative_loss(
                sim,
                z=z,
                s_id=s_id,
                b_id=b_id,
                hard_negative_rate=cfg.hard_negative_rate,
                margin=cfg.hard_margin,
                overlap_threshold=cfg.overlap_threshold,
            )
            loss = loss_infonce + cfg.hard_negative_weight * loss_hard

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total += float(loss.item())
            epoch_infonce += float(loss_infonce.item())
            epoch_hard += float(loss_hard.item())
            n_steps += 1

        last_loss = epoch_total / max(1, n_steps)
        print(
            f"  epoch {epoch + 1:02d}/{cfg.epochs:02d}"
            f"  loss={epoch_total / max(1, n_steps):.4f}"
            f"  infonce={epoch_infonce / max(1, n_steps):.4f}"
            f"  hard={epoch_hard / max(1, n_steps):.4f}"
        )

    model.eval()
    return {"train_loss": last_loss}


@torch.no_grad()
def score_matrix_batched(
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    audio: torch.Tensor,
    video: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    n_a = audio.shape[0]
    n_v = video.shape[0]
    out = torch.empty(n_a, n_v, dtype=torch.float32)

    for i in range(0, n_a, batch_size):
        a_chunk = audio[i : i + batch_size].to(device)
        a_count = a_chunk.shape[0]
        for j in range(0, n_v, batch_size):
            v_chunk = video[j : j + batch_size].to(device)
            v_count = v_chunk.shape[0]
            scores = score_fn(a_chunk, v_chunk).detach().cpu()
            out[i : i + a_count, j : j + v_count] = scores
    return out


def retrieval_metrics_from_scores(scores: torch.Tensor) -> Dict[str, float]:
    pos = scores.diagonal().unsqueeze(1)
    ranks = 1 + (scores > pos).sum(dim=1)
    ranks_f = ranks.float()
    return {
        "R@1": float((ranks <= 1).float().mean().item()),
        "R@5": float((ranks <= 5).float().mean().item()),
        "MRR": float((1.0 / ranks_f).mean().item()),
        "MedianRank": float(ranks_f.median().item()),
    }


def bidirectional_metrics(scores_a2v: torch.Tensor) -> Dict[str, Dict[str, float] | float]:
    a2v = retrieval_metrics_from_scores(scores_a2v)
    v2a = retrieval_metrics_from_scores(scores_a2v.T)
    mean_r1 = 0.5 * (a2v["R@1"] + v2a["R@1"])
    mean_mrr = 0.5 * (a2v["MRR"] + v2a["MRR"])
    return {
        "audio_to_video": a2v,
        "video_to_audio": v2a,
        "mean_R@1": mean_r1,
        "mean_MRR": mean_mrr,
    }


def raw_pooled_cosine_scores(audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
    a = F.normalize(audio.mean(dim=1), dim=-1)
    v = F.normalize(video.mean(dim=1), dim=-1)
    return a @ v.T


def raw_framewise_min_distance_scores(audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
    # score = - min_{s,t} ||a_s - v_t||^2
    diff = audio[:, None, :, None, :] - video[None, :, None, :, :]
    d2 = (diff * diff).sum(dim=-1)
    return -d2.amin(dim=(2, 3))


def raw_oracle_lag_scores(audio: torch.Tensor, video: torch.Tensor, max_lag: int) -> torch.Tensor:
    a = F.normalize(audio, dim=-1)
    best = None
    for delta in range(-max_lag, max_lag + 1):
        shifted_v = F.normalize(torch.roll(video, shifts=delta, dims=1), dim=-1)
        score = torch.einsum("qtd,ctd->qc", a, shifted_v)
        best = score if best is None else torch.maximum(best, score)
    return best


def format_metric_line(name: str, metrics: Dict[str, Dict[str, float] | float]) -> str:
    a2v = metrics["audio_to_video"]
    v2a = metrics["video_to_audio"]
    return (
        f"{name:28s} | mean_R@1={metrics['mean_R@1']:.3f}  mean_MRR={metrics['mean_MRR']:.3f}"
        f"  | A->V R@1={a2v['R@1']:.3f}, V->A R@1={v2a['R@1']:.3f}"
    )


def sample_indices(pool: torch.Tensor, n: int, g: torch.Generator) -> torch.Tensor:
    if n <= 0:
        return torch.empty(0, dtype=torch.long)
    if pool.numel() == 0:
        raise ValueError("Cannot sample from an empty pool.")
    if pool.numel() >= n:
        perm = torch.randperm(pool.numel(), generator=g)[:n]
        return pool[perm]
    extra = pool[torch.randint(0, pool.numel(), (n - pool.numel(),), generator=g)]
    return torch.cat([pool, extra], dim=0)


def hard_negative_metrics(
    scores_a2v: torch.Tensor,
    split: SplitData,
    mode: str,
    num_candidates: int,
    seed: int,
    overlap_threshold: float = 0.5,
) -> Dict[str, Dict[str, float] | float]:
    n = split.size()
    if num_candidates <= 1:
        raise ValueError("num_candidates must be >= 2")
    num_neg = num_candidates - 1
    g = make_cpu_generator(seed)

    z_overlap = (split.z[:, None, :] == split.z[None, :, :]).float().mean(dim=-1)
    all_idx = torch.arange(n, dtype=torch.long)

    def build_negatives(i: int, direction: str) -> torch.Tensor:
        not_self = all_idx[all_idx != i]
        if mode == "random":
            return sample_indices(not_self, num_neg, g)

        if mode == "same_nuisance":
            if direction == "a2v":
                mask = split.b_id == split.b_id[i]
            else:
                mask = split.s_id == split.s_id[i]
            mask[i] = False
            pool = all_idx[mask]
            if pool.numel() == 0:
                pool = not_self
            return sample_indices(pool, num_neg, g)

        if mode == "overlap":
            mask = z_overlap[i] >= overlap_threshold
            mask[i] = False
            pool = all_idx[mask]
            if pool.numel() == 0:
                pool = not_self
            return sample_indices(pool, num_neg, g)

        if mode == "mixed":
            half = num_neg // 2

            if direction == "a2v":
                same_mask = split.b_id == split.b_id[i]
            else:
                same_mask = split.s_id == split.s_id[i]
            same_mask[i] = False
            same_pool = all_idx[same_mask]
            if same_pool.numel() == 0:
                same_pool = not_self

            overlap_mask = z_overlap[i] >= overlap_threshold
            overlap_mask[i] = False
            overlap_pool = all_idx[overlap_mask]
            if overlap_pool.numel() == 0:
                overlap_pool = not_self

            part1 = sample_indices(same_pool, half, g)
            part2 = sample_indices(overlap_pool, num_neg - half, g)
            out = torch.cat([part1, part2], dim=0)
            if out.numel() < num_neg:
                filler = sample_indices(not_self, num_neg - out.numel(), g)
                out = torch.cat([out, filler], dim=0)
            return out

        raise ValueError(f"Unsupported hard-negative mode: {mode}")

    def direction_ranks(direction: str) -> torch.Tensor:
        ranks = []
        for i in range(n):
            neg = build_negatives(i, direction=direction)
            candidates = torch.cat([torch.tensor([i], dtype=torch.long), neg], dim=0)

            if direction == "a2v":
                cand_scores = scores_a2v[i, candidates]
            else:
                cand_scores = scores_a2v[candidates, i]

            pos_score = cand_scores[0]
            rank = 1 + int((cand_scores[1:] > pos_score).sum().item())
            ranks.append(rank)
        return torch.tensor(ranks, dtype=torch.long)

    a2v_ranks = direction_ranks("a2v")
    v2a_ranks = direction_ranks("v2a")

    def ranks_to_metrics(ranks: torch.Tensor) -> Dict[str, float]:
        ranks_f = ranks.float()
        return {
            "R@1": float((ranks <= 1).float().mean().item()),
            "R@5": float((ranks <= 5).float().mean().item()),
            "MRR": float((1.0 / ranks_f).mean().item()),
            "MedianRank": float(ranks_f.median().item()),
        }

    a2v = ranks_to_metrics(a2v_ranks)
    v2a = ranks_to_metrics(v2a_ranks)
    return {
        "audio_to_video": a2v,
        "video_to_audio": v2a,
        "mean_R@1": 0.5 * (a2v["R@1"] + v2a["R@1"]),
        "mean_MRR": 0.5 * (a2v["MRR"] + v2a["MRR"]),
    }


def lag_curve(
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    gen_cfg: GeneratorConfig,
    params: GeneratorParams,
    test_size: int,
    max_lag: int,
    seed: int,
    eval_batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    curve = {}
    for abs_tau in range(max_lag + 1):
        taus = [0] if abs_tau == 0 else [-abs_tau, abs_tau]
        r1_values = []
        for idx, tau in enumerate(taus):
            split = make_split(
                num_samples=test_size,
                cfg=gen_cfg,
                params=params,
                seed=seed + 101 * (abs_tau + 1) + idx,
                fixed_tau=tau,
                nuisance_strength=gen_cfg.nuisance_strength,
            )
            scores = score_matrix_batched(
                score_fn=score_fn,
                audio=split.audio,
                video=split.video,
                device=device,
                batch_size=eval_batch_size,
            )
            metrics = bidirectional_metrics(scores)
            r1_values.append(metrics["mean_R@1"])
        curve[str(abs_tau)] = float(sum(r1_values) / len(r1_values))
    return curve


def print_lag_curve(curve: Dict[str, float], name: str) -> None:
    ordered = sorted((int(k), v) for k, v in curve.items())
    pairs = ", ".join(f"|tau|={k}: {v:.3f}" for k, v in ordered)
    print(f"  {name:24s} | {pairs}")


def run_component_ablations(
    gen_cfg: GeneratorConfig,
    params: GeneratorParams,
    base_seed: int,
    train_cfg: TrainConfig,
    train_size: int,
    test_size: int,
    max_lag: int,
    eval_batch_size: int,
    device: torch.device,
) -> Dict[str, Dict[str, float] | Dict[str, Dict[str, float] | float]]:
    ablations = [
        ("no_nuisance", 0.0, [0], 0.0),
        ("nuisance_only", gen_cfg.nuisance_strength, [0], 0.0),
        ("lag_only", 0.0, list(range(-max_lag, max_lag + 1)), 0.0),
        ("nuisance_lag_hard", gen_cfg.nuisance_strength, list(range(-max_lag, max_lag + 1)), train_cfg.hard_negative_rate),
    ]

    out: Dict[str, Dict[str, float] | Dict[str, Dict[str, float] | float]] = {}
    for idx, (name, n_strength, lags, hard_rate) in enumerate(ablations):
        print(f"\n[Ablation] {name}")
        split_train = make_split(
            num_samples=train_size,
            cfg=gen_cfg,
            params=params,
            seed=base_seed + 700 + idx * 10,
            lag_values=lags,
            nuisance_strength=n_strength,
        )
        split_test = make_split(
            num_samples=test_size,
            cfg=gen_cfg,
            params=params,
            seed=base_seed + 701 + idx * 10,
            lag_values=lags,
            nuisance_strength=n_strength,
        )

        model = LagAwareEnergyModel(d_in=gen_cfg.d_obs, d_hidden=64, max_lag=max_lag)
        this_train_cfg = replace(train_cfg, hard_negative_rate=hard_rate)
        train_model(model, split_train=split_train, cfg=this_train_cfg, device=device)

        score_fn = lambda a, v, m=model: m.similarity_matrix(a, v)
        scores = score_matrix_batched(
            score_fn=score_fn,
            audio=split_test.audio,
            video=split_test.video,
            device=device,
            batch_size=eval_batch_size,
        )
        metrics = bidirectional_metrics(scores)
        print(format_metric_line("lag-aware energy", metrics))
        out[name] = {
            "mean_R@1": metrics["mean_R@1"],
            "mean_MRR": metrics["mean_MRR"],
            "metrics": metrics,
        }

    return out


def run(args: argparse.Namespace) -> Dict[str, object]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"device={device}")

    max_lag = args.max_lag
    lag_values = tuple(range(-max_lag, max_lag + 1))
    gen_cfg = GeneratorConfig(
        K=args.K,
        T=args.T,
        d_obs=args.d_obs,
        d_event=args.d_event,
        d_nuisance=args.d_nuisance,
        n_speakers=args.n_speakers,
        n_backgrounds=args.n_backgrounds,
        noise_std=args.noise_std,
        nuisance_strength=args.nuisance_strength,
        event_stay_prob=args.event_stay_prob,
        lag_values=lag_values,
    )

    params = build_generator_params(gen_cfg, seed=args.seed + 1)

    print("\n[Data] Generating train/val/test splits")
    split_train = make_split(
        num_samples=args.train_size,
        cfg=gen_cfg,
        params=params,
        seed=args.seed + 10,
    )
    split_val = make_split(
        num_samples=args.val_size,
        cfg=gen_cfg,
        params=params,
        seed=args.seed + 11,
    )
    split_test = make_split(
        num_samples=args.test_size,
        cfg=gen_cfg,
        params=params,
        seed=args.seed + 12,
    )
    _ = split_val  # reserved for further tuning hooks

    baseline_train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        hard_negative_rate=0.0,
        hard_negative_weight=args.hard_negative_weight,
        hard_margin=args.hard_margin,
        overlap_threshold=args.overlap_threshold,
    )
    energy_train_cfg = replace(
        baseline_train_cfg,
        hard_negative_rate=args.hard_negative_rate,
    )

    print("\n[Train] Dual-encoder pooled cosine (InfoNCE)")
    pooled_model = PooledCosineDualEncoder(d_in=gen_cfg.d_obs, d_hidden=64)
    pooled_train_stats = train_model(
        pooled_model,
        split_train=split_train,
        cfg=baseline_train_cfg,
        device=device,
    )

    print("\n[Train] Lag-aware bilinear energy (InfoNCE + optional hard negatives)")
    energy_model = LagAwareEnergyModel(d_in=gen_cfg.d_obs, d_hidden=64, max_lag=max_lag)
    energy_train_stats = train_model(
        energy_model,
        split_train=split_train,
        cfg=energy_train_cfg,
        device=device,
    )

    pooled_model.eval()
    energy_model.eval()

    print("\n[Eval] Bidirectional retrieval on base test split")
    pooled_scores = score_matrix_batched(
        score_fn=lambda a, v: pooled_model.similarity_matrix(a, v),
        audio=split_test.audio,
        video=split_test.video,
        device=device,
        batch_size=args.eval_batch_size,
    )
    energy_scores = score_matrix_batched(
        score_fn=lambda a, v: energy_model.similarity_matrix(a, v),
        audio=split_test.audio,
        video=split_test.video,
        device=device,
        batch_size=args.eval_batch_size,
    )
    raw_pooled_scores = score_matrix_batched(
        score_fn=raw_pooled_cosine_scores,
        audio=split_test.audio,
        video=split_test.video,
        device=device,
        batch_size=args.eval_batch_size,
    )
    raw_min_scores = score_matrix_batched(
        score_fn=raw_framewise_min_distance_scores,
        audio=split_test.audio,
        video=split_test.video,
        device=device,
        batch_size=args.eval_batch_size,
    )
    raw_oracle_scores = score_matrix_batched(
        score_fn=lambda a, v: raw_oracle_lag_scores(a, v, max_lag=max_lag),
        audio=split_test.audio,
        video=split_test.video,
        device=device,
        batch_size=args.eval_batch_size,
    )

    eval_metrics = {
        "dual_encoder_pooled_cosine": bidirectional_metrics(pooled_scores),
        "lag_aware_energy": bidirectional_metrics(energy_scores),
        "raw_pooled_cosine": bidirectional_metrics(raw_pooled_scores),
        "raw_framewise_min_distance": bidirectional_metrics(raw_min_scores),
        "raw_oracle_lag": bidirectional_metrics(raw_oracle_scores),
    }

    for name, metrics in eval_metrics.items():
        print(format_metric_line(name, metrics))

    hard_eval = {}
    if args.run_hard_negative_eval:
        print("\n[Eval] Hard-negative robustness")
        hard_modes = ["random", "same_nuisance", "overlap", "mixed"]
        for mode_idx, mode in enumerate(hard_modes):
            hard_eval.setdefault("lag_aware_energy", {})[mode] = hard_negative_metrics(
                scores_a2v=energy_scores,
                split=split_test,
                mode=mode,
                num_candidates=args.hard_eval_candidates,
                seed=args.seed + 200 + mode_idx,
                overlap_threshold=args.overlap_threshold,
            )
            hard_eval.setdefault("dual_encoder_pooled_cosine", {})[mode] = hard_negative_metrics(
                scores_a2v=pooled_scores,
                split=split_test,
                mode=mode,
                num_candidates=args.hard_eval_candidates,
                seed=args.seed + 300 + mode_idx,
                overlap_threshold=args.overlap_threshold,
            )
            print(
                f"  mode={mode:14s}"
                f"  energy mean_R@1={hard_eval['lag_aware_energy'][mode]['mean_R@1']:.3f}"
                f"  dual-enc mean_R@1={hard_eval['dual_encoder_pooled_cosine'][mode]['mean_R@1']:.3f}"
            )

    lag_curves = {}
    if args.run_lag_curve:
        print("\n[Eval] Lag robustness curve (mean R@1 vs |tau|)")
        lag_curves["lag_aware_energy"] = lag_curve(
            score_fn=lambda a, v: energy_model.similarity_matrix(a, v),
            gen_cfg=gen_cfg,
            params=params,
            test_size=args.lag_curve_size,
            max_lag=max_lag,
            seed=args.seed + 400,
            eval_batch_size=args.eval_batch_size,
            device=device,
        )
        lag_curves["dual_encoder_pooled_cosine"] = lag_curve(
            score_fn=lambda a, v: pooled_model.similarity_matrix(a, v),
            gen_cfg=gen_cfg,
            params=params,
            test_size=args.lag_curve_size,
            max_lag=max_lag,
            seed=args.seed + 500,
            eval_batch_size=args.eval_batch_size,
            device=device,
        )
        print_lag_curve(lag_curves["lag_aware_energy"], "lag-aware energy")
        print_lag_curve(lag_curves["dual_encoder_pooled_cosine"], "dual-encoder pooled")

    ablation_results = {}
    if args.run_ablations:
        print("\n[Eval] Component ablations")
        ablation_cfg = replace(
            energy_train_cfg,
            epochs=args.ablation_epochs,
            batch_size=args.ablation_batch_size,
        )
        ablation_results = run_component_ablations(
            gen_cfg=gen_cfg,
            params=params,
            base_seed=args.seed,
            train_cfg=ablation_cfg,
            train_size=args.ablation_train_size,
            test_size=args.ablation_test_size,
            max_lag=max_lag,
            eval_batch_size=args.eval_batch_size,
            device=device,
        )

    result = {
        "config": {
            "generator": {
                "K": gen_cfg.K,
                "T": gen_cfg.T,
                "d_obs": gen_cfg.d_obs,
                "d_event": gen_cfg.d_event,
                "d_nuisance": gen_cfg.d_nuisance,
                "lag_values": list(gen_cfg.lag_values),
                "noise_std": gen_cfg.noise_std,
                "nuisance_strength": gen_cfg.nuisance_strength,
            },
            "train": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "temperature": args.temperature,
                "hard_negative_rate": args.hard_negative_rate,
                "hard_negative_weight": args.hard_negative_weight,
            },
            "splits": {
                "train": args.train_size,
                "val": args.val_size,
                "test": args.test_size,
            },
        },
        "train_stats": {
            "dual_encoder_pooled_cosine": pooled_train_stats,
            "lag_aware_energy": energy_train_stats,
        },
        "retrieval_metrics": eval_metrics,
        "hard_negative_eval": hard_eval,
        "lag_curves": lag_curves,
        "ablations": ablation_results,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved results to {output_path}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lagged shared-source retrieval experiment")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Generator
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--d-obs", type=int, default=32)
    parser.add_argument("--d-event", type=int, default=16)
    parser.add_argument("--d-nuisance", type=int, default=8)
    parser.add_argument("--n-speakers", type=int, default=32)
    parser.add_argument("--n-backgrounds", type=int, default=32)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--nuisance-strength", type=float, default=1.0)
    parser.add_argument("--event-stay-prob", type=float, default=0.65)
    parser.add_argument("--max-lag", type=int, default=2)

    # Split sizes
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--val-size", type=int, default=1_000)
    parser.add_argument("--test-size", type=int, default=1_000)

    # Optimization
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hard-negative-rate", type=float, default=0.35)
    parser.add_argument("--hard-negative-weight", type=float, default=0.25)
    parser.add_argument("--hard-margin", type=float, default=0.2)
    parser.add_argument("--overlap-threshold", type=float, default=0.5)

    # Evaluation controls
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--run-hard-negative-eval", action="store_true")
    parser.add_argument("--hard-eval-candidates", type=int, default=100)
    parser.add_argument("--run-lag-curve", action="store_true")
    parser.add_argument("--lag-curve-size", type=int, default=500)
    parser.add_argument("--run-ablations", action="store_true")
    parser.add_argument("--ablation-train-size", type=int, default=4_000)
    parser.add_argument("--ablation-test-size", type=int, default=600)
    parser.add_argument("--ablation-epochs", type=int, default=8)
    parser.add_argument("--ablation-batch-size", type=int, default=128)

    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

