# Lagged Shared-Source Retrieval (Synthetic)

This experiment is a minimal controlled test for cross-modal retrieval:

- Shared latent event sequence drives both audio/video.
- Audio and video each get modality-private nuisance.
- Video is temporally shifted by a controllable lag.

## What it includes

- Generator:
  - `a_t = U_a e(z_t) + S_a s + eps_a`
  - `v_t = U_v e(z_{t-tau}) + B_v b + eps_v`
- Trainable models:
  - Dual-encoder pooled cosine (`InfoNCE`)
  - Lag-aware bilinear energy model (`InfoNCE` + optional hard-negative hinge)
- Evaluation baselines (non-trainable on raw features):
  - Pooled cosine
  - Framewise min-distance
  - Oracle lag cross-correlation
- Metrics:
  - Bidirectional retrieval (`R@1`, `R@5`, `MRR`)
  - Optional lag robustness curve
  - Optional hard-negative robustness
  - Optional component ablations

## Run

Full-sized defaults (10k/1k/1k):

```bash
python experiments/lagged_shared_source/run_lagged_shared_source.py
```

Smoke test:

```bash
python experiments/lagged_shared_source/run_lagged_shared_source.py \
  --train-size 512 --val-size 128 --test-size 128 --epochs 2 --device cpu
```

Enable extra diagnostics:

```bash
python experiments/lagged_shared_source/run_lagged_shared_source.py \
  --run-hard-negative-eval --run-lag-curve --run-ablations
```

Save metrics:

```bash
python experiments/lagged_shared_source/run_lagged_shared_source.py \
  --output-json experiments/lagged_shared_source/results.json
```

