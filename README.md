# MET — Generative Multimodal Energy Transformers

PyTorch implementation of the Generative Multimodal Energy Transformer (MET).

## Structure

```
met/core/       - Energy primitives (basis caches, attention, Hopfield, top-level energy)
met/solver/     - Inference solvers (deterministic, Langevin, EqProp)
met/training/   - Losses, EMA teacher, training loop
met/tokenizers/ - Video and audio front-ends
met/heads/      - Output heads (audio, video)
met/utils/      - Diagnostics and gradient checking
experiments/    - Tier 0–5 experiment runners
  └── lagged_shared_source/ - synthetic cross-modal retrieval stress test
tests/          - Unit + integration tests (run first)
configs/        - OmegaConf YAML configs
```

## Quick start

```bash
pip install -r requirements.txt
pytest tests/ -v                          # must pass before any training
python experiments/tier0/exp0_1_grad_check_intra.py
```

## Modular switches

- Basis family: set `METConfig.basis_v` / `METConfig.basis_a` to `"bspline"` or `"fourier"`.
- Training rule: use `METTrainer.train_epoch(..., method="bptt")` or `method="eqprop"`.
- EqProp nudge: pass a task-head reconstruction objective into `EqPropEstimator.estimate_gradient(..., nudge_objective=...)`.

## Implementation order

See `implementation_plan.md` for the full week-by-week plan.
BPTT first. EqProp after BPTT is stable. Tier 5 benchmarks last.
