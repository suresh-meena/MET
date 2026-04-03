# MET — Comprehensive PyTorch Implementation Plan
**Generative Multimodal Energy Transformers**

> Based on `writeup/main.tex` (corrected proposal) + `writeup/experiments.md` (validation protocol).
> Target: `/home/suresh/Documents/multimodal/MET/`

---

## Proposal Review: No Major Logical Errors Found

The paper is self-aware and internally consistent. Five caveats must shape all implementation decisions:

| Item | What the paper says | Implementation consequence |
|---|---|---|
| LayerNorm stability | Only `Ė ≤ 0` proved; null-space gap is open | Always log energy per step; never assume convergence |
| EqProp validity | Breaks when free/nudged phases hit different attractors | Attractor-agreement diagnostic from day 1 |
| Cross-modal gradient | Zero w.r.t. current modality **only** in conditional/alternating mode | Implement alternating-update flag explicitly |
| B-spline key-path gradient | Shared key-path accumulates across **all** queries, not just same-token | Use autograd; never hand-derive this term |
| Langevin vs MAP | Finite-run Langevin ≠ posterior sampling in multi-attractor regime | Separate `--mode langevin` / `--mode map` at inference |

---

## Project Layout

```
MET/
├── met/
│   ├── __init__.py
│   ├── tokenizers/
│   │   ├── video_tokenizer.py       # VideoTokenizer: patches -> (B, L, D_tok)
│   │   └── audio_tokenizer.py       # AudioTokenizer: mel -> (B, L, D_tok)
│   ├── core/
│   │   ├── layernorm.py             # TokenwiseLayerNorm (per-token, norm over D)
│   │   ├── spline.py                # BSplineCache: F, R, F_quad (precomputed buffers)
│   │   ├── attention.py             # ContinuousAttentionEnergy (intra + cross)
│   │   ├── hopfield.py              # HopfieldMemoryBank (per-modality)
│   │   └── energy.py               # METEnergy: top-level E(x^v, x^a) scalar
│   ├── solver/
│   │   ├── gradient_descent.py      # Deterministic T-step unrolled solver
│   │   ├── langevin.py              # ULA sampler (Langevin inference)
│   │   └── eqprop.py               # EqProp: free-phase + nudged-phase runner
│   ├── training/
│   │   ├── losses.py               # J_mel, J_sync, J_JEPA, J_sem, J_temp, J_rank
│   │   ├── jepa.py                 # EMA teacher + predictor heads
│   │   └── trainer.py              # Training loop (BPTT-first, then EqProp)
│   ├── heads/
│   │   ├── audio_head.py           # x_a(T) -> mel / codec latent
│   │   └── video_head.py           # x_v(T) -> visual latent
│   └── utils/
│       ├── diagnostics.py          # Energy logger, FP residual, attractor agreement
│       └── grad_check.py           # Finite-difference gradient verifier
├── experiments/
│   ├── tier0/   (exp0_1_grad_intra.py, exp0_2_grad_cross.py, exp0_3_monotonicity.py)
│   ├── tier1/   (exp1_1_unimodal.py, exp1_2_degradation.py, exp1_3_window.py)
│   ├── tier2/   (exp2_1_spline_error.py, exp2_2_ringing.py, exp2_3_quadrature.py)
│   ├── tier3/   (exp3_1_eqprop_vs_bptt.py, exp3_2_memory.py)
│   ├── tier4/   (exp4_1_linear.py, exp4_2_symmetry.py, exp4_3_ranking.py)
│   └── tier5/   (bench5_1_ave.py, bench5_2_foley.py, bench5_3_eqprop_real.py, bench5_4_jepa.py)
├── configs/
│   ├── base.yaml
│   ├── foley.yaml
│   └── jepa.yaml
├── tests/
│   ├── test_spline.py
│   ├── test_attention.py
│   ├── test_hopfield.py
│   └── test_energy.py
├── requirements.txt
└── README.md
```

---

## Phase 0 — Environment

**`requirements.txt`:**
```
torch>=2.2.0
torchvision
torchaudio
numpy
scipy          # Gauss-Legendre quadrature: roots_legendre(M)
einops
omegaconf
wandb
pytest
```

---

## Phase 1 — Core Mathematical Primitives

### 1.1 `met/core/layernorm.py` — Tokenwise LayerNorm

**Math:** `g_{li} = gamma_i * (x_li - x_bar_l) / sqrt(sigma_l + eps) + delta_i`

```python
class TokenwiseLayerNorm(nn.Module):
    """
    Per-token LayerNorm over D. nn.LayerNorm(D) with input (B, L, D) is exact.
    
    STABILITY NOTE: J_l @ 1 = 0 (mean-subtraction null space)
    => J_l J_l^T is PSD not PD
    => Energy descent gives E_dot <= 0, NOT E_dot < 0 (Lyapunov gap)
    """
    def __init__(self, D: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(D, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.ln(x)   # (B, L, D) -> (B, L, D)
```

---

### 1.2 `met/core/spline.py` — B-Spline Cache

**Math:**
```
F_m in R^{L x N}:  [F_m]_{lj} = phi_j(l/L)            (basis matrix)
R_m = (F_m^T F_m + lambda*I)^{-1} F_m^T  in R^{N x L}  (ONE-TIME solve)
C_bar_h = R_m @ K_h  in R^{N x D_k}                    (per-step)
F_quad in R^{M x N}: basis at Gauss-Legendre nodes
K_quad = F_quad @ C_bar  in R^{M x D_k}                 (for log-partition)
```

**Complexity:** One-time `O(LN^2 + N^3)`. Per-step: `O(H*L*N*D_k + H*L*M*D_k)`.

```python
class BSplineCache(nn.Module):
    """Precomputes F, R, F_quad as buffers (no grad, no update).
    
    encode(K: B,H,L,D_k) -> C_bar (B,H,N,D_k), K_quad (B,H,M,D_k)
    """
    def __init__(self, L, N, M, lam=1e-3, degree=3):
        super().__init__()
        t_tokens = torch.linspace(0, 1, L)
        F = self._build_basis(t_tokens, N, degree)          # (L, N)
        G = F.T @ F + lam * torch.eye(N)
        R = torch.linalg.solve(G, F.T)                     # (N, L)

        from scipy.special import roots_legendre
        xi, wi = roots_legendre(M)
        t_quad = torch.tensor((xi + 1) / 2, dtype=torch.float32)   # [0,1]
        w_quad = torch.tensor(wi / 2, dtype=torch.float32)
        F_quad = self._build_basis(t_quad, N, degree)              # (M, N)

        for name, buf in [('F',F),('R',R),('t_quad',t_quad),
                           ('w_quad',w_quad),('F_quad',F_quad)]:
            self.register_buffer(name, buf)

    def _build_basis(self, t, N, degree):
        import numpy as np
        from scipy.interpolate import BSpline
        knots = np.linspace(0, 1, N - degree + 1)
        knots = np.concatenate([[0]*degree, knots, [1]*degree])
        cols = []
        for i in range(N):
            c = np.zeros(N); c[i] = 1.0
            cols.append(BSpline(knots, c, degree)(t.numpy()))
        return torch.tensor(np.stack(cols, axis=1), dtype=torch.float32)

    def encode(self, K):
        """K: (B, H, L, D_k) -> C_bar (B,H,N,D_k), K_quad (B,H,M,D_k)"""
        C_bar  = torch.einsum('nl,bhld->bhnd', self.R, K)
        K_quad = torch.einsum('mn,bhnd->bhmd', self.F_quad, C_bar)
        return C_bar, K_quad
```

> [!IMPORTANT]
> `R` and `F_quad` are **buffers** — cached once at init, never recomputed during training/inference. This is the key efficiency claim in the paper.

---

### 1.3 `met/core/attention.py` — Continuous Attention Energy

**Math (intra-modal):**
```
E_m^intra = -(1/beta) * sum_h sum_l log sum_r omega_r * exp(beta * <Q_hl, K_quad_hr>)
```
**Math (cross-modal, bidirectional):**
```
E_cross = E_{v->a} + E_{a->v}
  = -(1/beta) * sum_h sum_l [log sum_r omega_r exp(beta <Q_hl^v, K_quad_hr^a>)
                             + log sum_r omega_r exp(beta <Q_hl^a, K_quad_hr^v>)]
```

```python
class ContinuousAttentionEnergy(nn.Module):
    def __init__(self, D, D_k, H, cache_v, cache_a, beta=1.0):
        super().__init__()
        self.H, self.D_k, self.beta = H, D_k, beta
        # Shared Q/K projections (both modalities use same weight matrices)
        self.W_Q = nn.Linear(D, H * D_k, bias=False)
        self.W_K = nn.Linear(D, H * D_k, bias=False)
        self.cache_v = cache_v
        self.cache_a = cache_a

    def _log_partition(self, Q, K_quad, w):
        """
        Q:      (B, L, H, D_k)
        K_quad: (B, H, M, D_k)
        w:      (M,)   Gauss-Legendre weights (all positive)
        
        Returns scalar: -(1/beta) * sum_h sum_l log sum_r omega_r exp(beta * <Q_hl, K_r>)
        
        Trick: log sum_r omega_r exp(s_r) = logsumexp(s_r + log(omega_r))
        """
        scores = torch.einsum('blhd,bhmd->blhm', Q, K_quad) * self.beta   # (B, L, H, M)
        log_Z = torch.logsumexp(scores + w.log(), dim=-1)   # (B, L, H)
        return -log_Z.sum() / self.beta

    def forward(self, g_v, g_a):
        """g_v, g_a: (B, L, D) -> E_intra_v, E_intra_a, E_cross (scalars)"""
        B, L, D = g_v.shape

        def proj_and_encode(g, cache):
            Q = self.W_Q(g).view(B, L, self.H, self.D_k)   # (B,L,H,D_k)
            K = self.W_K(g).view(B, L, self.H, self.D_k)
            _, K_quad = cache.encode(K.permute(0,2,1,3))    # (B,H,M,D_k)
            return Q, K_quad

        Q_v, Kq_v = proj_and_encode(g_v, self.cache_v)
        Q_a, Kq_a = proj_and_encode(g_a, self.cache_a)
        w = self.cache_v.w_quad

        E_iv = self._log_partition(Q_v, Kq_v, w)
        E_ia = self._log_partition(Q_a, Kq_a, w)
        E_cross = (self._log_partition(Q_v, Kq_a, w)    # v queries a
                 + self._log_partition(Q_a, Kq_v, w))   # a queries v
        return E_iv, E_ia, E_cross
```

> [!CAUTION]
> **Never hand-code the cross-token key-path gradient** (Appendix A, second sum). Autograd through `R @ K_h` handles all-query accumulation automatically. Exp 0.1 is the verification gate.

---

### 1.4 `met/core/hopfield.py` — Hopfield Memory Bank

**Math:**
```
g_bar_{m',l} = avg_{j in N(l)} g_{m',j}                (local temporal smooth)
c_{m'->m,l}  = W_{m'->m} @ g_bar_{m',l}                (cross-modal projection)
s_{lu}^m     = lambda_cross * <c, xi_u> + (1-lambda) * <g_m,l, xi_u>
E_m^HN       = -(1/beta_HN) * sum_l log sum_u exp(beta_HN * s_{lu}^m)
```

```python
class HopfieldMemoryBank(nn.Module):
    def __init__(self, D, D_cross, K, beta_HN=1.0,
                 lambda_cross=0.05, window=3):
        super().__init__()
        self.K = K
        self.beta_HN = beta_HN
        self.window = window
        # lambda_cross: learnable scalar, clamped to [0,1] in forward
        self.lambda_cross = nn.Parameter(torch.tensor(lambda_cross))
        # Prototypes: normalized at use time (not stored)
        self.Xi = nn.Parameter(F.normalize(torch.randn(K, D), dim=-1))
        self.W_cross = nn.Linear(D_cross, D, bias=False)

    def _smooth(self, g):
        """avg_pool1d for temporal smoothing. g: (B,L,D) -> (B,L,D)"""
        w = self.window
        return F.avg_pool1d(
            g.permute(0,2,1), kernel_size=w, stride=1,
            padding=w//2, count_include_pad=False
        ).permute(0,2,1)

    def forward(self, g_m, g_cross):
        """g_m: (B,L,D), g_cross: (B,L,D_cross) -> scalar E_m^HN"""
        lam = self.lambda_cross.clamp(0.0, 1.0)
        c = self.W_cross(self._smooth(g_cross))          # (B, L, D)
        Xi_n = F.normalize(self.Xi, dim=-1)              # (K, D)
        s = (lam   * torch.einsum('bld,kd->blk', c, Xi_n)
           +(1-lam) * torch.einsum('bld,kd->blk', g_m, Xi_n))   # (B, L, K)
        log_Z = torch.logsumexp(self.beta_HN * s, dim=-1)        # (B, L)
        return -log_Z.sum() / self.beta_HN
```

**Design decisions:**
- `lambda_cross` is `nn.Parameter` — gets gradient; clamped in forward to stay `[0,1]`.
- `F.avg_pool1d` with `count_include_pad=False` avoids boundary weight dilution.
- Prototypes L2-normalized at use time (not stored), preventing norm collapse.

---

### 1.5 `met/core/energy.py` — Top-level MET Energy

```python
class METEnergy(nn.Module):
    """E(x^v, x^a) = E_iv + E_ia + E_cross + E_HN_v + E_HN_a  (scalar)"""
    def __init__(self, cfg):
        super().__init__()
        D = cfg.D
        self.ln_v = TokenwiseLayerNorm(D)
        self.ln_a = TokenwiseLayerNorm(D)
        cache_v = BSplineCache(cfg.L, cfg.N_v, cfg.M_v, cfg.lam_spline)
        cache_a = BSplineCache(cfg.L, cfg.N_a, cfg.M_a, cfg.lam_spline)
        self.attention = ContinuousAttentionEnergy(
            D, cfg.D_k, cfg.H, cache_v, cache_a, cfg.beta)
        self.hopfield_v = HopfieldMemoryBank(
            D, D, cfg.K_v, cfg.beta_HN, cfg.lambda_cross, cfg.window)
        self.hopfield_a = HopfieldMemoryBank(
            D, D, cfg.K_a, cfg.beta_HN, cfg.lambda_cross, cfg.window)

    def forward(self, x_v, x_a, freeze_v=False, freeze_a=False):
        """
        freeze_v/freeze_a: conditional generation mode.
        Use .detach() on state — NOT no_grad() — so parameter grads still flow.
        """
        if freeze_v: x_v = x_v.detach()
        if freeze_a: x_a = x_a.detach()

        g_v = self.ln_v(x_v)
        g_a = self.ln_a(x_a)

        E_iv, E_ia, E_cross = self.attention(g_v, g_a)
        E_hv = self.hopfield_v(g_v, g_a)
        E_ha = self.hopfield_a(g_a, g_v)

        E_total = E_iv + E_ia + E_cross + E_hv + E_ha
        comps = dict(E_iv=E_iv.item(), E_ia=E_ia.item(), E_cross=E_cross.item(),
                     E_hv=E_hv.item(), E_ha=E_ha.item(), E_total=E_total.item())
        return E_total, comps
```

> [!IMPORTANT]
> `freeze_v` uses `.detach()` on the **state input**, not `torch.no_grad()`. Parameter gradients still flow. This correctly implements conditional generation.

---

## Phase 2 — Solvers

### 2.1 Deterministic T-step Solver (`solver/gradient_descent.py`)

```python
def run_deterministic_solver(
    energy, x_v, x_a, T, eta,
    freeze_v=False, freeze_a=False,
    grad_clip=1.0, early_stop_eps=1e-6,
    create_graph=False,    # True for BPTT training phase
):
    x_v, x_a = x_v.clone(), x_a.clone()
    if not freeze_v: x_v.requires_grad_(True)
    if not freeze_a: x_a.requires_grad_(True)

    E_prev, logs = None, []
    for t in range(T):
        active = [x for x in [x_v, x_a] if x.requires_grad]
        E, comps = energy(x_v, x_a, freeze_v=freeze_v, freeze_a=freeze_a)
        grads = torch.autograd.grad(E, active, create_graph=create_graph)

        fp_residual = sum(g.norm().item() for g in grads)
        comps.update(step=t, fp_residual=fp_residual,
                     monotone=(E_prev is None or E.item() <= E_prev))
        logs.append(comps)
        E_prev = E.item()

        with torch.no_grad():
            i = 0
            if not freeze_v:
                x_v = (x_v - eta * grads[i].clamp(-grad_clip, grad_clip)
                       ).detach().requires_grad_(True)
                i += 1
            if not freeze_a:
                x_a = (x_a - eta * grads[i].clamp(-grad_clip, grad_clip)
                       ).detach().requires_grad_(True)

        if fp_residual < early_stop_eps: break

    return x_v.detach(), x_a.detach(), logs
```

### 2.2 Langevin Sampler (`solver/langevin.py`)

```python
def run_langevin(energy, x_v, x_a, T, eta, beta_inv, freeze_v=True):
    """
    ULA: x^a(t+1) = x^a(t) - eta * grad_xa_E + sqrt(2*eta/beta) * eps
    
    WARNING: Does NOT guarantee posterior sampling in multi-attractor landscape.
    Use --mode langevin for exploration. For MAP use deterministic solver.
    """
    x_a = x_a.clone()
    noise_scale = (2 * eta * beta_inv) ** 0.5
    for _ in range(T):
        x_a_r = x_a.detach().requires_grad_(True)
        E, _ = energy(x_v, x_a_r, freeze_v=freeze_v)
        grad_a, = torch.autograd.grad(E, x_a_r)
        with torch.no_grad():
            x_a = x_a - eta * grad_a + noise_scale * torch.randn_like(x_a)
    return x_a.detach()
```

### 2.3 EqProp Estimator (`solver/eqprop.py`)

```python
class EqPropEstimator:
    """
    Free phase  -> x_free*  (deterministic solver)
    Nudged phase -> x_nudged* (warm-started from x_free*)
    Grad estimate = (grad_theta E(x_nudged*) - grad_theta E(x_free*)) / s
    
    Validity: free and nudged MUST converge to same attractor.
    Monitor attractor_agreement throughout training.
    
    ONLY J_gen (samplewise MSE) enters the nudge.
    J_sem, J_temp, J_JEPA are outer-loop auxiliaries.
    """
    def __init__(self, energy, s=0.01, T=50):
        self.energy, self.s, self.T = energy, s, T

    def estimate_gradient(self, x_v, x_a, x_a_target, eta, freeze_v=True):
        # Free phase
        x_v_f, x_a_f, _ = run_deterministic_solver(
            self.energy, x_v, x_a, self.T, eta, freeze_v=freeze_v)
        E_free, _ = self.energy(x_v_f, x_a_f, freeze_v=freeze_v)

        # Nudged phase: warm-start from x_free*
        J_nudge = F.mse_loss(x_a_f, x_a_target)
        nudge_delta, = torch.autograd.grad(J_nudge, x_a_f)
        x_a_n_init = (x_a_f + self.s * nudge_delta).detach()
        x_v_n, x_a_n, _ = run_deterministic_solver(
            self.energy, x_v_f, x_a_n_init, self.T, eta, freeze_v=freeze_v)
        E_nudged, _ = self.energy(x_v_n, x_a_n, freeze_v=freeze_v)

        # EqProp gradient estimate
        params = list(self.energy.parameters())
        gn = torch.autograd.grad(E_nudged, params, allow_unused=True)
        gf = torch.autograd.grad(E_free,   params, allow_unused=True)
        eqprop_grads = {
            name: (gni - gfi) / self.s if (gni is not None and gfi is not None) else None
            for (name, _), gni, gfi in zip(self.energy.named_parameters(), gn, gf)
        }

        # Attractor agreement diagnostic
        agreement = F.cosine_similarity(
            x_a_f.flatten(1), x_a_n.flatten(1), dim=-1).mean().item()

        return dict(eqprop_grads=eqprop_grads,
                    attractor_agreement=agreement,
                    E_free=E_free.item(), E_nudged=E_nudged.item())
```

---

## Phase 3 — Tokenizers & Projection

**Timeline alignment:** Both modalities onto `L` windows per clip.
- Audio: 16 kHz, hop=160 -> 100 fps -> resample to `L` with `F.interpolate(..., mode='linear')`
- Video: 25 fps -> resample to `L` with `F.interpolate(..., mode='bilinear')`

**Projection into shared width `D`:**
```python
x_v_0 = U_v(z_v) + pos_emb_v   # U_v: Linear(D_tok_v, D), pos_emb_v: (L, D)
x_a_0 = U_a(z_a) + pos_emb_a   # U_a: Linear(D_tok_a, D), pos_emb_a: (L, D)
```

Setting `D_v = D_a = D` simplifies cross-modal projections, memory banks, and batching. Modality-specific dimensions live only in tokenizers and output heads.

**Short-term (Tiers 0-4):** Use frozen ViT (DINOv2 / CLIP-ViT-B) for video, log-mel for audio.

---

## Phase 4 — Output Heads

```python
class AudioHead(nn.Module):
    """x_a(T): (B, L, D) -> mel_pred: (B, L, n_mels)"""
    def __init__(self, D, n_mels=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, n_mels))
    def forward(self, x): return self.proj(x)
```

Analogous `VideoHead`. These remain **outside** the energy — their job is to map converged latent states to measurable reconstruction targets.

---

## Phase 5 — Training

### 5.1 All Loss Terms (`training/losses.py`)

| Loss | Formula | Notes |
|---|---|---|
| `J_mel` | MSE(mel_pred, mel_gt) | Primary; weight `lambda_gen=1.0` |
| `J_sync` | `mean_w ||S_a(a_hat_w) - S_v(v_w)||^2` | AV-onset vs visual-motion |
| `J_JEPA` | `||P_m(g_tilde*) - sg(g_T*)||^2` | Stop-gradient on teacher targets |
| `J_sem` | InfoNCE on clip-pooled equilibria | Cross-batch negatives |
| `J_temp` | InfoNCE on window-level equilibria | Same-clip hard negatives (Foley) |
| `J_rank` | `softplus(E(x^v,x^a) - E(x_bar^v,x^a) + m)` | Matched < mismatched energy |

### 5.2 EMA Teacher (`training/jepa.py`)

```python
class JEPATeacher(nn.Module):
    def __init__(self, energy, mu=0.99):
        super().__init__()
        self.teacher = copy.deepcopy(energy)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.mu = mu

    @torch.no_grad()
    def update_ema(self, student):
        for tp, sp in zip(self.teacher.parameters(), student.parameters()):
            tp.data.mul_(self.mu).add_((1 - self.mu) * sp.data)

    def get_teacher_equilibrium(self, x_v, x_a, T, eta):
        with torch.no_grad():
            return run_deterministic_solver(self.teacher, x_v, x_a, T, eta)[:2]
```

### 5.3 Training Phases (Incremental)

```
Phase A  BPTT: unrolled T steps, J_gen only
         -> verify monotonicity + grad correctness (Tier 0 gates)

Phase B  Add Hopfield banks
         -> run Tier 1 unit tests

Phase C  Add J_JEPA + J_rank
         -> run Tier 2 + Tier 3 (EqProp vs BPTT)

Phase D  Switch primary updates to EqProp; keep BPTT as comparison

Phase E  Add Foley losses: J_sync + J_sem + J_temp

Phase F  Scale to Tier 5 real benchmarks
```

> [!IMPORTANT]
> "If the unrolled solver is unstable, EqProp will not rescue it." (paper §7). Start BPTT. Debug there.

---

## Phase 6 — Experiments (matching experiments.md exactly)

### Tier 0 — Gradient Correctness & Monotonicity (CPU, <30 min)

**Exp 0.1 — Intra-modal FD gradient check:**
- L=16, N=4, H=1, D_k=8, D=16
- autodiff vs central FD at eps=1e-4
- **Success:** relative error < 1e-3 at all token positions
- **If fail:** autograd is correct; fix manual implementation

**Exp 0.2 — Cross-modal FD gradient check:**
- Two modalities L=16, N=4, H=1, D_v=D_a=16
- Freeze one; FD the other; compare to autodiff
- **Success:** relative error < 1e-3 in both directions

**Exp 0.3 — Energy monotonicity:**
- L=32, N=8, 200 steps, eta in {0.1, 0.01, 0.001}
- **Success:** strict decrease at eta=0.001
- **Log:** LayerNorm null-space events (E_dot~0 AND grad_g E != 0)

### Tier 1 — Hopfield Unit Tests (CPU, <1 hr)

**Exp 1.1 — Unimodal completion:** K=5 prototypes; lambda_cross=0; cosine sim >0.95.

**Exp 1.2 — Cross-modal context degradation:** Sweep lambda_cross in {0,0.1,0.3,0.5,0.9} with noise context; sim >0.90 for lambda<=0.3.

**Exp 1.3 — Temporal window sensitivity:** Accuracy >0.9 when w >= 2*delta+1 for offset delta.

### Tier 2 — B-Spline Fidelity (CPU, <2 hr)

**Exp 2.1 — Compression error:** L=128 localized spike; error <5% at N=16.

**Exp 2.2 — B-spline vs Fourier ringing:** B-spline error within ±3 tokens of spike is <50% of Fourier error.

**Exp 2.3 — Quadrature convergence:** Error <1e-3 at M=32 for smooth trajectories (Gauss-Legendre reference).

### Tier 3 — EqProp Validation (small GPU, <4 hr)

**Exp 3.1 — EqProp vs BPTT:**
- L=8, N=4, K=3, D=16; 20 BPTT steps
- Sweep s in {0.1, 0.05, 0.01, 0.005, 0.001}
- **Success:** cosine sim >0.9 at s=0.01; O(s) error scaling; attractor agreement >95%

**Exp 3.2 — Memory scaling:**
- Profile BPTT vs EqProp for T_iter in {5, 10, 20, 50, 100}
- **Success:** BPTT linear in T; EqProp approximately constant

### Tier 4 — Synthetic Bidirectional (small GPU, <8 hr)

**Exp 4.1 — Linear map recovery:** N=1000 pairs, rank-4 map, J_gen MSE, alternating opt. Success: rel error <20%, monotone energy, FP residual <1e-3.

**Exp 4.2 — AV symmetry:** Symmetric error direction ±5%.

**Exp 4.3 — Energy ranking:** AUROC >0.90 matched vs mismatched.

### Tier 5 — Real Benchmarks (GPU cluster)

| Benchmark | Dataset | Metric | Target |
|---|---|---|---|
| 5.1 AVE Classification | AVE (4143 clips, 28 class) | Segment accuracy | vs ~74% Transformer |
| 5.2 Foley Generation | VGGSound-10 (~1K clips) | FAD, AV-Offset | vs Diff-Foley, MMAudio |
| 5.3 EqProp real data | AudioSet-10 (~5K clips) | Grad cosine sim | vs BPTT oracle |
| 5.4 Masked JEPA | AudioSet-10 | Masked mel MSE | vs audio-only JEPA |

---

## Performance & Efficiency

### Vectorization (no Python loops in hot path)

| Operation | Efficient form |
|---|---|
| Hopfield scores | `einsum('bld,kd->blk', g, Xi)` — single matmul |
| Log-partition | `logsumexp(scores + log_w, dim=-1)` — fused |
| Spline coefficients | `einsum('nl,bhld->bhnd', R, K)` |
| Temporal smoothing | `F.avg_pool1d` — GPU-native |

### `torch.compile`
```python
energy = torch.compile(energy)  # after Phase A debugging is complete
```
Expected ~2x speedup via kernel fusion on the inner quadrature loop.

### Mixed Precision
- `torch.autocast('cuda', torch.bfloat16)` for forward pass.
- State variables `x_v`, `x_a` stay as `float32` — bf16 gradient accumulation can break monotonicity.

### Memory Budget

| Component | Memory |
|---|---|
| State: x_v, x_a | `2 x B x L x D` float32 |
| Spline coefficients | `B x H x (N_v+N_a) x D_k` |
| Hopfield prototypes | `(K_v+K_a) x D` |
| EqProp: 2 equilibria | `2 x B x L x D` |
| BPTT trajectory | `T x B x L x D` |

For B=8, L=64, D=256, T=50: EqProp ~25 MB vs BPTT ~1.6 GB.

---

## Hyperparameter Defaults (from experiments.md)

| Parameter | Default | Notes |
|---|---|---|
| `N_m` | 16 | Spline basis size |
| `M_m` | 32 | Quadrature nodes (Gauss-Legendre) |
| `K_m` | 64 | Hopfield patterns per modality |
| `lambda_cross` | 0.05 | Init small; calibrate via Exp 1.2 |
| `window` | 3-5 | Calibrate to frame rate (Exp 1.3) |
| `beta` | 1.0 | Attention temperature |
| `beta_HN` | 1.0 | Hopfield temperature |
| `eta` | 1e-2 | Inference step size |
| `T_iter` | 50 | Inference iterations |
| `s` | 0.01 | EqProp nudge strength |
| `mu` | 0.99 | EMA teacher decay |
| `lambda_gen` | 1.0 | Dominant; always |
| `lambda_jepa` | 0.1 | Auxiliary |
| `lambda_rank` | 0.05 | Auxiliary |

---

## Diagnostic Logging Template (matches experiments.md)

```python
# Log at every epoch / every N solver steps:
{
  "E_free":                 # free-phase energy mean +/- std
  "E_nudged":               # nudged-phase energy
  "E_matched":              # matched-pair energy
  "E_mismatched":           # mismatched-pair energy
  "E_gap":                  # E_mismatched - E_matched (should be >0 after training)
  "fp_residual":            # ||grad_x E|| at convergence
  "attractor_agreement":    # fraction free/nudged -> same basin
  "energy_monotone_rate":   # fraction inference steps with E decreasing
  "grad_cosine_eqprop_bptt":# cosine sim when BPTT feasible
  "layernorm_null_events":  # steps where E_dot~0 but grad_g E != 0
}
```

---

## Common Pitfalls & Mitigations

| Pitfall | Symptom | Fix |
|---|---|---|
| Hand-coded key-path gradient | Exp 0.1 fails | Always use autograd through `R @ K_h` chain |
| Langevin used for MAP | Noisy / non-converging samples | Remove noise term; use deterministic solver |
| Free/nudged different attractors | EqProp grads wrong direction | Warm-start nudged from x_free*; reduce beta_HN |
| lambda_cross too large | Hopfield retrieval corrupted | Clamp [0, 0.3] initially; calibrate via Exp 1.2 |
| eta too large | Non-monotone energy (Exp 0.3) | Halve eta; log Hessian curvature estimate |
| Fourier basis for audio spikes | High error near spikes | Use B-splines; validate via Exp 2.2 |
| EMA teacher collapse (JEPA) | Student learns trivial solution | Stop-gradient on teacher; verify Exp 5.4 |
| Batch losses inside EqProp nudge | Theory violated | Only J_gen (samplewise) in nudge; rest are outer-loop |

---

## Open Theoretical Issues (Do Not Claim as Solved)

These are correctly labeled open in the paper. Implementation must not silently assume otherwise:

1. **LayerNorm stability** — Only Ė <= 0 proved. Absorbing invariance conjecture is unproved.
2. **EqProp in multi-attractor landscapes** — Valid only when free/nudged hit same attractor.
3. **Global energy landscape** — Joint attention + Hopfield: no global convergence theorem.
4. **Langevin mixing** — Exponentially slow in multi-attractor regime.

---

## Week-by-Week Execution

```
Week 1:  Phase 0 (env + scaffold) + Phase 1 (core primitives) + Tier 0 (grad gates)
Week 2:  Tier 1 (Hopfield) + Tier 2 (spline) + Phase 2 (solvers)
Week 3:  Tier 3 (EqProp) + Phase 3 (tokenizers) + Phase 4 (heads)
Week 4:  Tier 4 (synthetic bidirectional) + Phase 5A-B (BPTT training)
Week 5:  Phase 5C-D (EqProp training) + Phase 5E (Foley losses)
Week 6+: Tier 5 (real benchmarks: AVE, VGGSound, AudioSet)
```

> [!CAUTION]
> **Hard rule:** If any Tier 0 experiment fails, stop. Gradient errors propagate into every downstream result. Do not proceed to Tier 1 with a failing Tier 0.

---

## Test Suite (Written & Ready to Run)

Files are in `MET/tests/`. Each test is self-contained (inline stubs, no `met` package install needed yet) so they serve as the Phase 1 correctness contracts.

### `conftest.py` — Shared utilities
- `finite_difference_grad(fn, x, eps)` — central FD gradient, reusable across all test files
- `check_relative_error(grad_auto, grad_fd, threshold)` — assert helper with diagnostic message
- `TinyConfig` dataclass — consistent minimal sizes (L=16, N=4, M=8, D=16, D_k=8, H=1, K=4)
- Fixtures: `tiny_cfg`, `fd_grad`, `check_grad`

### `test_spline.py` — BSplineCache
| Test class | What it checks |
|---|---|
| `TestBSplineBasisMatrix` | Partition of unity (rows sum to 1), non-negativity, Gram symmetry, PD after ridge |
| `TestRidgeRegression` | Shape (N×L), roundtrip `F @ R @ F ~ F`, `R @ F ~ I_N` for small λ |
| `TestQuadrature` | Weights sum to 1, all positive, nodes in [0,1], integrates t² = 1/3 exactly |
| `TestEncodeContract` | Output shapes `(B,H,N,D_k)` + `(B,H,M,D_k)`, differentiable through K, linear in K |

### `test_attention.py` — ContinuousAttentionEnergy (**critical – Exp 0.1/0.2**)
| Test class | What it checks |
|---|---|
| `TestLogPartitionNumerics` | `logsumexp(s + log(w))` matches naive `log(Σ w exp(s))`, energy is scalar and finite |
| `TestGradientCorrectness` | **Exp 0.1**: intra-modal autodiff vs FD, rel error < 1e-3 |
| | **Exp 0.2**: cross-modal v→a and a→v directions, both < 1e-3 |
| `TestCrossModalSymmetry` | `E_cross == E_{v→a} + E_{a→v}`; single-step energy descent confirms |

### `test_hopfield.py` — HopfieldMemoryBank
| Test class | What it checks |
|---|---|
| `TestUnimodalPatternCompletion` | **Exp 1.1**: gradient points toward prototype; proto has lower E than noise |
| `TestCrossModalDegradation` | **Exp 1.2**: cosine sim thresholds at λ=0/0.1/0.3 with noise context |
| `TestTemporalSmoothing` | **Exp 1.3**: constant invariance, spike attenuation, shape preservation, window/offset coverage |
| `TestHopfieldGradients` | Grad w.r.t. g_m and Xi, λ_cross clamp safety, prototype normalization invariance |

### `test_energy.py` — Full METEnergy (integration)
| Test class | What it checks |
|---|---|
| `TestEnergyBasics` | Scalar output, finiteness, E changes with x_v and x_a |
| `TestFreezeSemantics` | freeze_v blocks state grad to x_v; still allows param grads (detach vs no_grad) |
| `TestEnergyMonotonicity` | **Exp 0.3**: monotonicity rate < 10% violations at small η; FP residual decreasing |
| `TestLayerNormNullSpace` | Null-space component detection (diagnostic, not a failure) |
| `TestEnergyRanking` | Energy gap computable; J_rank backward produces param grads |
| `TestEqPropMemoryBound` | EqProp uses 2 state tensors; BPTT uses T tensors (shape contract) |

### Run the tests
```bash
# From MET/
pip install torch scipy pytest
pytest tests/ -v

# Fast smoke run (skip gradient-heavy tests):
pytest tests/test_spline.py tests/test_hopfield.py -v

# Critical gate (must pass before any training):
pytest tests/test_attention.py -v -k "GradientCorrectness"
```
