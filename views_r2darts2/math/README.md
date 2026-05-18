# math

The **math** layer contains all custom `torch.nn.Module` loss functions and schedulers. Every class here operates on tensors in **asinh-transformed space** unless otherwise noted, and is designed for the zero-inflated, heavy-tailed structure of UCDP GED conflict fatality data (~90% zeros, ~10% events spanning four orders of magnitude).

---

## Files

| File | Class | Status | Base cell loss |
|------|-------|--------|----------------|
| `spotlight_loss_logcosh.py` | `SpotlightLossLogcosh` | ⭐ Production | log_cosh |
| `spotlight_loss.py` | `SpotlightLoss` | ⭐ Production | Barron(α=1.5) |
| `prism_loss.py` | `PrismLoss` | Research | MSE (= MSLE in log space) |
| `spotlight_focal_loss.py` | `SpotlightFocalLoss` | Research | log_cosh |
| `sentinel_loss.py` | `SentinelLoss` | Research | Generalised Charbonnier |
| `weighted_penalty_huber_loss.py` | `WeightedPenaltyHuberLoss` | Legacy | Huber |
| `weighted_huber_loss.py` | `WeightedHuberLoss` | Legacy | Huber |
| `time_aware_weighted_huber_loss.py` | `TimeAwareWeightedHuberLoss` | Legacy | Huber |
| `spike_focal_loss.py` | `SpikeFocalLoss` | Legacy | log_cosh |
| `tweedie_loss.py` | `TweedieLoss` | Legacy | Tweedie (p≈1.5) |
| `asymmetric_quantile_loss.py` | `AsymmetricQuantileLoss` | Legacy | Quantile |
| `zero_inflated_loss.py` | `ZeroInflatedLoss` | Legacy | Huber (two-part) |
| `shrinkage_loss.py` | `ShrinkageLoss` | Legacy | Shrinkage |
| `warmup_cawr.py` | `WarmupCAWR` | Production | — (scheduler) |

---

## Production losses: SpotlightLoss family

Both `SpotlightLoss` and `SpotlightLossLogcosh` share the same four-component architecture (v36/v37). They differ only in the **base cell loss** used for the shape component.

### Architecture

**Component 1 — DC/AC decomposition** (prevents RevIN bias amplification)

```
e_shape = e − mean(e)    per series
```

The centering matrix `J = I − 11ᵀ/T` has zero column sums, so the shape gradient sums to zero per series by construction. This structurally prevents the model from accumulating a DC bias in the shape loss. Without this, a small bias `b` in RevIN-normalized space becomes `b·σ` after denormalization, and `sinh(b·σ)` is exponentially larger than `sinh(E[b·σ])` (Jensen's inequality) — meaning even tiny shape biases produce massive overpredictions in raw death counts.

**Component 2 — Adaptive compound weighting** (parameter-free, replaces the `alpha` hyperparameter)

```
difficulty  = 1 − exp(−|e_shape|)           ∈ [0, 1)
importance  = 1 − exp(−max(|y|, |ŷ_sg|))   ∈ [0, 1)
w_compound  = 1 + difficulty × importance    ∈ [1, 2)
```

Both signals must be non-zero simultaneously for maximum weight. A cell the model already predicts well gets `difficulty→0 → w→1` regardless of its magnitude, preventing the loss from over-focusing on large-but-easy samples. The stop-gradient on `ŷ_sg` prevents second-order dynamics through the weight.

**Component 3 — KL-DRO tail aggregation** (proportional outlier detection, parameter-free)

```
log_l  = log(l + ε)
z      = (log_l − mean(log_l)) / std(log_l)
dro_w  = log1p(clamp(1+z, min=0))
α_soft = log_std / (log_std + 1.0)          # → 0 early in training (uniform)
dro_w  = α_soft × dro_w + (1−α_soft)
```

KL-DRO works in log-loss space, making it sensitive to *proportional* outliers rather than *absolute* ones. A village that misses by 10× the median loss gets the same emphasis as a Syria-scale miss at 10× the median. This is proportional-error-aligned, matching the MSLE evaluation metric. In contrast, χ²-DRO (z-scoring raw losses) causes Syria to permanently dominate gradients regardless of whether the model has already improved on it.

The soft activation `α_soft` blends toward uniform aggregation during early training when log-loss variance is small, providing a natural curriculum: start with more uniform optimization, shift to tail-focused as the model improves on easy samples.

**Component 4 — Level anchor** (T-scaled log_cosh on per-series mean error)

```
L_level = T · mean_per_series[ log_cosh(mean(ŷ) − mean(y)) ]
```

The *only* term that can shift per-series mean predictions. Necessary because the shape loss (DC/AC decomposed) is structurally blind to level. The `T` scaling factor compensates for the `1/T` chain-rule factor from the mean reduction.

**Component 5 — Spectral regularization** (optional, `δ > 0`)

Multi-resolution STFT magnitude comparison with the DC frequency bin masked. Phase-invariant; uses log_cosh on magnitude differences. Enforces temporal structure without penalizing phase shifts. Disabled by setting `delta=0`.

### Choosing between the two

| | `SpotlightLossLogcosh` | `SpotlightLoss` |
|---|---|---|
| Base cell loss | log_cosh(x) | Barron(α=1.5) |
| Gradient profile | `tanh(x)` — saturates at ±1 | `(2x² + 1)^(-0.25) · x` — heavier tail |
| Best for | General use; bounded gradient is safe for basis-expansion models (N-BEATS, N-HiTS) | When log_cosh is too aggressive in clipping large error gradients |
| Config | `"loss_function": "SpotlightLossLogcosh"` | `"loss_function": "SpotlightLoss"` |

### Configuration

Both losses require only two hyperparameters (both sweepable):

```python
"loss_function": "SpotlightLossLogcosh",  # or "SpotlightLoss"
"delta": 0.02,              # Spectral weight; 0 disables spectral term entirely
"non_zero_threshold": 0.88  # asinh(1) ≈ 0.88 → boundary of 1 battle death in asinh space
```

Typical `delta` ranges from W&B sweeps: `0.01–0.12` depending on model. Higher `delta` = more spectral pressure = stronger enforcement of temporal structure.

---

## Research losses

### PrismLoss

MSE base cell loss (equivalent to MSLE in asinh/log space). Uses the same compound weighting and KL-DRO as SpotlightLoss but **without the DC/AC decomposition or level anchor**. Use when RevIN is disabled or when direct MSLE optimization is required.

### SpotlightFocalLoss

Regression adaptation of the RetinaNet focal loss. Weights each cell by `(1 − exp(−|e|))^γ` — suppresses easy cells (small error), amplifies hard cells. No DRO, no DC/AC decomposition. Requires tuning `γ` (typically 1.0–3.0). Operates in `log1p` space (`LogTransform` target scaler).

### SentinelLoss

Generalised Charbonnier base loss (power `α`) with power-law magnitude weighting `(1 + |y|)^κ`, SiLU-based symmetric amplification, and a magnitude-weighted temporal gradient term. Provides independent control of which samples matter (`κ`) vs. directional error sensitivity (`β`). Useful for ablations comparing loss shape parametrizations.

---

## Legacy losses

These predate the Spotlight family and are retained for ablation studies and backward compatibility. Not recommended for new models.

| Loss | Core mechanism | Key hyperparameters |
|------|---------------|---------------------|
| `WeightedPenaltyHuberLoss` | Huber + FP/FN multiplicative penalties | `delta`, `non_zero_weight`, `false_positive_weight`, `false_negative_weight` |
| `WeightedHuberLoss` | Huber + non-zero class reweighting | `delta`, `non_zero_weight`, `zero_threshold` |
| `TimeAwareWeightedHuberLoss` | Huber + temporal decay on event weights | `delta`, `decay_factor`, `non_zero_weight` |
| `SpikeFocalLoss` | log_cosh focal by absolute magnitude | `gamma`, `spike_threshold` |
| `TweedieLoss` | Compound Poisson-Gamma | `p` (1.5 = recommended), `non_zero_weight` |
| `AsymmetricQuantileLoss` | Quantile regression, asymmetric τ | `tau` (0.75 = 3× underestimation penalty) |
| `ZeroInflatedLoss` | Two-part: BCE (zero / non-zero) + Huber (count) | `zero_weight`, `count_weight`, `delta` |
| `ShrinkageLoss` | Suppresses easy samples via sigmoid gate | `a` (shrinkage speed), `c` (threshold) |

---

## WarmupCAWR (scheduler)

`torch.optim.lr_scheduler._LRScheduler` subclass. CosineAnnealingWarmRestarts with a linear warmup phase.

Without warmup, CAWR starts at peak lr from epoch 0. For Transformers this causes up to 30 epochs of maximum-strength gradient updates while weights are still random, frequently leading to NaN or peace-attractor collapse. Warmup ramps lr linearly from `eta_min` to `base_lr` over `warmup_epochs`, then hands off to standard CAWR.

```python
"lr_scheduler_cls": "WarmupCAWR",
"lr_scheduler_warmup_epochs": 5,
"lr_scheduler_T_0": 30,
"lr_scheduler_T_mult": 2,
"lr_scheduler_eta_min": 1e-6,
```

---

## Loss function evolution timeline

| Generation | Loss | What it solved | What it revealed |
|---|---|---|---|
| Gen 1 | `WeightedPenaltyHuberLoss` | Basic class reweighting | Manually tuned FP/FN; no RevIN awareness |
| Gen 2 | `TweedieLoss`, `SpikeFocalLoss` | Distribution-appropriate base; focal weighting | No DRO; γ is a fragile hyperparameter |
| Gen 3 | `PrismLoss` | KL-DRO (parameter-free); compound weighting | No DC/AC; RevIN bias accumulates |
| Gen 4 | `SpotlightFocalLoss` | Focal regression without class-specific logic | Still requires γ; no DRO |
| Gen 5 | `SpotlightLossLogcosh` | DC/AC + compound + KL-DRO + level anchor | log_cosh saturates at large errors |
| Gen 5b | `SpotlightLoss` | Barron(α=1.5) for heavier-tail gradients | Same overall architecture |
