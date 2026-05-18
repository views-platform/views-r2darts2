<p align="center">
  <img src="https://github.com/user-attachments/assets/4cd8129b-9ad6-4fa3-a4ca-8288b0ab610f" alt="r2darts2 Banner" width="85%">
</p>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python Version" />
  &nbsp;&nbsp;
  <img src="https://img.shields.io/badge/darts-0.40.0-green.svg" alt="Darts Version" />
  &nbsp;&nbsp;
  <img src="https://img.shields.io/badge/pytorch-2.x-orange.svg" alt="PyTorch" />
  &nbsp;&nbsp;
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License" />
</p>

<p align="center">
  A time series forecasting package for conflict prediction within the VIEWS (Violence and Impacts Early-Warning System) ecosystem. Built on Darts and PyTorch Lightning, it provides deep learning models, domain-specific loss functions, and reproducibility infrastructure tuned for zero-inflated, heavy-tailed conflict fatality data.
</p>

---

## Key Features

- 🚀 **Production-Ready Integration**: Seamlessly integrates with the VIEWS pipeline ecosystem via the Genomic Firewall and DNA manifest validation
- ⚡ **Zero-Inflated Data Handling**: Specialized loss functions (SpotlightLoss family) and scalers (AsinhTransform chains) purpose-built for conflict fatality distributions
- 🧠 **10 Model Architectures**: TFT, N-BEATS, N-HiTS, TiDE, TCN, BlockRNN, Transformer, NLinear, DLinear, TSMixer
- 📊 **Static Covariate Fingerprints**: Per-entity conflict statistics (µ, σ, max, trend, sparsity)
- 🔗 **Chained Scalers**: Arrow-syntax pipelines (`AsinhTransform->MaxAbsScaler`) for multi-stage feature normalization
- 🛡️ **Fortress Architecture**: ADR-governed reproducibility, NaN detection, gradient health monitoring, and training stability callbacks throughout
- 🧮 **GPU Acceleration**: Optimized for single- and multi-GPU training via PyTorch Lightning

---

## 📦 Installation

```bash
git clone https://github.com/views-platform/views-r2darts2.git
cd views-r2darts2
pip install -e .
```

Requires `darts==0.40.0` and `views-pipeline-core>=2.0.0`. For GPU support, install the appropriate PyTorch version for your CUDA setup first. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

---

## 🧠 Supported Models

| Model | Static Covariates | Description | Ideal For |
|-------|:-----------------:|-------------|-----------|
| **TFT** (Temporal Fusion Transformer) | ✅ VSN+GRN gating | Hybrid LSTM + multi-head attention. Variable Selection Networks gate static covariate influence per time step. | Interpretable multivariate forecasting; best when entity identity strongly conditions the forecast. |
| **TSMixer** | ✅ Concatenation | Alternating time-mixing and feature-mixing MLP blocks. Static covariates are concatenated at each block (no gating). | Large-scale multivariate; fast training. Requires `AsinhTransform->MaxAbsScaler` on static cov stats due to blunt concat. |
| **TiDE** | ✅ Concatenation | MLP encoder-decoder for long-horizon forecasting. Efficient and scalable. | Resource-constrained environments, long input horizons. |
| **BlockRNN** | ✅ Concatenation | Stacked RNN/LSTM/GRU with optional static covariate injection. | Sequential dependency modeling; autoregressive forecasting. |
| **NLinear** | ✅ Concatenation | Lightweight linear model with optional trend/seasonality decomposition. | Baseline modeling; rapid prototyping. |
| **DLinear** | ✅ Concatenation | Decomposition-Linear — separate linear layers for trend and seasonal components. | Trend/seasonality separation; fast inference. |
| **N-HiTS** | ❌ Not consumed | Hierarchical interpolation with per-stack pooling→FC→theta pipelines. Flag accepted but **not passed to the Darts constructor** — static covariate fingerprints are computed and attached but silently ignored by the model. | Multi-scale temporal pattern extraction; long-term forecasting. |
| **Transformer** | ❌ Not consumed | Self-attention encoder with positional encoding. Flag accepted but **silently ignored** — no `use_static_covariates` is passed to the Darts constructor. | Long-range temporal dependency modeling. |
| **N-BEATS** | ❌ Not supported | Fully-connected stacks with basis expansions for trend/seasonality. Static covariates are architecturally incompatible. Do not configure `use_static_covariates=True`. | Interpretable decomposition; univariate/multivariate forecasting. |
| **TCN** (Temporal Convolutional Network) | ❌ Not consumed | Dilated causal convolutions with residual connections. | High-frequency series; long-range dependencies. |

> **Important**: Models marked ❌ will silently drop static covariate fingerprints even if `use_static_covariates=True` is set in the config. This is a Darts library constraint, not a bug in this package. For N-HiTS and Transformer, the fingerprints are computed (respecting `stat_time_range` for leakage prevention) but never seen by the model weights.

---

## 📊 Scalers

Proper data scaling is critical for neural network training. For conflict data — zero-inflated, heavily right-skewed, spanning four orders of magnitude — the choice of transform directly determines whether models converge.

### Available Scalers

| Scaler | Formula | Best For | Notes |
|--------|---------|----------|-------|
| **AsinhTransform** | $y = \operatorname{asinh}(x)$ | Zero-inflated counts, data with negatives | ⭐ **Recommended for all fatality data**. Handles zeros and extreme outliers. Near-linear below 1. |
| **MaxAbsScaler** | $y = x / \max(\|x\|)$ | Cross-entity normalization after element-wise transform | Maps to $[-1,1]$. Preserves ordinal rank. Required after AsinhTransform for static covariate injection in concatenation-based models. |
| **StandardScaler** | $y = (x - \mu)/\sigma$ | Roughly normal data | Zero-mean, unit-variance. Poor choice for zero-inflated distributions. |
| **MinMaxScaler** | $y = (x - x_{\min})/(x_{\max} - x_{\min})$ | Bounded data (0–1, 0–100) | Best for V-Dem indices and WDI percentages. |
| **SqrtTransform** | $y = \sqrt{x}$ | Moderate skew, count data | Gentler compression than asinh; undefined for negatives. |
| **LogTransform** | $y = \log(1 + x)$ | Strictly positive skewed data | Undefined for negatives. Prefer AsinhTransform for conflict data. |
| **RobustScaler** | Median + IQR | Outlier-heavy data | Resistant to extreme values. |
| **QuantileNormal** | Maps to $\mathcal{N}(0,1)$ | Any distribution | Forces Gaussian marginal. |
| **QuantileUniform** | Maps to $U(0,1)$ | Any distribution | Forces uniform marginal. |
| **YeoJohnsonTransform** | Power transform | Mixed positive/negative | Makes data more Gaussian-like. |

### AsinhTransform vs LogTransform

```
            Log(1+x)       AsinhTransform
x = -50     ❌ undefined    ✓ asinh(-50) = -4.61
x = 0       0.00            0.00
x = 1       0.69            0.88  (non-zero threshold in asinh space: 0.88 ≈ 1 death)
x = 100     4.62            5.30
x = 10000   9.21            9.90
```

The `non_zero_threshold` in SpotlightLoss is set to `0.88` because `asinh(1) ≈ 0.88` — this exactly corresponds to the boundary of "at least 1 battle death" in raw space.

### 🔗 Chained Scalers

Use the `->` operator to compose transforms sequentially. This is the **production standard** for all conflict and covariate features:

```python
# Target: suppress zeros + bound output
"target_scaler": "AsinhTransform"

# Features: element-wise compression then cross-entity normalization
"feature_scaler_map": {
    "AsinhTransform->MaxAbsScaler": [
        "lr_splag_1_ged_sb", "lr_ged_ns", "lr_ged_os",
        "lr_acled_sb", "lr_wdi_ny_gdp_mktp_kd",
        # ... all conflict and macro features
    ],
}

# Static covariate statistics (for models that consume them)
"static_covariate_stats": {"transform": "AsinhTransform->MaxAbsScaler"}
```

**Why MaxAbsScaler after AsinhTransform for features?**  
AsinhTransform is element-wise: it compresses Syria's `ged_sb ≈ 5000` and Chad's `ged_sb ≈ 3` independently. After asinh, Syria is at ~8.5 and Chad at ~1.8 — a 4.7× gap persists. MaxAbsScaler maps the entire feature column to `[-1, 1]` across all 180+ countries, collapsing cross-entity scale while preserving ordinal rank. Without this, concatenation-based models (TSMixer, TiDE) inject raw magnitude bias at every block.

**Forward / inverse chain direction:**
```
Forward:  X → Scaler₁.fit_transform(X) → Scaler₂.fit_transform(X') → X_scaled
Inverse:  X_scaled → Scaler₂.inverse_transform → Scaler₁.inverse_transform → X_original
```

### Static Covariate Fingerprints

For models that consume static covariates (TFT, TSMixer, TiDE, BlockRNN, NLinear, DLinear), five per-entity statistics are computed from the **training partition only** (via `stat_time_range`) and injected as `TimeSeries.static_covariates` metadata:

| Statistic | Meaning | Transform Recommendation |
|-----------|---------|--------------------------|
| `target_mu` | Mean conflict level | `AsinhTransform->MaxAbsScaler` |
| `target_sigma` | Volatility / spread | `AsinhTransform->MaxAbsScaler` |
| `target_max` | Peak value (spike extremity) | `AsinhTransform->MaxAbsScaler` |
| `target_trend` | OLS slope over training window | `AsinhTransform->MaxAbsScaler` |
| `target_sparsity` | Fraction of zero months | No transform (already in [0,1]) |

Always pass `stat_time_range` to `as_darts_timeseries()` to prevent test-period leakage:
```python
ts_list = dataset.as_darts_timeseries(
    stat_time_range=(training_start_month_id, training_end_month_id),
    static_cov_transform="AsinhTransform->MaxAbsScaler",
)
```

### Recommended Scaler by Data Source

| Data Source | Feature Type | Recommended Chain | Rationale |
|-------------|--------------|-------------------|-----------|
| **UCDP / ACLED** | Fatality counts | `AsinhTransform->MaxAbsScaler` | Zero-inflated; cross-entity normalization required |
| **WDI** | GDP, population, aid flows | `AsinhTransform->MaxAbsScaler` | Spans many orders of magnitude; can be negative (net migration) |
| **WDI** | Percentages (`_zs` suffix) | `AsinhTransform->MaxAbsScaler` | Handles near-zero values; consistent with other features |
| **V-Dem** | Democracy indices (0–1 bounded) | `AsinhTransform->MaxAbsScaler` or `MinMaxScaler` | Already bounded; either works |
| **Static cov stats** | µ, σ, max, trend | `AsinhTransform->MaxAbsScaler` | Required for concatenation-based models |
| **Static cov stats** | Sparsity | None (raw) | Already in [0,1] |

---

## ⚡ Loss Functions

All loss functions target **zero-inflated conflict data**: ~90% zeros, ~10% events spanning four orders of magnitude. The loss function family has evolved substantially — the table below shows the current production-recommended functions and the full catalog.

### Loss Function Catalog

| Loss Function | Status | Base Cell Loss | Key Mechanism | Use When |
|---------------|--------|---------------|---------------|----------|
| **SpotlightLossLogcosh** | ⭐ **Production** | log_cosh | DC/AC decomp + compound weights + KL-DRO + level anchor + spectral | Default for all models in production |
| **SpotlightLoss** | ⭐ **Production** | Barron(α=1.5) | Same as above with more robust base cell loss | When log_cosh gradient is too aggressive on large errors |
| **PrismLoss** | Research | MSE (= MSLE in log space) | KL-DRO + compound weights, no DC/AC decomp, no level anchor | MSLE-aligned optimization without RevIN |
| **SpotlightFocalLoss** | Research | log_cosh | Focal weighting by difficulty `(1−exp(−\|e\|))^γ`, no DRO | Models without RevIN; exploration |
| **SentinelLoss** | Research | Generalised Charbonnier | Power-law magnitude weights + SiLU symmetry + temporal gradient | Alternative robust base when Barron α needs tuning |
| **WeightedPenaltyHuberLoss** | Legacy | Huber | FP/FN multiplicative penalties | Simple baselines; not recommended for production |
| **WeightedHuberLoss** | Legacy | Huber | Non-zero reweighting | Simple baselines |
| **TimeAwareWeightedHuberLoss** | Legacy | Huber | Temporal decay + event weights | Time-sensitive ablations |
| **TweedieLoss** | Legacy | Tweedie (p≈1.5) | Compound Poisson-Gamma | Count data without asinh transform |
| **AsymmetricQuantileLoss** | Legacy | Quantile | Asymmetric τ-penalty | When underestimation cost >> overestimation |
| **ZeroInflatedLoss** | Legacy | Huber (two-part) | Explicit binary + count split | Explicit zero-inflation modeling |
| **SpikeFocalLoss** | Legacy | log_cosh | Focal on absolute magnitude | Predates KL-DRO; superseded by Spotlight family |
| **ShrinkageLoss** | Legacy | Shrinkage | Suppresses easy samples via sigmoid gate | Exploratory; not validated for conflict |

### SpotlightLossLogcosh — Architecture Deep Dive

The production loss for all current VIEWS models. Operates entirely in **asinh space**; the target scaler must be `AsinhTransform`.

**Five orthogonal components:**

**1. DC/AC decomposition** — prevents RevIN from amplifying bias:
```
e_shape = e − mean(e)    per series
```
The shape gradient sums to zero per series by construction (`J = I − 11ᵀ/T`). A small bias `b` in normalized space becomes `b·σ` after RevIN denormalization, and `sinh(b·σ) > sinh(E[b·σ])` via Jensen's inequality — exponential overprediction in raw death counts. The DC/AC split structurally blocks this. The level anchor (component 4) is the *only* mechanism that can shift per-series means.

**2. Adaptive compound weighting** — parameter-free event focus:
```
difficulty  = 1 − exp(−|e_shape|)           ∈ [0, 1)
importance  = 1 − exp(−max(|y|, |ŷ_sg|))   ∈ [0, 1)
w_compound  = 1 + difficulty × importance    ∈ [1, 2)
```
Both signals must be active simultaneously. Perfect predictions get `difficulty→0 → w→1` regardless of magnitude. Replaces the `alpha` hyperparameter from earlier versions.

**3. KL-DRO tail aggregation** — proportional outlier detection:
```
log_l  = log(l + ε)
z      = (log_l − mean(log_l)) / std(log_l)
dro_w  = log1p(clamp(1+z, min=0))
dro_w  = dro_w / mean(dro_w)
α_soft = log_std / (log_std + 1.0)          # soft activation (uniform early in training)
```
Unlike χ²-DRO (which detects *absolute* loss outliers and causes Syria to dominate), KL-DRO detects *proportional* outliers: a village miss at 10× the median receives the same weight as a Syria miss at 10× the median. Aligned with the proportional error sensitivity of asinh-space MSE.

**4. Level anchor** — T-scaled log_cosh on per-series mean error:
```
L_level = T · mean_per_series[ log_cosh(mean(ŷ) − mean(y)) ]
```
The *only* mechanism that can shift series-level means. T-scaling compensates for the `1/T` chain-rule factor from mean reduction.

**5. Spectral regularization** (optional, `δ > 0`):
Multi-resolution STFT magnitude comparison with the DC bin masked. Enforces temporal structure without caring about phase. Proportional contribution tuned via `delta` (current production values: 0.015–0.12 depending on model).

**Configuration:**
```python
"loss_function": "SpotlightLossLogcosh",
"delta": 0.02,               # Spectral weight; 0 disables spectral term
"non_zero_threshold": 0.88,  # asinh(1) ≈ 0.88 (= 1 battle death in raw space)
```

### Loss Function Evolution

The loss function development followed a clear progression as each failure mode was identified and addressed:

| Generation | Loss | Problem It Solved | Limitation Discovered |
|---|---|---|---|
| Gen 1 | `WeightedPenaltyHuberLoss` | Basic non-zero reweighting | Huber is symmetric; no handling of RevIN bias; manually tuned FP/FN parameters |
| Gen 2 | `TweedieLoss`, `SpikeFocalLoss` | Compound Poisson structure; focal weighting | No DRO; focal exponent γ is a fragile hyperparameter |
| Gen 3 | `PrismLoss` | KL-DRO replaces χ²-DRO; compound weighting is parameter-free | No DC/AC decomposition; RevIN bias accumulates; no level anchor |
| Gen 4 | `SpotlightFocalLoss` | Focal mechanism adapted for regression, no class-specific logic | No DRO; still requires γ tuning |
| Gen 5 | `SpotlightLossLogcosh` | DC/AC decomp + compound + KL-DRO + level anchor + spectral; fully parameter-free weighting | log_cosh gradient can clip large errors aggressively |
| Gen 5b | `SpotlightLoss` | Barron(α=1.5) base cell loss — heavier tail than log_cosh | Same architecture as v36 Logcosh |

---

## 🏗️ Multi-Stack Model Configuration

N-HiTS and TSMixer are multi-stack architectures where residuals are passed between stacks. **Layer width ordering is critical and non-obvious.**

### The Layer Widths Trap

Both N-HiTS and TSMixer use a **residual stacking pipeline**:

```
Stack 0 (coarse) → absorbs easy patterns (trend, long cycles)
        ↓ residual
Stack 1 (mid)    → absorbs medium-frequency patterns
        ↓ residual  
Stack 2 (fine)   → must absorb ALL remaining residuals, including spike patterns
```

**The fine stack always has the hardest job.** Under SpotlightLoss (DRO weighting), high-conflict country residuals (e.g., Sudan) dominate gradients. If the fine stack has minimal capacity, it cannot model these spikes, producing erratic theta coefficients that cause:

- **Explosion on high-conflict countries** (Sudan, Syria): fine stack theta blows up
- **Flatline on peaceful countries**: coarse stack weights drift toward dominant loss signal, collapsing peaceful predictions to near-zero

**Correct configuration:**
```python
# WRONG — coarse gets most capacity, fine gets least
"layer_widths": [256, 128, 64]   # ← Sudan explosion + flatline

# CORRECT — fine stack gets most capacity for residual absorption
"layer_widths": [64, 128, 256]   # ← stable, fine stack can handle spikes
```

### N-HiTS Pooling / Frequency Alignment

N-HiTS pools the input sequence before each stack's FC block. The `pooling_kernel_sizes` and `n_freq_downsample` must be **aligned**:

```
pool_k = 4  →  input compressed to ceil(36/4) = 9 time steps → FC input dim = 9
n_freq = 4  →  theta output has 4 frequency coefficients, interpolated to 36 steps
```

If `pool_k=4` but `n_freq=3`, there are 9 FC inputs but only 3 theta points — implicit upsampling by 9→36 via 3 basis functions creates a 3:1 gap. This forces the interpolation to guess intermediate values, introducing artificial smoothing that conflicts with spike reconstruction.

**Correct alignment:**
```python
"pooling_kernel_sizes": [[4, 2, 1]],     # coarse: 9 steps, mid: 18 steps, fine: 36 steps
"n_freq_downsample": [[4, 2, 1]],        # 4 theta / 2 theta / 1 theta, interpolated to 36
```

The fine stack (`pool_k=1, n_freq=1`) sees all 36 time steps and produces 36 theta coefficients — effectively identity interpolation. This is correct: the fine stack should not impose any temporal compression on spike signals.

Also use `max_pool_1d=True` in the coarse stack to preserve spike maxima during pooling (average pooling dilutes spike information that should be routed to the coarse trend stack, not the fine detail stack).

---

## ⚡ Loss Functions

*(See complete catalog above.)*

---

## 🛡️ Fortress Architecture & Governance

This repository adheres to the **Fortress Architecture**: strict engineering and mathematical standards designed to guarantee scientific integrity and reproducibility in conflict forecasting.

The repository is governed by:
- **[Architectural Decision Records (ADRs)](docs/ADRs/README.md)**: Sequential, authoritative records of every major design choice.
- **[Class Intent Contracts (CICs)](docs/CICs/README.md)**: Explicit declarations of purpose and responsibility for every critical class.
- **[Reproducibility Manifest](docs/standards/REPRODUCIBILITY_MANIFEST.md)**: The mandatory DNA genome that every experiment must declare before execution.

### Training Stability Callbacks

Every training run is monitored by mandatory Fortress callbacks configured in `ModelCatalog`:

| Callback | Purpose |
|----------|---------|
| `NaNDetectionCallback` | Halts training immediately on NaN in loss or weights |
| `GradientHealthCallback` | Monitors gradient norm; warns on explosion/vanishing |
| `WeightNormCallback` | Tracks parameter norm evolution across epochs |
| `LossStabilityCallback` | Detects loss spikes and plateau regimes |
| `RevINMonitorCallback` | Monitors RevIN affine parameters for drift |
| `PredictionSanityCallback` | Validates output shape and value range each epoch |
| `YHatBarCallback` | Tracks per-series mean predictions (ŷ bar) against targets |
| `EpochTimingCallback` | Logs wall-clock time per epoch for performance tracking |

---

## 🔧 API Reference

Every core class follows the **1-Class-1-File** standard.

### ScalerSelector
```python
from views_r2darts2.transformers.scaler_selector import ScalerSelector

scaler = ScalerSelector.get_scaler("AsinhTransform")
pipeline = ScalerSelector.get_chained_scaler("AsinhTransform->MaxAbsScaler")
```

### FeatureScalerManager
```python
from views_r2darts2.transformers.feature_scaler_manager import FeatureScalerManager

manager = FeatureScalerManager(
    feature_scaler_map={"AsinhTransform->MaxAbsScaler": ["lr_ged_sb", "lr_ged_ns"]},
    default_scaler=None,
)
```

### Triple Catalogs (Genomic Firewall)
```python
from views_r2darts2.catalogs.model_catalog import ModelCatalog
from views_r2darts2.catalogs.loss_catalog import LossCatalog
from views_r2darts2.catalogs.optimizer_catalog import OptimizerCatalog
from views_r2darts2.catalogs.scheduler_catalog import SchedulerCatalog

# Catalogs validate the DNA manifest on initialization
loss_fn = LossCatalog(config).get_loss()
model   = ModelCatalog(config).get_model("NHiTSModel")
```

### _ViewsDatasetDarts
```python
from views_r2darts2.transformers.views_dataset_darts import _ViewsDatasetDarts

dataset = _ViewsDatasetDarts.from_views_path(path_raw, run_type, config)

# Always pass stat_time_range to prevent leakage into static covariate stats
ts_list = dataset.as_darts_timeseries(
    stat_time_range=(training_start_id, training_end_id),
    static_cov_transform="AsinhTransform->MaxAbsScaler",
)
```

### ReproducibilityGate
```python
from views_r2darts2.infrastructure.reproducibility_gate import ReproducibilityGate

ReproducibilityGate.Config.audit_manifest(config)
ReproducibilityGate.Data.audit_dataframe_schema(df, expected_targets, expected_features)
ReproducibilityGate.Temporal.audit_continuity(partition)
```

---

## 📐 Production Configuration Template

Minimal validated configuration for a new model:

```python
def get_hp_config():
    return {
        # Forecast horizon
        "steps": [*range(1, 37)],
        "input_chunk_length": 36,
        "output_chunk_length": 36,
        "output_chunk_shift": 0,

        # Scaling — production standard
        "target_scaler": "AsinhTransform",
        "feature_scaler": None,
        "feature_scaler_map": {
            "AsinhTransform->MaxAbsScaler": [
                # All conflict counts, macro indicators, and lagged features
            ],
        },
        "static_covariate_stats": {"transform": "AsinhTransform->MaxAbsScaler"},

        # Loss — production standard
        "loss_function": "SpotlightLossLogcosh",
        "delta": 0.02,              # Spectral weight; tune via W&B sweep
        "non_zero_threshold": 0.88, # asinh(1): boundary of 1 battle death

        # Optimizer
        "optimizer_cls": "AdamW",
        "lr": 0.0005,
        "weight_decay": 0.0002,
        "gradient_clip_val": 3,
        "optimizer_kwargs": {"lr": 0.0005, "weight_decay": 0.0002},

        # Scheduler
        "lr_scheduler_cls": "ReduceLROnPlateau",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 12,
        "lr_scheduler_min_lr": 1e-6,
        "lr_scheduler_kwargs": {
            "mode": "min", "factor": 0.5, "patience": 12,
            "min_lr": 1e-6, "cooldown": 3,
            "threshold": 0.01, "threshold_mode": "rel",
        },

        # Training
        "batch_size": 128,
        "n_epochs": 300,
        "early_stopping_patience": 35,
        "early_stopping_min_delta": 0.001,
        "force_reset": True,

        # Normalization
        "use_reversible_instance_norm": True,
        "use_cyclic_encoders": True,
        "use_static_covariates": True,  # Set False for N-BEATS, N-HiTS, Transformer

        # Reproducibility
        "random_state": 67,
        "time_steps": 36,
        "rolling_origin_stride": 1,
        "prediction_format": "dataframe",

        # Prediction
        "likelihood": None,
        "num_samples": 1,
        "mc_dropout": False,
        "n_jobs": -1,
    }
```

---

## 📚 References

- **Darts**: [unit8co/darts](https://github.com/unit8co/darts) — Time series forecasting library
- **VIEWS**: [viewsforecasting.org](https://viewsforecasting.org/) — Violence Early-Warning System