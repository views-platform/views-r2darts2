<p align="center">
  <img src="https://github.com/user-attachments/assets/4cd8129b-9ad6-4fa3-a4ca-8288b0ab610f" alt="r2darts2 Banner" width="85%">
</p>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="GitHub License" />
  &nbsp;&nbsp;
   <img src="https://img.shields.io/badge/darts-0.35.0%2B-green.svg" alt="GitHub License" />
  &nbsp;&nbsp;
</p>

<p align="center">
  A time series forecasting package designed for conflict prediction within the VIEWS (Violence and Impacts Early-Warning System) ecosystem. It employs state-of-the-art deep learning models from the Darts library to produce forecasts and conflict-related metrics
</p>


---

## Key Features

- 🚀 **Production-Ready Integration**: Seamlessly integrates with the VIEWS pipeline ecosystem
-  **Zero-Inflated Data Handling**: Specialized scalers and loss functions for conflict data
- 🧠 **Multiple Model Architectures**: Supports 8+ cutting-edge forecasting models
- 📈 **Probabilistic Forecasting**: Quantifies uncertainty through multiple samples
- ⚙️ **Hyperparameter Management**: Centralized configuration for all models
- 🧮 **GPU Acceleration**: Optimized for GPU training and inference

---

## 📦 Installation

```bash
git clone https://github.com/views-platform/views-r2darts2.git
cd views-r2darts2
pip install -e .
```
For GPU support, install the correct PyTorch version for your CUDA setup. See [PyTorch installation guide](https://pytorch.org/get-started/locally/).

---

## 🧠 Supported Models

| Model | Description | Key Strengths | Ideal For |
|-------|-------------|---------------|-----------|
| **TFT** (Temporal Fusion Transformer) | Hybrid architecture combining LSTM encoders, variable selection networks, and multi-head attention. Integrates static, known, and observed covariates for interpretable, context-aware forecasting. | Captures complex temporal and feature relationships, interpretable feature importances, handles missing data and static covariates. | Multivariate forecasting, interpretability, heterogeneous data sources. |
| **N-BEATS** | Deep stack of fully-connected blocks with backward and forward residual links. Each block models trend and seasonality using interpretable basis expansions. | Highly interpretable, flexible for both univariate and multivariate series, excels at trend/seasonality extraction. | Univariate/multivariate forecasting, interpretable decomposition, long-term prediction. |
| **TiDE** | MLP-based architecture that models temporal dependencies using time and feature mixing blocks. Avoids recurrence and attention for efficient training and inference. | Fast training, scalable to large datasets, effective for long input sequences. | Large-scale forecasting, resource-constrained environments, long input horizons. |
| **TCN** (Temporal Convolutional Network) | Deep convolutional network with dilated causal convolutions and residual connections. Captures long-range dependencies without recurrence. | Highly parallelizable, robust to vanishing gradients, effective for high-frequency and long-range temporal patterns. | High-frequency time series, long-range dependencies, parallel training. |
| **Block RNN** | Sequential model using stacked RNN, LSTM, or GRU layers. Supports autoregressive and direct multi-step forecasting. | Good for capturing sequential and temporal dependencies, flexible for various RNN types. | Sequential dependencies, classic time series modeling, autoregressive tasks. |
| **Transformer** | Self-attention-based model with positional encoding, multi-head attention, and feed-forward layers. Models long-term dependencies and complex temporal relationships. | Scalable to large datasets, handles multivariate and long-horizon forecasting, interpretable attention weights. | Multivariate, long-term forecasting, complex temporal patterns. |
| **NLinear** | Simple linear model optionally decomposing input into trend and seasonality components. Uses linear layers for fast, interpretable predictions. | Extremely fast inference, interpretable, suitable for baseline and trend/seasonality separation. | Baseline modeling, trend/seasonality separation, rapid prototyping. |
| **DLinear** | Decomposition-Linear model that splits input into trend and seasonal components, each modeled by separate linear layers. | Efficient for large datasets, interpretable decomposition, improved performance over simple linear models. | Trend/seasonality separation, fast inference, large-scale forecasting. |
| **TSMixer** | MLP-based model that alternates mixing along time and feature dimensions using dedicated mixing blocks. Captures temporal and cross-feature interactions without convolutions or attention. | Lightweight, highly parallelizable, effective for large-scale and multivariate time series, fast training. | Large-scale time series, fast training, scalable architectures, cross-feature interaction modeling. |

---

## 📊 Scalers

Proper data scaling is critical for neural network training. For conflict data with its unique characteristics (zero-inflation, heavy tails, extreme outliers), choosing the right scaler per feature type significantly impacts model performance.

### Available Scalers

| Scaler | Formula | Best For | Notes |
|--------|---------|----------|-------|
| **AsinhTransform** | $y = \text{asinh}(x)$ | Zero-inflated counts, data with negatives | ⭐ **Recommended for fatality data**. Handles zeros gracefully, compresses outliers like log but works with negatives |
| **SqrtTransform** | $y = \sqrt{x}$ | Moderate skew, count data | Gentler than log, defined at zero |
| **LogTransform** | $y = \log(1 + x)$ | Strictly positive skewed data | Classic choice but undefined for negatives |
| **RobustScaler** | Uses median/IQR | Outlier-heavy data | Resistant to extreme values |
| **StandardScaler** | $y = \frac{x - \mu}{\sigma}$ | Normally distributed data | Zero mean, unit variance |
| **MinMaxScaler** | $y = \frac{x - x_{min}}{x_{max} - x_{min}}$ | Bounded data (0-1, 0-100) | Best for percentages and indices |
| **QuantileNormal** | Maps to $\mathcal{N}(0,1)$ | Any distribution | Forces Gaussian output |
| **QuantileUniform** | Maps to $U(0,1)$ | Any distribution | Forces uniform output |
| **YeoJohnsonTransform** | Power transform | Mixed positive/negative | Makes data more Gaussian-like |

### AsinhTransform vs LogTransform

```
            Log Transform              Asinh Transform
x = -50     ❌ undefined               ✓ asinh(-50) = -4.61
x = 0       log(1+0) = 0               asinh(0) = 0  
x = 1       log(2) = 0.69              asinh(1) = 0.88
x = 100     log(101) = 4.62            asinh(100) = 5.30
x = 10000   log(10001) = 9.21          asinh(10000) = 9.90
```

**Key advantages:**
- ✅ Handles zeros perfectly: `asinh(0) = 0`
- ✅ Works with negative values (net migration, differences)
- ✅ Near-linear for small values (preserves distinction between 1-3 fatalities)
- ✅ Compresses large outliers (mass atrocity events)
- ✅ Invertible everywhere

### Feature-Specific Scaling with FeatureScalerManager

Apply different scalers to different feature groups based on their data characteristics:

```python
"feature_scaler_map": {
    # Zero-inflated conflict counts - asinh handles zeros and extreme spikes
    "AsinhTransform": [
        "ged_sb", "ged_ns", "ged_os", "acled_sb", "acled_os",
        "ged_sb_tsum_24", "splag_1_ged_sb",
        # Large-scale economic data with extreme skew
        "wdi_ny_gdp_mktp_kd", "wdi_sm_pop_netm", "wdi_sm_pop_refg_or"
    ],
    # Bounded percentages and rates (0-100 scale) and V-Dem indices (0-1)
    "MinMaxScaler": [
        "wdi_sl_tlf_totl_fe_zs", "wdi_sp_urb_totl_in_zs",
        "vdem_v2x_polyarchy", "vdem_v2x_libdem"
    ],
    # Growth rates (can be negative, roughly normal)
    "StandardScaler": ["wdi_sp_pop_grow"],
    # Mortality rates (positive, moderate skew)
    "SqrtTransform": ["wdi_sp_dyn_imrt_fe_in"]
}
```

### 🔗 Chained Scalers

For complex data distributions, a single scaler may not be enough. **Chained scalers** allow you to apply multiple transformations sequentially, combining the benefits of each.

#### Why Chain Scalers?

Consider fatality data:
1. **Zero-inflated**: Most observations are 0
2. **Heavy-tailed**: Rare mass atrocity events (1000+ deaths)
3. **Need bounded output**: Neural networks train better with bounded inputs

A single `AsinhTransform` handles zeros and compresses outliers, but the output range varies. Adding `MinMaxScaler` afterwards bounds the output to [0, 1]:

```
Raw data:     [0, 0, 1, 5, 10, 1000, 0, 3]
                    ↓ AsinhTransform
Asinh output: [0, 0, 0.88, 2.31, 2.99, 7.60, 0, 1.82]
                    ↓ MinMaxScaler  
Final output: [0, 0, 0.12, 0.30, 0.39, 1.0, 0, 0.24]
```

#### Chain Syntax

Use the `->` operator to chain scalers. Transforms are applied **left to right**:

```python
# Single chain for target
"target_scaler": "AsinhTransform->StandardScaler"

# Feature-specific chains in feature_scaler_map
"feature_scaler_map": {
    "AsinhTransform->MinMaxScaler": ["ged_sb", "ged_ns", "acled_sb"],
    "LogTransform->StandardScaler": ["wdi_ny_gdp_mktp_kd"],
    "SqrtTransform->MinMaxScaler": ["wdi_sp_dyn_imrt_fe_in"],
}
```

#### How It Works

**Forward transform** (during training):
```
X → Scaler₁.fit_transform(X) → Scaler₂.fit_transform(X') → ... → X_scaled
```

**Inverse transform** (during prediction):
```
X_scaled → Scaler_n.inverse_transform(X) → ... → Scaler₁.inverse_transform(X') → X_original
```

The inverse is applied in **reverse order** to correctly recover the original data.

#### Mathematical Verification

For `AsinhTransform->StandardScaler`:

```python
# Forward: X → asinh(X) → standardize(asinh(X))
X_asinh = np.arcsinh(X)                    # Step 1: Asinh
X_scaled = (X_asinh - μ) / σ               # Step 2: Standardize

# Inverse: X_scaled → unstandardize → sinh
X_asinh_recovered = X_scaled * σ + μ       # Step 1: Unstandardize
X_recovered = np.sinh(X_asinh_recovered)   # Step 2: Sinh (inverse of asinh)

# X_recovered == X (within floating-point precision)
```

#### Complete Configuration Example

```python
# config_hyperparameters.py
def get_hp_config():
    return {
        "steps": [*range(1, 37)],
        "input_chunk_length": 36,
        "output_chunk_shift": 0,
        
        # Target scaling: asinh to handle zeros, then standardize
        "target_scaler": "AsinhTransform->StandardScaler",
        
        # No global feature scaler - using feature_scaler_map instead
        "feature_scaler": None,
        
        # Feature-specific chained scalers
        "feature_scaler_map": {
            # Zero-inflated counts: asinh + bounded output
            "AsinhTransform->MinMaxScaler": [
                "ged_sb", "ged_ns", "ged_os",
                "acled_sb", "acled_os",
                "splag_1_ged_sb", "splag_1_ged_ns",
            ],
            # Large economic values: log + standardize
            "LogTransform->StandardScaler": [
                "wdi_ny_gdp_mktp_kd", 
                "wdi_pop_totl",
            ],
            # Distance features: sqrt + bounded
            "SqrtTransform->MinMaxScaler": [
                "dist_to_capital", 
                "dist_to_border",
            ],
            # Already bounded (0-1): just MinMax for consistency
            "MinMaxScaler": [
                "vdem_v2x_polyarchy", 
                "vdem_v2x_libdem",
            ],
        },
        
        # Model architecture...
        "hidden_size": 128,
        "num_encoder_layers": 2,
        # ...
    }
```

#### Alternative Configuration Formats

The `feature_scaler_map` also supports list and dict formats for chains:

```python
# Format 1: Arrow syntax (recommended)
"feature_scaler_map": {
    "AsinhTransform->MinMaxScaler": ["ged_sb", "ged_ns"],
}

# Format 2: List syntax
"feature_scaler_map": {
    "conflict_features": {
        "scaler": ["AsinhTransform", "MinMaxScaler"],
        "features": ["ged_sb", "ged_ns"]
    }
}

# Format 3: Dict with chain key
"feature_scaler_map": {
    "conflict_features": {
        "scaler": {"chain": ["AsinhTransform", "MinMaxScaler"]},
        "features": ["ged_sb", "ged_ns"]
    }
}
```

#### Triple Chains

For extreme cases, you can chain three or more scalers:

```python
# Apply asinh, then robust scaling, then bound to [0,1]
"AsinhTransform->RobustScaler->MinMaxScaler": ["ged_sb"]
```

#### Verifying Correctness

You can verify that chained scaling is invertible:

```python
from views_r2darts2.utils.scaling import ScalerSelector
import numpy as np

# Create test data with zeros and outliers
X = np.array([[0], [1], [10], [100], [1000], [10000]])

# Create chained scaler
chained = ScalerSelector.get_chained_scaler("AsinhTransform->StandardScaler")

# Forward transform
X_scaled = chained.fit_transform(X)
print(f"Scaled mean: {X_scaled.mean():.10f}")  # Should be ~0
print(f"Scaled std:  {X_scaled.std():.10f}")   # Should be ~1

# Inverse transform
X_recovered = chained.inverse_transform(X_scaled)
max_error = np.max(np.abs(X - X_recovered))
print(f"Max recovery error: {max_error:.2e}")  # Should be ~1e-14
```

### Recommended Scaler by Data Source

| Data Source | Feature Type | Recommended Scaler | Rationale |
|-------------|--------------|-------------------|-----------|
| **UCDP/ACLED** | Fatality counts | AsinhTransform->MinMaxScaler | Zero-inflated with extreme outliers |
| **WDI** | GDP, population | LogTransform->StandardScaler | Spans millions to trillions |
| **WDI** | Percentages (`_zs` suffix) | MinMaxScaler | Already bounded 0-100 |
| **WDI** | Growth rates | StandardScaler | Can be negative, roughly normal |
| **V-Dem** | Democracy indices | MinMaxScaler | Already bounded 0-1 |
| **Topic models** | Theta proportions | MinMaxScaler | Probabilities bounded 0-1 |

---

## ⚡ Loss Functions

All loss functions are designed for **zero-inflated conflict data** where the majority of observations are zeros, and correctly predicting conflict onset is critical.

### Available Loss Functions

| Loss Function | Key Features | Best For |
|---------------|--------------|----------|
| **WeightedPenaltyHuberLoss** | Huber + FP/FN penalties | ⭐ General purpose, imbalanced data |
| **WeightedHuberLoss** | Huber + non-zero weighting | Simple weighted loss |
| **TimeAwareWeightedHuberLoss** | Temporal decay + event weights | Time-sensitive forecasting |
| **SpikeFocalLoss** | Focal mechanism for spikes | Rare extreme events |
| **TweedieLoss** | Compound Poisson-Gamma | Zero-inflated continuous |
| **AsymmetricQuantileLoss** | Quantile regression | Asymmetric error costs |
| **ZeroInflatedLoss** | Two-part model | Explicit zero modeling |

### WeightedPenaltyHuberLoss (Recommended)

The default choice for conflict forecasting. Combines Huber loss with multiplicative penalties:

```python
"loss_function": "WeightedPenaltyHuberLoss",
"zero_threshold": 0.05,      # Values below this are considered "zero"
"delta": 0.5,                # Huber loss transition point
"non_zero_weight": 5.0,      # Base weight for non-zero targets
"false_positive_weight": 2.0, # Multiplier for false alarms
"false_negative_weight": 3.0  # Multiplier for missed conflicts
```

**Weight calculation:**
- Zero target, zero prediction: `1.0`
- Non-zero target, correct: `non_zero_weight` (5.0)
- False positive: `1.0 × false_positive_weight` (2.0)
- False negative: `non_zero_weight × false_negative_weight` (15.0)

### TweedieLoss

For data with excess zeros and heavy-tailed positive values. The power parameter `p` controls the distribution family:

```python
"loss_function": "TweedieLoss",
"p": 1.5,                    # 1 < p < 2 for zero-inflated continuous
"non_zero_weight": 5.0,
"zero_threshold": 0.01
```

- `p = 1.0`: Poisson (discrete counts)
- `p = 1.5`: Compound Poisson-Gamma (**recommended for conflict**)
- `p = 2.0`: Gamma (continuous positive)

### AsymmetricQuantileLoss

When underestimating conflict is costlier than overestimating:

```python
"loss_function": "AsymmetricQuantileLoss",
"tau": 0.75,                 # Higher = penalize underestimation more
"non_zero_weight": 5.0,
"zero_threshold": 0.01
```

- `tau = 0.5`: Symmetric (equivalent to MAE)
- `tau = 0.75`: 3× penalty for underestimation
- `tau = 0.9`: 9× penalty for underestimation

### ZeroInflatedLoss

Explicitly models zero-inflated structure with two components:
1. **Binary**: Is there any conflict? (BCE loss)
2. **Count**: How much conflict? (Huber loss on non-zeros)

```python
"loss_function": "ZeroInflatedLoss",
"zero_weight": 1.0,          # Weight for binary component
"count_weight": 1.0,         # Weight for count component
"delta": 0.5,
"zero_threshold": 0.01
```

---

## 🛡️ Fortress Architecture & Governance

This repository adheres to the **Fortress Architecture**, a set of strict engineering and mathematical standards designed to guarantee absolute scientific integrity and reproducibility in conflict forecasting.

The repository is governed by:
- **[Architectural Decision Records (ADRs)](docs/ADRs/README.md)**: Sequential, authoritative records of every major design choice (e.g., Physical Symmetry, Hardware Integrity, Genomic Firewall).
- **[Class Intent Contracts (CICs)](docs/CICs/README.md)**: Explicit, normative declarations of purpose and responsibility for every critical class.
- **[Reproducibility Manifest](docs/standards/REPRODUCIBILITY_MANIFEST.md)**: The mandatory "DNA" genome that every experiment must declare before execution.

---

## 🔧 API Reference (Symmetrical Access)

Every core class in the Fortress follows the **1-Class-1-File** Zen standard.

### ScalerSelector
```python
from views_r2darts2.utils.scaler_selector import ScalerSelector

# Get a scaler by name
scaler = ScalerSelector.get_scaler("AsinhTransform")
# Get a chained pipeline
pipeline = ScalerSelector.get_chained_scaler("AsinhTransform->StandardScaler")
```

### FeatureScalerManager
```python
from views_r2darts2.utils.feature_scaler_manager import FeatureScalerManager

manager = FeatureScalerManager(
    feature_scaler_map={"AsinhTransform": ["ged_sb"]},
    default_scaler="RobustScaler"
)
```

### Triple Catalogs (The Genomic Firewall)
```python
from views_r2darts2.model.model_catalog import ModelCatalog
from views_r2darts2.utils.loss.loss_catalog import LossCatalog
from views_r2darts2.utils.optimizer_catalog import OptimizerCatalog

# Catalogs automatically enforce DNA completeness at initialization.
loss_fn = LossCatalog(config).get_loss()
model = ModelCatalog(config).get_model("NBEATSModel")
```

### ReproducibilityGate (The Law)
```python
from views_r2darts2.utils.reproducibility_gate import ReproducibilityGate

# Centralized validation of physical and temporal invariants.
ReproducibilityGate.Config.audit_manifest(config)
ReproducibilityGate.Temporal.audit_continuity(partition)
```

---

## 📚 References

- **Darts**: [unit8co/darts](https://github.com/unit8co/darts) - Time series library
- **VIEWS**: [Violence Early-Warning System](https://viewsforecasting.org/) - Conflict forecasting platform
