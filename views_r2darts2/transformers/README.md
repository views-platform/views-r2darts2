# transformers

The **transformers** layer handles all data preprocessing: scaler construction, per-feature scaling pipelines, and the conversion from VIEWS multi-index dataframes into Darts-compatible `TimeSeries` collections. It is the data boundary between the VIEWS pipeline and the Darts ecosystem.

---

## Files

| File | Class | Responsibility |
|------|-------|----------------|
| `scaler_selector.py` | `ScalerSelector` | Factory for individual scalers and chained scaler pipelines |
| `feature_scaler_manager.py` | `FeatureScalerManager` | Manages multiple scalers for different feature groups |
| `views_dataset_darts.py` | `_ViewsDatasetDarts` | Converts VIEWS dataframes to Darts `TimeSeries` with static covariate injection |

---

## ScalerSelector

A factory class (all static methods) that maps string names to fitted `darts.dataprocessing.transformers.Scaler` instances. Supports single scalers and chained pipelines via the `->` arrow syntax.

### Available scalers

| Name | Underlying transform | Notes |
|------|---------------------|-------|
| `AsinhTransform` | `arcsinh(x)` | ⭐ Production standard for all conflict/macro features. Handles zeros and negatives; near-linear below 1; compresses outliers like log. |
| `MaxAbsScaler` | `x / max(|x|)` | Maps to [-1, 1]. Required after `AsinhTransform` for cross-entity normalization in concatenation-based models. |
| `StandardScaler` | `(x − µ) / σ` | Poor for zero-inflated data; use only for roughly normal features. |
| `MinMaxScaler` | `(x − min) / (max − min)` | Best for V-Dem indices and bounded percentages. |
| `RobustScaler` | Median + IQR | Resistant to extreme values. |
| `LogTransform` | `log(1 + x)` | Undefined for negatives. Prefer `AsinhTransform`. |
| `SqrtTransform` | `sqrt(max(x, 0))` | Gentler compression; undefined for negatives. |
| `FourthRootTransform` | `(1 + x)^0.25 − 1` | Polynomial inverse (65k cap at overshoot 15 vs sinh's 1.6M). Less explosive than asinh for extreme model overshoots. |
| `YeoJohnsonTransform` | Power transform | Mixed positive/negative; makes data more Gaussian. |
| `QuantileNormal` | Quantile → N(0,1) | Forces Gaussian marginal. |
| `QuantileUniform` | Quantile → U(0,1) | Forces uniform marginal. |

### Chained scalers

Use the `->` operator to compose transforms left-to-right:

```python
from views_r2darts2.transformers.scaler_selector import ScalerSelector

# Single scaler
scaler = ScalerSelector.get_scaler("AsinhTransform")

# Chained pipeline
pipeline = ScalerSelector.get_chained_scaler("AsinhTransform->MaxAbsScaler")
```

**Forward:** `X → Scaler₁.fit_transform(X) → Scaler₂.fit_transform(X') → X_scaled`

**Inverse:** `X_scaled → Scaler₂.inverse_transform → Scaler₁.inverse_transform → X_original`

The chain is invertible to floating-point precision. Triple chains are supported: `"AsinhTransform->RobustScaler->MinMaxScaler"`.

### Why `AsinhTransform->MaxAbsScaler` is the production standard

`AsinhTransform` is element-wise: it compresses each value independently. Syria's `ged_sb ≈ 5000` → 8.5 and Chad's `ged_sb ≈ 3` → 1.8 after asinh. A 4.7× cross-entity scale gap persists. For concatenation-based models (TSMixer, TiDE, TFT) this bias is injected at every block.

`MaxAbsScaler` maps the full feature column to `[-1, 1]` across all 180+ countries, collapsing the cross-entity scale while preserving ordinal rank. The combination:
- Element-wise: `AsinhTransform` compresses outliers and handles zeros/negatives
- Cross-entity: `MaxAbsScaler` removes scale bias between countries

---

## FeatureScalerManager

Manages independent scalers for multiple feature groups. Handles three config formats:

### Format 1: Arrow syntax (recommended)

```python
feature_scaler_map = {
    "AsinhTransform->MaxAbsScaler": [
        "lr_splag_1_ged_sb", "lr_ged_ns", "lr_ged_os",
        "lr_acled_sb", "lr_wdi_ny_gdp_mktp_kd", ...
    ],
}
```

### Format 2: Named group format

```python
feature_scaler_map = {
    "conflict_group": {
        "scaler": "AsinhTransform->MaxAbsScaler",
        "features": ["lr_ged_sb", "lr_ged_ns"],
    },
    "macro_group": {
        "scaler": "LogTransform->StandardScaler",
        "features": ["lr_wdi_ny_gdp_mktp_kd"],
    },
}
```

### Behavior

- A feature assigned to multiple groups raises `ValueError`
- Features in `all_features` but not in any group receive `default_scaler` (if set)
- Scalers are fit on training data only; `transform()` and `inverse_transform()` are called separately

```python
from views_r2darts2.transformers.feature_scaler_manager import FeatureScalerManager

manager = FeatureScalerManager(
    feature_scaler_map={"AsinhTransform->MaxAbsScaler": ["lr_ged_sb", "lr_ged_ns"]},
    default_scaler=None,
    all_features=["lr_ged_sb", "lr_ged_ns"],
)

train_covariates_scaled = manager.fit_transform(train_ts_list)
test_covariates_scaled  = manager.transform(test_ts_list)
```

---

## _ViewsDatasetDarts

Subclass of `views_pipeline_core.data.handlers._ViewsDataset`. The data boundary between VIEWS dataframes and Darts `TimeSeries`.

### Responsibilities

1. **Schema validation at construction.** `ReproducibilityGate.Data.audit_dataframe_schema()` is called immediately — raises at init if expected target or feature columns are missing.

2. **Group-by-entity conversion.** `as_darts_timeseries()` groups the multi-index dataframe by `country_id` and converts each group to a `darts.TimeSeries`, preserving the `(month_id, country_id)` multi-index semantics.

3. **Static covariate injection.** Five per-entity fingerprint statistics are computed from the training partition and attached as `TimeSeries.static_covariates` metadata:

| Statistic | Meaning |
|-----------|---------|
| `target_mu` | Mean conflict level |
| `target_sigma` | Standard deviation (volatility) |
| `target_max` | Peak value (spike extremity) |
| `target_trend` | OLS slope over training window |
| `target_sparsity` | Fraction of zero months (structural peace rate) |

4. **Data-leakage prevention.** The `stat_time_range` parameter restricts which months are used to compute fingerprint statistics. Always pass `stat_time_range=(train_start, train_end)`:

```python
ts_list = dataset.as_darts_timeseries(
    stat_time_range=(training_start_month_id, training_end_month_id),
    static_cov_transform="AsinhTransform->MaxAbsScaler",
)
```

Without `stat_time_range`, statistics are computed from the full dataframe — this would leak test-period conflict levels into the training signal for entity conditioning.

### Static covariate notes by model

| Model | `use_static_covariates` | Result |
|-------|------------------------|--------|
| TFT | ✅ | Injected via VSN+GRN gating — model can learn to suppress low-quality stats |
| TSMixer | ✅ | Blunt concatenation at every block — requires `AsinhTransform->MaxAbsScaler` to prevent scale bias |
| TiDE, BlockRNN, NLinear, DLinear | ✅ | Concatenation-based injection |
| N-HiTS | ⚠️ | Flag accepted by Darts but **not passed to constructor** — stats computed and attached, silently ignored |
| Transformer | ⚠️ | Same as N-HiTS — flag silently ignored |
| N-BEATS | ❌ | Architecturally incompatible — do not set `use_static_covariates=True` |

### Factory method

```python
from views_r2darts2.transformers.views_dataset_darts import _ViewsDatasetDarts

dataset = _ViewsDatasetDarts.from_views_path(
    path_raw="/path/to/raw/data",
    run_type="calibration",
    config=config,
)
```

This loads `{path_raw}/{run_type}_viewser_df.parquet`, constructs the dataset, and immediately validates the schema.
