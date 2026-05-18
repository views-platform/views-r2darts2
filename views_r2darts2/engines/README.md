# engines

The **engines** layer owns the full experiment lifecycle: data loading, scaler fitting, model training, prediction, inverse-transform, and artifact persistence. It is the only layer that mutates disk state (saving model weights + scalers) or performs gradient updates.

---

## Files

| File | Class | Responsibility |
|------|-------|----------------|
| `darts_forecaster.py` | `DartsForecaster` | Stateful wrapper coupling a single `TorchForecastingModel` to its scaler pipeline |
| `darts_forecasting_model_manager.py` | `DartsForecastingModelManager` | Orchestrates full experiment runs: config loading, DNA audit, train/predict/save lifecycle |

---

## DartsForecaster

The low-level stateful engine. Holds:
- a `TorchForecastingModel` (from `ModelCatalog`)
- a fitted target `Scaler`
- a fitted `FeatureScalerManager` (per-feature or per-group scalers)
- partition boundaries (`_train_start`, `_train_end`, `_test_start`, `_test_end`)
- static covariate configuration (transform chain, `stat_time_range`)

### Key guarantees

- **Scalers fit on training data only.** `_train_end` is passed as `stat_time_range` when building Darts `TimeSeries` objects — entity-level fingerprint statistics (µ, σ, max, trend, sparsity) are computed exclusively from months ≤ `_train_end`.
- **Float32 enforcement.** All input tensors are downcast to `float32` before entering the model.
- **Numerical sanity checks.** `ReproducibilityGate.Physics.audit_tensor()` is called at data boundaries — NaN or Inf raises `NumericalSanityError` immediately.
- **Probabilistic inverse transform.** Inverse scaler correctly handles the `(samples, time, components)` shape from Monte Carlo dropout or quantile forecasts.

### Usage

```python
forecaster = DartsForecaster(
    dataset=dataset,
    model=model,
    partition_dict={"train": (t_start, t_train_end), "test": (t_test_start, t_test_end)},
    target_scaler="AsinhTransform",
    feature_scaler_map={"AsinhTransform->MaxAbsScaler": ["lr_ged_sb", ...]},
    static_covariate_stats={"transform": "AsinhTransform->MaxAbsScaler"},
    random_state=67,
)

forecaster.fit()
df_predictions = forecaster.predict()
```

---

## DartsForecastingModelManager

The high-level orchestrator. Inherits from `views_pipeline_core.managers.model.ForecastingModelManager`.

### Responsibilities

| Phase | What happens |
|-------|-------------|
| **Handshake** | `ReproducibilityGate.Config.audit_manifest()` validates the DNA before any I/O |
| **Partition resolution** | `_resolve_active_partition_dict()` re-derives train/test windows from `config["steps"]` — prevents the "stale DataLoader" bug where cached windows from a previous run bleed into the current one |
| **Model construction** | Delegates to `ModelCatalog` with the validated config |
| **Dataset construction** | Instantiates `_ViewsDatasetDarts` from the raw VIEWS parquet |
| **Training** | Calls `DartsForecaster.fit()` with `apply_all_patches()` pre-applied |
| **Evaluation** | Calls `DartsForecaster.predict()`, runs metric computation |
| **Artifact persistence** | Saves coupled (model weights + fitted scalers) under `ModelPathManager` paths |

### Patch application

`apply_all_patches()` is called at `__init__`. This applies:
- **`torch.load` patch** — forces `weights_only=False` for environment compatibility
- **N-BEATS dropout patch** — fixes Darts' `_Block.__init__` to pass `dropout` correctly when `MonteCarloDropout` is used (upstream Darts bug workaround)

---

## Data flow

```
Raw parquet
    │
    ▼
_ViewsDatasetDarts.from_views_path()
    │
    ▼
DartsForecaster.__init__()   ← scalers initialized (not yet fitted)
    │
    ▼
DartsForecaster.fit()
    ├── ScalerSelector.fit_transform(train_targets)
    ├── FeatureScalerManager.fit_transform(train_features)
    ├── _ViewsDatasetDarts.as_darts_timeseries(stat_time_range=train_end)
    └── model.fit(train_series, past_covariates, ...)
    │
    ▼
DartsForecaster.predict()
    ├── model.predict(n=output_chunk_length, ...)
    ├── ScalerSelector.inverse_transform(predictions)
    └── DataFrame reconstruction (month_id × country_id)
    │
    ▼
DartsForecastingModelManager  ← saves artifacts, logs metrics
```
