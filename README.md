# views-r2darts2

Darts-based forecasting for the VIEWS platform

## What changed in v0.2.0

This release eliminates pandas from the entire package. The only pandas touchpoint is in `views_r2darts2/transformers/darts_bridge.py`, where Darts' API requires a `pd.Index` for the time index and a `pd.DataFrame` for static covariates. Every other module is numpy-only.

### Data layer

- **New:** `views_r2darts2.data.parquet_loader.load_views_parquet` reads a VIEWS viewser parquet file directly via `pyarrow.parquet` and returns a `views_frames.FeatureFrame` — no pandas DataFrame is ever materialized.
- **New:** Optional `cache_dir` writes a native `FeatureFrame` save directory (`values.npy` + `identifiers.npz` + `header.json`) on first read and memmaps the values on subsequent reads. Peak RSS stays the working set.
- **Replaced:** `_ViewsDatasetDarts` (which inherited from `views_pipeline_core.data.handlers._ViewsDataset` and held a pandas DataFrame) is now `ViewsDatasetDarts` (no underscore prefix, no inheritance) — it holds a `FeatureFrame` directly and exposes `as_darts_timeseries()` to build per-entity Darts `TimeSeries` on demand.

### Scaler layer

- `FeatureScalerManager` is pandas-free (the legacy `pd.RangeIndex` usage is replaced by `TimeSeries.from_values` which auto-creates a RangeIndex).
- The duplicated inverse-transform logic that lived in both `FeatureScalerManager` and `DartsForecaster` is extracted into `views_r2darts2.transformers.inverse` — a single set of helpers that confine all Darts private-attribute access (`_fitted_params`, `_fit_called`) to one location.

### Engine layer

- `DartsForecaster.predict()` now returns `dict[str, PredictionFrame]` (one frame per target column) instead of a pandas DataFrame. Each frame carries a `SpatioTemporalIndex` of `(time, entity)` pairs and a `(N, S)` float32 value array.
- `DartsForecastingModelManager` (name unchanged per request) lazy-imports `views_pipeline_core` so the rest of the package is importable without it. The duplicated 17-kwarg `DartsForecaster(...)` construction is extracted into a single `_build_forecaster` factory.

### Infrastructure layer

- `ReproducibilityGate.Data.audit_dataframe_schema(df: pd.DataFrame, ...)` is replaced by `audit_frame_schema(feature_frame: FeatureFrame, ...)`. The checks are equivalent (MultiIndex → 2-D SpatioTemporalIndex; column presence → feature_names set membership; float64 warning → float32 invariant by construction).
- `patches.py` cleaned: removed ~194 lines of dead code (disabled NBEATS patch, disabled Transformer multi-token decoder patch, unused imports, commented-out soft-σ experiment).
- `callbacks.py` cleaned: removed unused `import math`, fixed `GradientHealthCallback` docstring/default mismatch.

### Package layout

```
views_r2darts2/
├── __init__.py              # public API surface
├── data/
│   ├── __init__.py
│   ├── parquet_loader.py    # load_views_parquet (pyarrow → FeatureFrame)
│   └── views_dataset.py     # ViewsDatasetDarts (FeatureFrame-backed)
├── transformers/
│   ├── __init__.py
│   ├── darts_bridge.py      # ONLY module that imports pandas (Darts boundary)
│   ├── feature_scaler_manager.py
│   ├── inverse.py           # shared inverse-transform helpers
│   ├── scaler_selector.py
│   └── static_covariates.py # numpy-only per-entity fingerprint
├── engines/
│   ├── __init__.py
│   ├── darts_forecaster.py
│   └── darts_forecasting_model_manager.py
├── infrastructure/
│   ├── __init__.py
│   ├── callbacks.py
│   ├── device.py            # get_device (broke circular import)
│   ├── encoders.py
│   ├── exceptions.py
│   ├── patches.py
│   └── reproducibility_gate.py
├── catalogs/
│   └── ... (loss, model, optimizer, scheduler)
└── math/
    └── ... (17 loss classes + 2 LR schedulers)
```

## Parity guarantees

The refactor preserves **absolute parity** of data as it passes through the dataloader, scaler, model, and inverse scalers:

1. **Parquet → FeatureFrame:** bit-for-bit identical to a direct `pyarrow.parquet.read_table` column read (verified by `tests/test_parquet_loader.py::TestParquetLoaderBitParity`).
2. **FeatureFrame → Darts TimeSeries:** bit-for-bit identical values (verified by `tests/test_darts_bridge.py` and `tests/test_parity_e2e.py::test_darts_bridge_parity`).
3. **Scaler round-trip:** `AsinhTransform->MaxAbsScaler`, `LogTransform`, `SqrtTransform`, `FourthRootTransform` all round-trip to float32 precision (rtol < 1e-5; verified by `tests/test_scaler_selector.py` and `tests/test_feature_scaler_manager.py`).
4. **Memmap cache:** second read produces bit-identical values to the first read (verified by `tests/test_parquet_loader.py::TestParquetLoaderMemmapCache`).
5. **Static covariates:** per-entity fingerprint (mu/sigma/max/trend/sparsity) matches pandas `groupby` to float32 precision (verified by `tests/test_static_covariates.py::test_parity_with_pandas_groupby` — the one test that imports pandas as a reference oracle).

## Usage

### Loading data

```python
from views_r2darts2 import load_views_parquet, ViewsDatasetDarts

frame, features, targets = load_views_parquet(
    "/path/to/validation_viewser_df.parquet",
    targets=["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
    features=["lr_ged_sb_delta", "lr_splag_1_ged_sb", ...],
    cache_dir="/tmp/views_cache",  # optional: memmap-backed cache
)
dataset = ViewsDatasetDarts(
    feature_frame=frame,
    targets=targets,
    features=features,
)
```

### Training + prediction

```python
from views_r2darts2 import DartsForecaster
from darts.models import NBEATSModel

model = NBEATSModel(input_chunk_length=12, output_chunk_length=6, n_epochs=10)
forecaster = DartsForecaster(
    dataset=dataset,
    model=model,
    partition_dict={"train": (121, 500), "test": (501, 552)},
    target_scaler="AsinhTransform",
    feature_scaler_map={
        "AsinhTransform->MaxAbsScaler": ["lr_ged_sb_delta", "lr_splag_1_ged_sb", ...],
    },
    random_state=42,
)
forecaster.train()
predictions = forecaster.predict(sequence_number=0, output_length=36, num_samples=100, mc_dropout=True)
# predictions is dict[str, PredictionFrame] — one frame per target
```

## Tests

```bash
cd views-r2darts2-new
python -m pytest tests/ -v
```

482 tests pass, 7 skipped (the skipped tests require `views_pipeline_core`, which is an optional dependency).
