# Class Intent Contract: ViewsDataset

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-09-10  
**Related ADRs:** ADR-001, ADR-002, ADR-009, ADR-012, ADR-016  

---

## 1. Purpose

`ViewsDataset` is the single source of truth for data in the 0.2.x package: a lazy, Zarr-backed spatiotemporal dataset (time × entity × sample) that owns ingest, slicing, scaler state, log transforms, inverse transforms, and construction of Darts `TimeSeries`.

> **Everything that touches data goes through it. The forecaster and manager hold no data-manipulation logic of their own.**

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** train or run models.
- This class does **not** choose scalers — it instantiates what the DNA declares via `ScalerSelector` / `FeatureScalerManager`.
- This class does **not** validate the DNA manifest (that is `ReproducibilityGate.Config`, called by the catalogs and manager).
- This class does **not** touch pandas except at one function-local import in `to_darts_timeseries` (`base.py:1659`, for `pd.RangeIndex`/`pd.Index`/`pd.Series` that Darts requires) and through the two modules that are allowed to: `dataset/converters.py` (ingest) and `transformers/darts_bridge.py` (Darts boundary).

---

## 3. Responsibilities and Guarantees

- **Disk-resident by construction:** the whole dataset lives as chunked Zarr arrays in a `ZarrStore` temp directory; every accessor returns a lazy Dask-backed `xarray` object, so peak memory is bounded by the largest chunk.
- **One ingest path per source kind:** DataFrame, parquet, `views_frames.PredictionFrame`, `views_frames.FeatureFrame`, Zarr, or an in-memory `xarray.Dataset` — dispatched to the matching converter; an unsupported *object* raises `TypeError`, a `str`/`Path` with an unrecognised suffix raises `ValueError` from `readers._detect_path_type`.
- **Dimension names are validated** (`validate_indices`), and the level-of-analysis subclasses (`CMDataset`, `PGMDataset`, …) add one invariant each on top.
- **Scaler state is owned here:** `fit_scalers` fits target and feature scalers on whatever `time_ids` the caller passes — **the default `time_ids=None` fits on the full dataset, test period included, with no warning** (`base.py:1139`, docstring `:1151`; register C-49). `DartsForecaster.train()` passes the training window; any other caller must too. It returns the aligned `(targets, past_covariates)` lists in one call (`_split_targets_covariates`), so the two can never be misaligned by index.
- **Inverse transforms preserve the sample dimension** for probabilistic predictions (via `transformers/inverse.py`).
- **Predictions are ingested with `clip_negatives=True` by default** (`ingest_darts_predictions`, `ingest_numpy_predictions`). This is the current behaviour; whether it should be is register **D-05** / ADR-016.
- **Streaming construction:** `ViewsDataset.builder(...)` yields a `DatasetBuilder` for datasets too large to hold in RAM.
- **Incremental construction:** `create_empty` / `add_row` / `add_batch` mutate the store in place — a second write path alongside the builder, which bypasses `build_schema_attrs` (see the converters CIC).
- **Structural NaN → 0:** on the Darts path, `to_darts_timeseries` applies `np.nan_to_num(nan=0.0)` (`base.py:1645`) on the premise that NaN means "entity absent for those time steps". A genuinely missing observation becomes a zero silently. This is register **D-07** against the fortress protocol's explicit `nan_to_num` prohibition.

---

## 4. Inputs and Assumptions

- A `source` of one of the supported kinds, and `targets` when the source does not carry that role itself.
- Time and entity dimension names from the VIEWS vocabulary (`readers.TIME_IDS`, `readers.ENTITY_IDS`).
- Callers use it as a context manager or call `close()`; the backing `ZarrStore` is otherwise removed at `__del__`/`atexit`.

---

## 5. Outputs and Side Effects

- Lazy `xarray` tensors (`to_tensor`, `get_subset_tensor`), Darts `TimeSeries` lists (`to_darts_timeseries`, `get_scaled_darts_timeseries`), `views_frames` frames (`to_predictionframe`, `to_featureframe`).
- Persistence: `save_parquet`, `save_zarr`, `save_zarrzip`, `save_npz`; `save_predstore` / `save_appwrite` and `from_predstore_latest` / `from_appwrite_latest` delegate to `views_pipeline_core` (optional extra).
- **Side effects:** creates and later deletes a temp directory; mutates its own scaler state on `fit_scalers`.

---

## 6. Failure Modes and Loudness

- `TypeError` — unsupported source object; `ValueError` — path with unrecognised suffix.
- `ValueError` — missing required dimension; targets not found among columns; `split_data` on a prediction dataset; `to_predictionframe` outside prediction mode.
- `RuntimeError` — `get_scaled_darts_timeseries` before `fit_scalers`.
- `LookupError` — `from_*_latest` finds nothing.
- **Silent:** the inverse path falls through to unscaled values, with no log line, when a scaler's fitted params cannot be extracted (register C-10).

---

## 7. Boundaries and Interactions

- **Physical Zen:** `views_r2darts2/dataset/base.py` (1,727 lines — the largest module in the package; C-19).
- **Layer:** ADR-002 Layer 2. Imports `dataset/*`, `transformers/*`, `infrastructure/encoders.py`. Must not import `catalogs/` or `engines/`.
- **Consumers:** `DartsForecaster`, `DartsForecastingModelManager`, `transformers/frame_builder.py`.

---

## 8. Examples of Correct Usage

```python
with ViewsDataset(source="cm_features.parquet", targets=["ged_sb"]) as ds:
    targets, past_cov = ds.fit_scalers(target_scaler="AsinhTransform", ..., return_series=True)
    # train ...
    frames = ds.to_predictionframe()
```

---

## 9. Examples of Incorrect Usage

- Fitting a scaler outside the dataset and passing pre-scaled series to the forecaster.
- Holding a reference past `close()` — the Zarr directory is gone.
- Reading `_ds` directly instead of the accessor methods.

---

## 10. Test Alignment

- **Green:** `tests/test_views_dataset.py` (42 tests: ingest kinds, `to_darts_timeseries` incl. single-row and empty entity, scalers, persistence), `tests/test_parquet_loader.py`, `tests/test_parity_e2e.py`, `tests/test_builder.py`, `tests/test_streaming_predict_builder.py`.
- **Red:** schema failures in `tests/test_views_dataset.py` (missing target column, empty targets, missing file, unsupported source).
- **Lifecycle:** `tests/test_zarr_cleanup.py`.
- **Not covered:** an explicit cross-entity-contamination guard (C-14 residual); the C-10 silent passthrough.

---

## 11. Evolution Notes

- `to_darts_timeseries` is 140 lines; `ingest_numpy_predictions` 103 (C-19).
- Cyclic-encoder columns are computed here and then discarded before `fit` because `_split_targets_covariates` selects only `self.features` (C-22).
- The module docstring's "pandas-free" claim is false as written: `converters.py` imports pandas at module level, and `readers.py` and this module import it locally (register C-45).

---

## End of Contract

This document defines the **intended meaning** of `ViewsDataset`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
