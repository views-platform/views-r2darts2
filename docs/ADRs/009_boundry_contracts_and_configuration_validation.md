# ADR-009: Boundary Contracts and Configuration Validation

**Status:** Accepted  
**Date:** 2026-02-11  
**Deciders:** Simon Polichinel von der Maase  
**Revised:** 2026-09-10 — re-derived against the 0.2.x codebase (`development` @ `fe7e681`). Original decision unchanged unless stated.  

---

## Context

Complex forecasting systems fail most often at boundaries: between the researcher's configuration and the model catalog, or between raw dataframes and the training tensors. Hidden defaults and ambiguous schemas in these "handshakes" lead to silent semantic drift.

To maintain the "Fortress" integrity, every boundary must be explicit and validated.

---

## Decision

This repository adopts the invariant: **All architectural boundaries must declare explicit contracts and be validated at entry via the Handshake Principle.**

### 1. The Core Handshakes in `views-r2darts2`

#### `views_pipeline_core` (DNA) -> The Four Catalogs
- **Contract:** The merged manifest must contain all keys in the relevant Core, Algorithm, Loss, Optimizer, and Scheduler Genomes.
- **Validation:** `ReproducibilityGate.Config.audit_manifest` is called at the entry point of `ModelCatalog`. 
- **Delegation:** `ModelCatalog` delegates genomic validation to `LossCatalog`, `OptimizerCatalog`, and `SchedulerCatalog`.
- **Fail-Loud:** Missing or `None` values raise `MissingHyperparameterError` from `audit_manifest`. The delegated catalogs raise plain `ValueError` for the same condition (`loss_catalog.py`, `optimizer_catalog.py`, `scheduler_catalog.py`); `MissingHyperparameterError` is not a `ValueError` subclass, so a caller catching one misses the other.

#### `views_frames.FeatureFrame` / parquet / Zarr -> `ViewsDataset`
- **Contract:** Data must match the DNA's `targets` and `features`; the time and entity dimensions must be one of the recognised VIEWS levels of analysis.
- **Validation:** `ReproducibilityGate.Data.audit_frame_schema` exists and is fail-loud, **but has no production caller on 0.2.x** — no module under `dataset/` imports the gate; the schema is enforced only by `views_frames` construction and the converters' own `ValueError`s. Register **C-44**. The pandas-based `audit_dataframe_schema` of 0.1.x no longer exists.
- **Numerical Airlock:** Frames are `float32` by construction; the gate scans for NaNs/Infs. (ADR-016).

#### `DartsForecaster` -> `dict[str, PredictionFrame]`
- **Contract:** `predict()` returns one `views_frames.PredictionFrame` per target, finite, on the original data scale, with the sample dimension preserved for probabilistic runs. DataFrame conversion is on demand via `transformers/darts_bridge.py`, never implicit.
- **Numerical Airlock:** Predictions are scanned for NaNs per batch in `DartsForecaster.predict` (`engines/darts_forecaster.py:492-496`) before being written; the dataset's own `ingest_*_predictions` methods are unguarded, and `audit_numerical_sanity` is not invoked on any production path (C-44).

### 2. The Handshake Principle
Validation must occur **before** execution begins. We do not "try and see." We audit the requirements, and if the handshake fails, the run terminates immediately.

### 3. Forbidden Semantic Defaults
No parameter that affects the mathematical identity of a model (loss choice, scaling method, stochastic seed) may have a silent default in the core pipeline. If the researcher doesn't declare it, the system doesn't guess it.

---

## Configuration as a First-Class Artifact

- **Traceability:** The configuration used for a training run must be saved alongside the model weights.
- **Immutability:** Once the handshake is complete and training starts, the configuration must be treated as read-only.

---

## Consequences

### Positive
- **Eliminates Configuration Drift:** You always know exactly what parameters were used.
- **Boundary Robustness:** Errors are caught at the point of entry, not deep inside PyTorch Lightning.
- **Reproducibility:** Forces researchers to be explicit about every choice.

### Negative
- Increases "boilerplate" in `sweep_configs`.
- Requires rigorous maintenance of the `ReproducibilityGate` as the system evolves.

---

## Notes

This ADR operationalizes the "Authority of Declarations" (ADR-003). It defines *where* and *how* we verify that those declarations are valid.
