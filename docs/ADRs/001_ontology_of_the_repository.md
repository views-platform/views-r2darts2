# ADR-001: Ontology of the Repository

**Status:** Accepted  
**Date:** 2026-02-11  
**Deciders:** Simon Polichinel von der Maase  
**Revised:** 2026-09-10 — re-derived against the 0.2.x codebase (`development` @ `fe7e681`). Original decision unchanged unless stated.  

---

## Context

This repository supports long-lived development under research uncertainty. Without an explicit ontology, deep learning systems tend to accumulate "convenience" objects that mix responsibilities (e.g., a script that both generates data and trains a model) or semantics that exist only in naming conventions.

An explicit ontology is required to define **what kinds of things are allowed to exist** in this repository.

---

## Decision

This repository defines a **closed set of conceptual categories** ("entities"). Anything that does not clearly belong to one of these categories is considered out of scope.

---

## Core Ontological Categories

### 1. The DNA (Configurations)
- **Purpose:** Authoritative declarations of an experiment's identity (hyperparameters, temporal steps, loss choices).
- **Origin:** Delivered via `views_pipeline_core` as a merger of model-specific configs declared in the upstream orchestration repo `views_models`.
- **Genomic Polymorphism:** The DNA is not monolithic. It consists of a **Core Genome** (universal parameters like `random_state`) and an **Algorithm-Specific Genome**. The requirements for an experiment's DNA are determined dynamically by the `algorithm` key. 
- **Constraint:** Parameters irrelevant to a specific architecture (e.g., `use_static_covariates` for N-BEATS) are not permitted in its manifest.
- **Authority:** Authoritative.
- **Stability:** Evolving (new research needs new genes in `views_models`), but once a run starts, the DNA is immutable within this library.
- **Must not contain:** Runtime logic or data tensors.

### 2. The Fortress (Gates)
- **Purpose:** Stateless runtime validators (`ReproducibilityGate`) that enforce physical invariants (temporal continuity, numerical sanity, DNA completeness).
- **Physical Standard:** Must live in `views_r2darts2/infrastructure/reproducibility_gate.py`. Exception definitions live in `views_r2darts2/infrastructure/exceptions.py`.
- **Authority:** Authoritative (The "Law").

### 3. The Dataset (`ViewsDataset`)
- **Purpose:** The single source of truth for all data operations — ingest (parquet / `views_frames` / Zarr), slicing, scaler fitting and application, log transforms, inverse transforms, and construction of Darts `TimeSeries`. Zarr-backed, disk-resident, lazily Dask/xarray-loaded. The forecaster and manager hold no data-manipulation logic of their own; they delegate to it.
- **Physical Standard:** `ViewsDataset` lives in `views_r2darts2/dataset/base.py`. Its supporting entities live in the same package: `DatasetBuilder` (`builder.py`), the converter family (`converters.py`), the level-of-analysis subclasses (`subclasses.py`), `ZarrStore` (`zarr_store.py`), and source detection (`readers.py`). The pandas boundary is confined to `views_r2darts2/transformers/darts_bridge.py`.
- **Authority:** Derived (from raw data) — but it *owns* scaler state once fitted, which the Forecaster of 0.1.x used to own.

### 4. Forecasters (`DartsForecaster`)
- **Purpose:** Slim orchestration of one Darts model against one `ViewsDataset` and one partition: train, predict, save, load. Delegates all preprocessing and scaler state to the dataset.
- **Physical Standard:** Must live in `views_r2darts2/engines/darts_forecaster.py`.
- **Authority:** Operational.

### 5. Artifacts
- **Purpose:** Immutable persistence of a trained Forecaster (weights + scaler states).
- **Authority:** Derived.

### 6. The Manager (`DartsForecastingModelManager`)
- **Purpose:** Orchestration of the lifecycle (Train -> Save -> Evaluate -> Forecast).
- **Physical Standard:** Must live in `views_r2darts2/engines/darts_forecasting_model_manager.py`. Inherits from `views_pipeline_core`'s `ForecastingModelManager`, resolved lazily so the package imports without the optional `manager` extra.
- **Authority:** Execution.

### 7. Catalogs (The Quadruple Catalog Architecture)
- **Purpose:** Genome Translators that map DNA to concrete instances.
    - **ModelCatalog:** Orchestrates algorithms; also resolves Darts likelihood objects.
    - **LossCatalog:** Orchestrates mathematical objectives.
    - **OptimizerCatalog:** Orchestrates PyTorch optimizers.
    - **SchedulerCatalog:** Orchestrates learning-rate schedulers.
- **Physical Standard:** Each lives in its own file under `views_r2darts2/catalogs/`, matching the class name.
- **Authority:** Translation.

---

## Stability Rules

- **The Fortress**, **DNA Schemas**, and **Catalogs** are the most stable layers.

---

## Explicit Non-Entities

- **Implicit Semantics:** Behavior inferred from filenames or folder structures is forbidden.
- **Mixed-Role Scripts:** A single file must not act as both a "Gate" and a "Model." (See ADR-013: Physical Symmetry).
- **Ghost Imports:** Importing this package from a stale or sibling copy (e.g. a `temp-views-r2darts2/` checkout, or a path outside the current workspace) is a violation of ontology — see the workspace-integrity check in `tests/conftest.py`. Declared third-party dependencies (`darts`, `torch`, `views_frames`, `views_pipeline_core`) are not ghosts.

---

## Consequences

### Positive
- Shared vocabulary (e.g., "This belongs in the DNA, not the Forecaster").
- Clear review criteria: Does this new class fit an ontological category?
- Prevents "God Objects" that manage both data and math.

### Negative
- Requires upfront design thinking.
- Some "quick experiments" may be blocked until they are ontologicaly mapped.

---

## Notes

This ADR defines *what exists*. Dependency rules (who can talk to whom) are defined in ADR-002 (Topology).
