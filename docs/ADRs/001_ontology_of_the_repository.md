# ADR-001: Ontology of the Repository

**Status:** Accepted  
**Date:** 2026-02-11  
**Deciders:** Simon Polichinel von der Maase  

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
- **Physical Standard:** Must live in `reproducibility_gate.py`. Definitions of exceptions must be moved to `exceptions.py`.
- **Authority:** Authoritative (The "Law").

### 3. Data Handlers (`_ViewsDatasetDarts`)
- **Purpose:** Manage the transition from raw VIEWS dataframes to Darts-compatible `TimeSeries`.
- **Physical Standard:** Must live in `views_dataset_darts.py`.
- **Authority:** Derived.

### 4. Forecasters (`DartsForecaster`)
- **Purpose:** Stateful wrappers that manage the coupling of a Model, its Scalers, and its Preprocessing state.
- **Physical Standard:** Must live in `darts_forecaster.py`.
- **Authority:** Operational.

### 5. Artifacts
- **Purpose:** Immutable persistence of a trained Forecaster (weights + scaler states).
- **Authority:** Derived.

### 6. The Manager (`DartsForecastingModelManager`)
- **Purpose:** Orchestration of the lifecycle (Train -> Save -> Evaluate -> Forecast).
- **Physical Standard:** Must live in `darts_forecasting_model_manager.py`.
- **Authority:** Execution.

### 7. Catalogs (The Triple Catalog Architecture)
- **Purpose:** Genome Translators that map DNA to concrete instances.
    - **ModelCatalog:** Orchestrates algorithms.
    - **LossCatalog:** Orchestrates mathematical objectives.
    - **OptimizerCatalog:** Orchestrates PyTorch optimizers.
- **Physical Standard:** Each must live in its own file matching the class name.
- **Authority:** Translation.

---

## Stability Rules

- **The Fortress**, **DNA Schemas**, and **Catalogs** are the most stable layers.

---

## Explicit Non-Entities

- **Implicit Semantics:** Behavior inferred from filenames or folder structures is forbidden.
- **Mixed-Role Scripts:** A single file must not act as both a "Gate" and a "Model." (See ADR-013: Physical Symmetry).
- **Ghost Imports:** Importing from outside the local `views_r2darts2` package is a violation of ontology.

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
