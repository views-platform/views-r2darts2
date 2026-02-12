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
- **Authority:** Authoritative (The "Law").
- **Stability:** Stable. Changes are high-impact and require ADR updates.
- **Must not contain:** Model weights or business logic.

### 3. Data Handlers
- **Purpose:** Manage the transition from raw VIEWS dataframes to Darts-compatible `TimeSeries`.
- **Authority:** Derived (from raw data and configuration).
- **Stability:** Stable.
- **Must not contain:** Model training logic.

### 4. Forecasters (`DartsForecaster`)
- **Purpose:** Stateful wrappers that manage the coupling of a Model, its Scalers, and its Preprocessing state.
- **Authority:** Operational.
- **Stability:** Evolving.
- **Must not contain:** Orchestration logic (like W&B sweep management).

### 5. Artifacts
- **Purpose:** Immutable persistence of a trained Forecaster (weights + scaler states).
- **Authority:** Derived (output of a training run).
- **Stability:** Permanent. Once created, an artifact must never be modified.
- **Must not contain:** Code logic (only state).

### 6. The Manager (`DartsForecastingModelManager`)
- **Purpose:** Orchestration of the lifecycle (Train -> Save -> Evaluate -> Forecast).
- **Authority:** Execution.
- **Stability:** Evolving.
- **Must not contain:** Model-specific math or low-level tensor operations.

---

## Stability Rules

- **The Fortress** and **DNA Schemas** are the most stable layers.
- **Artifacts** are strictly permanent.
- **Forecasters** and **Managers** are allowed to evolve to support new models.

---

## Explicit Non-Entities

- **Implicit Semantics:** Behavior inferred from filenames or folder structures is forbidden.
- **Mixed-Role Scripts:** A single file must not act as both a "Gate" and a "Model."
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
