# Class Intent Contracts README

This directory contains **Intent Contracts** as defined in ADR-006.

An Intent Contract is a human-readable, unambiguous declaration of:

- what a non-trivial class is meant to do,
- what it must never do,
- its invariants,
- and its failure semantics.

Intent Contracts are architectural artifacts.
They are not implementation documentation.

---

## When Is an Intent Contract Required?

An Intent Contract is mandatory for:

- Core domain classes
- Architectural boundary classes
- Orchestration components
- State-owning components
- Classes that enforce invariants
- Classes that modify semantics or transformation

Trivial value objects and pure utility functions do not require one.

---

## Structure of an Intent Contract

Each contract must define:

1. Purpose
2. Responsibility Boundary
3. Invariants
4. Explicit Non-Responsibilities
5. Failure Semantics
6. Observable Effects (if applicable)

Contracts must be clear enough that:

- Tests (ADR-005) can be derived from them.
- Architectural violations can be detected.
- Silicon-based agents cannot reinterpret intent (ADR-007).

---

## Active Contracts

### Orchestration & Management
- `darts_forecasting_model_manager.md`
- `darts_forecaster.md`

### Catalogs (Genome Translators)
- `model_catalog.md`
- `loss_catalog.md`
- `optimizer_catalog.md`
- `scheduler_catalog.md`

### Data Infrastructure
- `views_dataset.md`
- `dataset_builder.md`
- `zarr_store.md`
- `dataset_converters.md`
- `dataset_subclasses.md`

### Transformation
- `feature_scaler_manager.md`
- `scaler_selector.md`
- `inverse.md`
- `darts_bridge.md`
- `frame_builder.md`
- `static_covariates.md`

### Governance & Observability
- `reproducibility_gate.md`
- `fortress_monitoring_callbacks.md`

### Retired
- *views_dataset_darts* → moved to `docs/archive/CIC_views_dataset_darts.md` (class deleted in 0.2.x; replaced by `views_dataset.md`)

---

## Governance Relationship

Intent Contracts are governed by:

- ADR-006 (Intent Contracts for Non-Trivial Classes)
- ADR-003 (Authority of Declarations)
- ADR-005 (Testing Doctrine)

If a class changes meaning, its Intent Contract must be updated.
