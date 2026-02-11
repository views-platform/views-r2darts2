# ADR-009: Boundary Contracts and Configuration Validation

**Status:** Accepted  
**Date:** 2026-02-11  
**Deciders:** Simon Polichinel von der Maase  

---

## Context

Complex forecasting systems fail most often at boundaries: between the researcher's configuration and the model catalog, or between raw dataframes and the training tensors. Hidden defaults and ambiguous schemas in these "handshakes" lead to silent semantic drift.

To maintain the "Fortress" integrity, every boundary must be explicit and validated.

---

## Decision

This repository adopts the invariant: **All architectural boundaries must declare explicit contracts and be validated at entry via the Handshake Principle.**

### 1. The Core Handshakes in `views-r2darts2`

#### `views_pipeline_core` (DNA) -> `ModelCatalog`
- **Contract:** The merged manifest (sourced from `views_models`) must contain all keys in `MANDATORY_MANIFEST`.
- **Validation:** `ReproducibilityGate.Config.audit_manifest` is called before model instantiation to verify the handshake from the orchestration layer.
- **Fail-Loud:** Missing or `None` values raise `MissingHyperparameterError`.

#### Raw VIEWS DF -> `DataHandler`
- **Contract:** Data must be non-empty, have the correct entity/time index, and match the DNA's `targets` list.
- **Validation:** Type checking and index integrity checks during `_ViewsDatasetDarts` initialization.

#### `DartsForecaster` -> Prediction DF
- **Contract:** Predictions must be non-negative (clipped if necessary) and contain all target components.
- **Validation:** `_process_predictions` ensures shape and numerical sanity before returning to the manager.

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
