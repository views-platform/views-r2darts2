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

#### `views_pipeline_core` (DNA) -> Triple Catalogs
- **Contract:** The merged manifest must contain all keys in the relevant Core, Algorithm, Loss, and Optimizer Genomes.
- **Validation:** `ReproducibilityGate.Config.audit_manifest` is called at the entry point of `ModelCatalog`. 
- **Delegation:** `ModelCatalog` delegates genomic validation to `LossCatalog` and `OptimizerCatalog`.
- **Fail-Loud:** Missing or `None` values raise `MissingHyperparameterError`.

#### Raw VIEWS DF -> `_ViewsDatasetDarts`
- **Contract:** Data must match the DNA's `targets` and `features`.
- **Numerical Airlock:** Incoming data is strictly downcast to `float32` and scanned for NaNs/Infs. (ADR-010).

#### `DartsForecaster` -> Prediction DF
- **Contract:** Predictions must be scalar, finite, and scalar-squeezed before returning.
- **Numerical Airlock:** Every prediction is scanned for NaNs. If found, the system raises `NumericalSanityError`.

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
