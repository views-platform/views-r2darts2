# ADR-006: Intent Contracts for Non-Trivial Classes

**Status:** Accepted — **compliance open**, see register D-02  
**Date:** 2026-02-11  
**Deciders:** Simon Polichinel von der Maase  
**Revised:** 2026-09-10 — re-derived against the 0.2.x codebase (`development` @ `fe7e681`). Original decision unchanged unless stated.  

---

## Context

In conflict forecasting, deep learning models often accumulate implicit responsibilities. For example, a "Forecaster" might silently start handling data cleaning or metric calculation. This leads to semantic drift, where a class no longer does what its name suggests, making refactoring dangerous and reproducibility fragile.

Tests verify *how* a class works, but they don't capture *what* it is meant to do. We need explicit intent contracts to preserve scientific meaning.

---

## Decision

All **non-trivial and substantial classes** in this repository must have an explicit **intent contract**.

An intent contract is a short, human-readable declaration of:
- **Purpose:** What is the class meant to achieve?
- **Non-Goals:** What is it explicitly *not* responsible for?
- **Guarantees:** What invariants does it promise to maintain?
- **Failure Behavior:** How does it fail when its assumptions are violated?

---

## Non-Trivial Classes in `views-r2darts2`

The following are automatically considered non-trivial and must maintain an intent contract:

- **Managers (`DartsForecastingModelManager`):** Orchestrate the high-level lifecycle.
- **Forecasters (`DartsForecaster`):** Couple one model to one dataset and one partition; delegate preprocessing to the dataset.
- **Gates (`ReproducibilityGate`):** Enforce physical and temporal invariants.
- **Catalogs (`ModelCatalog`, `LossCatalog`, `OptimizerCatalog`, `SchedulerCatalog`):** Translate the DNA manifests into concrete instances and enforce the Genomic Firewall.
- **The Dataset (`ViewsDataset`, `DatasetBuilder`, `ZarrStore`, the converter family):** Own all data ingest, slicing, scaling, and inverse transforms.
- **Scaling (`FeatureScalerManager`, `ScalerSelector`):** Construct and apply scaler pipelines.

---

## Form of the Contract

The contract must live in the class docstring or a linked Markdown file. It must be unambiguous and readable by both carbon and silicon agents.

### Example: `DartsForecaster`
- **Purpose:** Coupling a Darts model with a `ViewsDataset` and a partition, so that predictions come back on the original data scale.
- **Non-Goals:** Does not own scalers or log-transforms (the dataset does); does not handle W&B logging.
- **Guarantees:** Inverse transforms are applied before returning predictions; predictions are clipped to non-negative.
- **Failure Behavior:** Raises `RuntimeError` if prediction is attempted before the dataset's scalers are fitted.

### Compliance status (2026-09-10)
On `development` @ `fe7e681`, an `Intent Contract:` block is present on `ModelCatalog`, `DartsForecaster`, and eleven callbacks — and absent on `ViewsDataset`, `DartsForecastingModelManager`, `LossCatalog`, `OptimizerCatalog`, `SchedulerCatalog`, `ReproducibilityGate`, `FeatureScalerManager`, `ScalerSelector`, `DatasetBuilder`, and `ZarrStore`. The Markdown contracts under `docs/CICs/` cover the gap for now. Bringing the docstrings into compliance is a code change, tracked as register D-02 / C-39; this ADR is not marked compliant until then.

---

## Relationship to Tests

- **Tests must reflect intent:** A Green Team test should verify a "Guarantee." A Red Team test should verify a "Failure Behavior."
- **Intent-First Refactoring:** If you need to change what a class *does*, you must update the contract first.

---

## Consequences

### Positive
- **Architectural Integrity:** Prevents "God Objects" from emerging silently.
- **Safer Refactoring:** You can delete code with confidence if it doesn't serve the declared intent.
- **Better AI Assistance:** Silicon agents (like LLMs) can respect the boundaries defined in the contracts.

### Negative
- Requires more upfront thinking during the design phase.
- Some "routine" changes now require updating docstrings.

---

## Notes

Intent contracts are our defense against "Architectural Rot." They ensure the system continues to mean what we think it means, even as research requirements evolve.
