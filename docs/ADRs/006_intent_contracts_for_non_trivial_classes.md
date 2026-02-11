# ADR-006: Intent Contracts for Non-Trivial Classes

**Status:** Accepted  
**Date:** 2026-02-11  
**Deciders:** Simon Polichinel von der Maase  

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

The following are automatically considered non-trivial:

- **Managers (`DartsForecastingModelManager`):** Orchestrate the high-level lifecycle.
- **Forecasters (`DartsForecaster`):** Manage the coupling of models and stateful preprocessing (scalers).
- **Gates (`ReproducibilityGate`):** Enforce physical and temporal invariants.
- **Catalogs (`ModelCatalog`):** Translate the merged DNA manifests from `views_pipeline_core` into concrete Darts Model instances.
- **Data Handlers:** Manage the transformation of raw VIEWS data to Darts types.

---

## Form of the Contract

The contract must live in the class docstring or a linked Markdown file. It must be unambiguous and readable by both carbon and silicon agents.

### Example: `DartsForecaster`
- **Purpose:** Coupling a Darts model with the exact scalers and log-transforms used during its training.
- **Non-Goals:** Does not handle database connections or W&B logging.
- **Guarantees:** Ensures that inverse transforms are applied in the correct order before returning predictions.
- **Failure Behavior:** Raises `NotFittedError` if prediction is attempted before scalers are fit.

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
