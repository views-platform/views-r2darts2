# ADR-002: Topology and Dependency Rules

**Status:** Accepted  
**Date:** 2026-02-11  
**Deciders:** Simon Polichinel von der Maase  

---

## Context

In complex machine learning systems, fragility often emerges from uncontrolled dependencies. Without explicit topology rules, high-level orchestration code becomes coupled to low-level tensor operations, circular dependencies emerge (e.g., a model needing to know about its manager), and refactoring becomes impossible without breaking the entire system.

A clear rule is required to define **who may depend on whom**.

---

## Decision

This repository enforces a strict, **directional dependency structure**. Dependencies must follow the declared architectural direction. No component may depend on a layer above it.

Circular dependencies are forbidden. Cross-layer "shortcuts" are forbidden.

---

## The Layered Hierarchy

We define the following four layers (from lowest to highest):

### Layer 0: Core Utilities (`utils/`)
- **Examples:** `loss_catalog.py`, `scaler_selector.py`, `reproducibility_gate.py`, `exceptions.py`.
- **Constraint:** Must remain dependency-free (except for standard libraries and framework-agnostic torch/numpy). They must never import from layers above them.

### Layer 1: Data Handling (`data/`)
- **Examples:** `views_dataset_darts.py`.
- **Constraint:** May depend on Layer 0 (for gates and scaling definitions). Must not depend on models or managers.

### Layer 2: Model & Forecasting (`model/`)
- **Examples:** `model_catalog.py`, `darts_forecaster.py`.
- **Constraint:** May depend on Layer 1 (for data handling) and Layer 0 (for loss and scaling). Must not depend on the Orchestration Layer.

### Layer 3: Orchestration & Management (`manager/`)
- **Examples:** `darts_forecasting_model_manager.py`.
- **Constraint:** The "highest" layer. May depend on all layers below it. This is the only layer allowed to coordinate the lifecycle of artifacts and data flows.

---

## Topological Invariants

1.  **Upward Imports are Forbidden:** `utils` must never import `model`. `data` must never import `manager`.
2.  **Stateless Flow:** Information flows down (configurations, requirements) and results flow up (predictions, metrics).
3.  **The "Ghost" Boundary:** The orchestration layer (`manager`) must interact with models via the `ModelCatalog` or `DartsForecaster` interface, never by reaching into the internal private methods of a PyTorch module.

---

## Forbidden Patterns

- **Circular Logic:** A model calling a method in the `DartsForecastingModelManager`.
- **Leakage:** Data handlers importing `ReproducibilityGate` is allowed, but `ReproducibilityGate` importing a specific `Forecaster` is a violation.
- **Convenience shortcuts:** Importing `ModelCatalog` inside `loss.py` to get a parameter.

---

## Consequences

### Positive
- **Modularity:** Layer 0 and 1 can be tested in isolation.
- **Cognitive Load:** When working in `utils`, you don't need to understand the `manager`.
- **Predictable Refactoring:** You can change the `manager` without any risk of affecting the math in `loss.py`.

### Negative
- Requires more careful placement of new code.
- May require adding interfaces or "bridging" objects if two layers need to share a concept.

---

## Notes

Topology governs *structure*. The specific handshake rules (what data looks like at the boundary) are governed by ADR-009 (Boundary Contracts).
