# ADR-002: Topology and Dependency Rules

**Status:** Accepted  
**Date:** 2026-02-11  
**Deciders:** Simon Polichinel von der Maase  
**Revised:** 2026-09-10 — re-derived against the 0.2.x codebase (`development` @ `fe7e681`). Original decision unchanged unless stated.  

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

The package has six sub-packages. Measured against `development` @ `fe7e681` (2026-09-10), their import graph is acyclic and forms four layers (from lowest to highest):

### Layer 0: Foundations — `views_r2darts2/infrastructure/` and `views_r2darts2/math/`
- **`infrastructure/`:** `exceptions.py`, `reproducibility_gate.py`, `device.py`, `encoders.py`, `patches.py`, `callbacks.py`. Imports nothing from the rest of the package — with one recorded inversion: `patches.py:45` attempts `from tests.conftest import CLEAN_TORCH_LOAD` inside a `try/except` (register C-25; the symbol no longer exists, so the branch is dead-by-`ImportError`).
- **`math/`:** loss functions and the `WarmupCAWR` scheduler. May import `infrastructure/exceptions.py` (for `NumericalSanityError`) and nothing else internal.
- **Constraint:** Neither may import from any layer above. These are the physical laws and the arithmetic.

### Layer 1: Translators — `views_r2darts2/transformers/` and `views_r2darts2/catalogs/`
- **`transformers/`:** `scaler_selector.py`, `feature_scaler_manager.py`, `inverse.py`, `darts_bridge.py`, `frame_builder.py`, `static_covariates.py`. Imports only within itself. `darts_bridge.py` is the Darts pandas boundary; `dataset/converters.py` (module-level) and function-local imports in `dataset/readers.py` and `dataset/base.py` are the other three sanctioned pandas importers (see ADR-001).
- **`catalogs/`:** the four Genome Translators. Import `infrastructure/` and `math/` only.
- **Constraint:** May depend on Layer 0. `transformers/` and `catalogs/` do not import each other.

### Layer 2: The Dataset — `views_r2darts2/dataset/`
- `base.py` (`ViewsDataset`), `builder.py`, `converters.py`, `readers.py`, `subclasses.py`, `zarr_store.py`.
- **Constraint:** May depend on Layers 0 and 1 (`transformers/` for scaling and the Darts bridge; `infrastructure/encoders.py`). Must not import `catalogs/` or `engines/`. Intra-package cycles (`base ↔ subclasses`, `base ↔ builder`) are broken by function-body imports and must stay that way.

### Layer 3: Engines — `views_r2darts2/engines/`
- `darts_forecaster.py`, `darts_forecasting_model_manager.py`.
- **Constraint:** The highest layer. May depend on everything below. The only layer that coordinates the lifecycle of artifacts and data flows. It imports `views_pipeline_core` lazily in the manager; `dataset/base.py` also imports `views_pipeline_core` lazily (predstore/datastore/appwrite persistence methods) — a sanctioned Layer-2 exception because those are storage back-ends, not orchestration.

---

## Topological Invariants

1.  **Upward Imports are Forbidden:** `infrastructure/` and `math/` must never import `transformers/`, `catalogs/`, `dataset/` or `engines/`. `dataset/` must never import `catalogs/` or `engines/`. `catalogs/` must never import `engines/` — `infrastructure/device.py` exists precisely to break a former `catalogs → engines → catalogs` cycle, and must not be folded back.
2.  **Stateless Flow:** Information flows down (configurations, requirements) and results flow up (predictions, metrics).
3.  **The "Ghost" Boundary:** The engines layer must interact with models via the `ModelCatalog` or `DartsForecaster` interface, never by reaching into the internal private methods of a PyTorch module. The one sanctioned exception is `transformers/inverse.py`, which confines all access to Darts' private scaler attributes to a single module (see ADR-012, D-03).
4.  **No Mechanical Enforcement Yet:** There is no import-linter contract in `pyproject.toml`. This ADR is enforced by review and by the measurement recorded above. Adding a contract is tracked in the register.

---

## Forbidden Patterns

- **Circular Logic:** A model calling a method in the `DartsForecastingModelManager`.
- **Leakage:** Data handlers importing `ReproducibilityGate` is allowed, but `ReproducibilityGate` importing a specific `Forecaster` is a violation.
- **Convenience shortcuts:** Importing `ModelCatalog` inside a loss module under `views_r2darts2/math/` to get a parameter.

---

## Consequences

### Positive
- **Modularity:** Layer 0 and 1 can be tested in isolation.
- **Cognitive Load:** When working in `infrastructure/` or `math/`, you don't need to understand `engines/`.
- **Predictable Refactoring:** You can change `engines/` without any risk of affecting the math in `views_r2darts2/math/`.

### Negative
- Requires more careful placement of new code.
- May require adding interfaces or "bridging" objects if two layers need to share a concept.

---

## Notes

Topology governs *structure*. The specific handshake rules (what data looks like at the boundary) are governed by ADR-009 (Boundary Contracts).
