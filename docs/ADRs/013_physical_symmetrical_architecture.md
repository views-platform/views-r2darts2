# ADR-013: Physical Symmetrical Architecture

**Status:** Accepted  
**Date:** 2026-02-16  
**Deciders:** Simon Polichinel von der Maase  

---

## Context

In complex scientific repositories, organizational entropy leads to "ghost logic"—classes hidden in generically named files (e.g., `utils.py`), combined in "catch-all" buckets, or buried in `__init__.py` files. This increases cognitive load and makes it impossible for both carbon and silicon agents to reliably find logic without global searches.

We need a standard that makes the physical location of code as predictable as its mathematical behavior.

---

## Decision

This repository adopts the **Absolute Physical Symmetry Standard** (The "Zen Standard").

### 1. The 1-Class-1-File Invariant
For every substantial and non-trivial class `MyClass`, there must be exactly one file `my_class.py` where it is the primary resident.
- **CamelCase Class:** `DartsForecaster`
- **snake_case File:** `darts_forecaster.py`

### 2. Predictable Discovery
The folder structure must match the **Ontology (ADR-001)**. A developer should be able to guess the path of any class based on its name and category.
- `ModelCatalog` → `views_r2darts2/model/model_catalog.py`
- `LossCatalog` → `views_r2darts2/utils/loss/loss_catalog.py`

### 3. Consolidation of Heterogeneous Logic
Logic that is not a primary class (monkey-patches, shared exceptions, training callbacks) must be consolidated into **Symmetrical Hubs**:
- All custom exceptions → `utils/exceptions.py`
- All training callbacks → `utils/callbacks.py`
- All library patches → `utils/patches.py`

### 4. Prohibition of Generic Buckets
Generic file names like `utils.py`, `model.py`, `handlers.py`, or `gates.py` are **forbidden**. They are "gravity wells" for entropy and must be split or renamed into specific entities.

---

## Consequences

### Positive
- **Predictability:** Zero-guess discovery of logic.
- **Scalability:** New components have a clear, pre-defined slot.
- **Refactoring Safety:** Imports are specific and meaningful, making moves easier to track.
- **Agent Efficiency:** Silicon agents can operate with 100% path accuracy.

### Negative
- Higher file count.
- Initial refactoring cost (requires updating all internal imports and test patches).

---

## Notes

Physical symmetry is the final layer of the **Fortress Hardening**. It ensures that the repository's appearance is as rigorous as its scientific integrity.
