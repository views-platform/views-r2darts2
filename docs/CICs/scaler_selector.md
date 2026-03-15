# Class Intent Contract: ScalerSelector

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-02-16  
**Related ADRs:** ADR-001, ADR-002, ADR-012, ADR-013  

---

## 1. Purpose

The `ScalerSelector` is a specialized factory responsible for instantiating data transformation objects (Scalers).

> **Its primary goal is to provide a unified mapping between abstract string identifiers (e.g., "AsinhTransform") and concrete Scikit-Learn or custom estimators, enabling complex data pipelines to be declared in the DNA manifest.**

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** manage groups of features (delegated to `FeatureScalerManager`).
- This class does **not** fit scalers to data or execute transformations.
- This class does **not** handle file I/O for persistent scalers.

---

## 3. Responsibilities and Guarantees

- **Enforces Transformation Registry:** Maintains the authoritative mapping of string keys to concrete Python classes or `FunctionTransformer` instances.
- **Supports Chained Specifications:** Correctly parses and instantiates complex "->"-delimited chains (e.g., "AsinhTransform->StandardScaler") into Darts native `Pipeline` objects.
- **Ensures Global Calibration Defaults:** Guarantees that all Darts-wrapped scalers produced via chains are instantiated with `global_fit=True` (ADR-012).
- **Provides Custom Transformation Logic:** Implements domain-specific transforms like `AsinhTransform` and `SqrtTransform` that are optimized for zero-inflated conflict counts.
- **Provides Unified Config-to-Darts Factory:** `instantiate_darts_scaler()` translates flexible configuration formats (string, list, dict with chain/kwargs) into Darts `Scaler` or `Pipeline` objects, ensuring `global_fit=True` on all produced scalers (ADR-012).

---

## 4. Inputs and Assumptions

- **Scaler Name:** Assumes a string matching an entry in the internal registry.
- **Chain Syntax:** Assumes that multi-step transforms use the `->` separator.
- **Flexible Config:** `instantiate_darts_scaler` accepts `None`, string, list, or dict (with `name`/`kwargs` or `chain` keys).

---

## 5. Outputs and Side Effects

- **Estimator:** Produces an uninstantiated class type or a `partial` function for single scalers.
- **Pipeline:** Produces an instantiated `darts.dataprocessing.Pipeline` for chained scalers.
- **Darts Scaler/Pipeline:** `instantiate_darts_scaler` produces Darts-wrapped `Scaler` or `Pipeline` objects (not raw sklearn estimators).
- **Side Effects:** None. This is a stateless factory.

---

## 6. Failure Modes and Loudness

- **Unknown Scaler:** Raises `ValueError` if the requested name is not in the registry.
- **Invalid Chain:** Raises `ValueError` if a chain specification is syntactically invalid or contains unknown components.
- **Invalid Config Type:** Raises `TypeError` if `instantiate_darts_scaler` receives a type other than None/str/list/dict.
- **Missing Dict Key:** Raises `ValueError` if a dict config lacks both `name` and `chain` keys.

---

## 7. Boundaries and Interactions

- **Upstream:** Primarily consumed by `FeatureScalerManager` and `DartsForecaster`.
- **Physical Zen:** Lives in `views_r2darts2/transformers/scaler_selector.py`.
- **Dependency:** Depends on Sklearn, Numpy, and Darts (`dataprocessing.transformers.Scaler`, `dataprocessing.Pipeline`).

---

## 8. Examples of Correct Usage

```python
# Get a single standard scaler
scaler = ScalerSelector.get_scaler("StandardScaler")

# Get a domain-specific custom transform
asinh = ScalerSelector.get_scaler("AsinhTransform")

# Get a Darts Pipeline chain
pipeline = ScalerSelector.get_chained_scaler("AsinhTransform->RobustScaler")

# Instantiate from flexible config (used by DartsForecaster and FeatureScalerManager)
scaler = ScalerSelector.instantiate_darts_scaler("AsinhTransform->StandardScaler")
scaler = ScalerSelector.instantiate_darts_scaler({"chain": ["AsinhTransform", "RobustScaler"]})
scaler = ScalerSelector.instantiate_darts_scaler(None)  # returns None
```

---

## 9. Examples of Incorrect Usage

- **Direct Sklearn Import:** Importing `StandardScaler` from `sklearn.preprocessing` bypassing the selector (violates ADR-001).
- **Manual String Parsing:** Manually splitting "->" in the Manager instead of calling `get_chained_scaler`.

---

## 10. Test Alignment

- **Green Team:** `tests/test_scaling.py` (Verification of transform registry).
- **Infrastructure:** `tests/test_scaling_robustness.py` (Validation of pipeline chaining).

---

## End of Contract

This document defines the **intended meaning** of `ScalerSelector`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
