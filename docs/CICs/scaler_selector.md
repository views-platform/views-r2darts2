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

---

## 4. Inputs and Assumptions

- **Scaler Name:** Assumes a string matching an entry in the internal registry.
- **Chain Syntax:** Assumes that multi-step transforms use the `->` separator.

---

## 5. Outputs and Side Effects

- **Estimator:** Produces an uninstantiated class type or a `partial` function for single scalers.
- **Pipeline:** Produces an instantiated `darts.dataprocessing.Pipeline` for chained scalers.
- **Side Effects:** None. This is a stateless factory.

---

## 6. Failure Modes and Loudness

- **Unknown Scaler:** Raises `ValueError` if the requested name is not in the registry.
- **Invalid Chain:** Raises `ValueError` if a chain specification is syntactically invalid or contains unknown components.

---

## 7. Boundaries and Interactions

- **Upstream:** Primarily consumed by `FeatureScalerManager` and `DartsForecaster`.
- **Physical Zen:** Lives in `views_r2darts2/utils/scaler_selector.py`.
- **Dependency:** Strictly Layer 0 (ADR-002); depends only on Sklearn and Numpy.

---

## 8. Examples of Correct Usage

```python
# Get a single standard scaler
scaler = ScalerSelector.get_scaler("StandardScaler")

# Get a domain-specific custom transform
asinh = ScalerSelector.get_scaler("AsinhTransform")

# Get a Darts Pipeline chain
pipeline = ScalerSelector.get_chained_scaler("AsinhTransform->RobustScaler")
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
