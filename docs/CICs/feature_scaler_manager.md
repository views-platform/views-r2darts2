# Class Intent Contract: FeatureScalerManager

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-02-11  
**Related ADRs:** ADR-001, ADR-002, ADR-012  

---

## 1. Purpose

The `FeatureScalerManager` is a specialized orchestrator responsible for applying heterogeneous scaling strategies to different groups of covariates (features). 

> **Its primary goal is to ensure that complex feature sets (e.g., combining economic indicators, conflict counts, and geographic lags) are correctly transformed while maintaining the global calibration required for cross-sectional learning.**

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** manage target scaling (delegated directly to `DartsForecaster`).
- This class does **not** implement individual scaler logic (delegated to `ScalerSelector` and Sklearn).
- This class does **not** perform data cleaning or missing value imputation.
- This class does **not** handle file I/O or artifact persistence.

---

## 3. Responsibilities and Guarantees

- **Guarantees Group-Specific Isolation:** Ensures that each feature is scaled only by its assigned group scaler, preventing leakage of scales across unrelated feature types.
- **Enforces Transformation Sequence:** Correctly manages the forward and inverse execution of "Chained Scalers" (e.g., Asinh -> Standard) using Darts native `Pipeline` (ADR-012).
- **Ensures Global Calibration:** Guarantees that all internal scalers are instantiated with `global_fit=True` to preserve the semantic meaning of values across different entities (countries).
- **Guarantees Shape Preservation:** Ensures that transformations do not collapse the sample dimension, maintaining compatibility with probabilistic forecasting tensors.
- **Ensures Total Coverage:** Guarantees that any feature not explicitly mapped in the configuration is automatically assigned to a `default_scaler`.

---

## 4. Inputs and Assumptions

- **Scaler Map:** Assumes a dictionary mapping scaler types/chains to lists of feature names (Simple or Named Group format).
- **Feature Registry:** Assumes a complete list of `all_features` to identify unmapped covariates.
- **Data Shape:** Assumes inputs are lists of `darts.TimeSeries` objects.

---

## 5. Outputs and Side Effects

- **Scaled Data:** Produces lists of `darts.TimeSeries` where components have been transformed in-place or returned as new objects.
- **State Mutation:** Updates internal fitted states for each group-specific scaler during `fit()`.
- **Logging:** Emits descriptive logs of the mapping structure during initialization.

---

## 6. Failure Modes and Loudness

- **Collision Failure:** Raises `ValueError` if a single feature is assigned to multiple scaling groups.
- **Unfitted Transform:** Fails loudly if `transform()` is called before `fit()`.
- **Config Ambiguity:** Raises `ValueError` if the `feature_scaler_map` follows an unrecognized schema.
- **Mathematical Insanity:** Passes through numerical errors from underlying Sklearn scalers if input data contains prohibited values (Infs/NaNs).

---

## 7. Boundaries and Interactions

- **Upstream:** Orchestrated by `DartsForecaster`.
- **Physical Zen:** Lives in `views_r2darts2/utils/feature_scaler_manager.py`.
- **Factory:** Depends on `ScalerSelector` for individual estimator instantiation.
- **Framework:** Tight coupling with `darts.dataprocessing.transformers.Scaler` and `Pipeline`.

---

## 8. Examples of Correct Usage

```python
# Named Group Format
scaler_map = {
    "economic": {"scaler": "StandardScaler", "features": ["gdp", "inflation"]},
    "conflict": {"scaler": "AsinhTransform->RobustScaler", "features": ["ged_sb"]}
}
manager = FeatureScalerManager(feature_scaler_map=scaler_map, all_features=cols)
manager.fit(training_data)
scaled_data = manager.transform(test_data)
```

---

## 9. Examples of Incorrect Usage

- **Overlapping Groups:** Assigning `ged_sb` to both "RobustScaler" and "StandardScaler" (causes `ValueError`).
- **Manual Chaining:** Attempting to manually pipe data through two managers instead of using the native `->` chain syntax.

---

## 10. Test Alignment

- **Green Team:** `tests/test_scaling.py` (Verification of simple and named group formats).
- **Green Team:** `tests/test_scaling_robustness.py` (Verification of global_fit and sample dimension preservation).

---

## 11. Evolution Notes

### Known Deviations / Technical Debt
- **Global Fit Enforcement:** While ADR-012 mandates `global_fit=True`, the current implementation still relies on the `ScalerSelector` to pass this correctly. The manager should explicitly enforce `global_fit=True` during its own `_instantiate_scaler` pass to ensure "Fortress" compliance even if the factory defaults change.

---

## End of Contract

This document defines the **intended meaning** of `FeatureScalerManager`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
