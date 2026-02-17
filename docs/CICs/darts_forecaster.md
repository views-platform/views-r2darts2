# Class Intent Contract: DartsForecaster

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-02-11  
**Related ADRs:** ADR-001, ADR-002, ADR-006, ADR-008, ADR-010, ADR-011, ADR-012  

---

## 1. Purpose

The `DartsForecaster` is a stateful wrapper that manages the tight coupling between a forecasting model and its required preprocessing pipeline (scalers and log-transforms). 

> **Its primary goal is to ensure that predictions are returned on the original "raw" scale by maintaining the exact state of transformations applied during training.**

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** manage Weights & Biases logging (delegated to the Manager).
- This class does **not** perform model architecture selection (delegated to the Catalog).
- This class does **not** handle high-level rolling-origin logic (delegated to the Manager).
- This class does **not** implement semantic thresholding or "clipping" for business logic (ADR-010).

---

## 3. Responsibilities and Guarantees

- **Guarantees Scaler Coupling:** Ensures that the target scaler used during `train()` is exactly the one used for `inverse_transform()` during `predict()`.
- **Enforces Numerical Precision:** Guarantees that all data enters the model as `float32` (ADR-010).
- **Preserves Probabilistic Calibration:** Enforces `global_fit=True` for target scalers to preserve sample dimensions (ADR-012).
- **Hardware Self-Healing:** Audits and restores model weights to the target device before every prediction to prevent Darts CPU-drift (ADR-011).
- **Temporal Boundary Enforcement:** Invokes `ReproducibilityGate.Temporal` to ensure no future peeking occurs during data slicing.

---

## 4. Inputs and Assumptions

- **Dataset:** Assumes a `_ViewsDatasetDarts` instance which provides standard VIEWS time/entity indices.
- **Model:** Assumes a `TorchForecastingModel` instance (already instantiated).
- **Partition:** Assumes a validated partition dictionary provided by the Manager.
- **Fitted State:** Assumes `predict()` will only be called after `train()` or `load_model()` has successfully fitted the scalers.

---

## 5. Outputs and Side Effects

- **Transformed Data:** Produces `TimeSeries` objects scaled and transformed for model consumption.
- **Raw Predictions:** Produces `pd.DataFrame` objects on the original data scale.
- **State Mutation:** Updates its internal `scaler_fitted` flag and persists scaler states during training.
- **Persistence:** Saves and loads comprehensive artifacts containing both model weights and scaler pipelines.

---

## 6. Failure Modes and Loudness

- **Unfitted Predict:** Raises `RuntimeError` if prediction is attempted without fitted scalers.
- **Device Failure:** Raises `RuntimeError` if hardware self-healing (CPU -> GPU) fails (ADR-008).
- **Numerical Insanity:** Fails loudly if NaNs or Infs are detected in the model input stream.
- **Schema Mismatch:** Fails if the provided dataset components do not match the expected DNA manifest.

---

## 7. Boundaries and Interactions

- **Upstream:** Managed by `DartsForecastingModelManager`.
- **Physical Zen:** Lives in `views_r2darts2/model/darts_forecaster.py`.
- **Downstream:** Orchestrates `FeatureScalerManager` and specific Darts `TorchForecastingModel` instances.
- **Validator:** Deeply coupled with `ReproducibilityGate` for boundary auditing.

---

## 8. Examples of Correct Usage

```python
# Instantiate and train
forecaster = DartsForecaster(dataset=ds, model=m, partition_dict=p)
forecaster.train()

# Predict with automatic scaling handling
results_df = forecaster.predict(sequence_number=0)
```

---

## 9. Examples of Incorrect Usage

- **Manual Scaling:** Fitting a scaler outside the forecaster and then passing scaled data to `train()`.
- **Direct Model Fit:** Calling `forecaster.model.fit()` directly, bypassing the gate-protected `forecaster.train()` method.
- **Inferred Device:** Manually moving the model to a device without updating the `forecaster.device` property.

---

## 10. Test Alignment

- **Green Team:** `tests/test_forecaster.py` (Functional integration).
- **Green Team:** `tests/test_scaling.py` (Transformation accuracy).
- **Red Team:** `tests/test_reproducibility_infra.py` (Device-drift injection).

---

## 11. Evolution Notes

### Known Deviations / Technical Debt
- **Silent Data Cleaning:** Currently replaces `NaN` and `Inf` with `0.0` in `_process_predictions`. This violates ADR-008 (Fail-Loud) and should be refactored to raise an error if numerical insanity is detected post-inference.
- **Manual Log Implementation:** Implements its own `_apply_log_to_features`. This should be refactored to use Darts' native `LogTransformer` inside the scaling `Pipeline` for better consistency.

---

## End of Contract

This document defines the **intended meaning** of `DartsForecaster`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
