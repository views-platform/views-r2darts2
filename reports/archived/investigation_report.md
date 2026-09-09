# Model Performance Investigation Report: Main vs. Development
**Date:** 10-02-2026
**Status:** Completed (Root Cause Identified & Fixed)

## 1. Context
A drastic deterioration in model performance was observed when moving from the `main` branch (MSLE ~0.36) to the `development` branch (MSLE ~1.6). This investigation aimed to determine if the high performance on `main` was due to data leakage ("cheating") or if a bug was introduced in `development`.

## 2. Hypotheses & Falsification Results

### Hypothesis A: Feature Look-Ahead (Temporal Leakage)
*   **Theory:** On `main`, the training features (`past_covariates`) were passed as full series without being sliced to the training end-date. It was suspected the model was "peeking" at future values of covariates.
*   **Test:** Modified `_preprocess_timeseries` to strictly slice features to the training end-date.
*   **Result:** **FALSIFIED.** MSLE remained at ~0.36. The model does not rely on future covariate values for its performance.

### Hypothesis B: Global Scaling "Peek" (Distributional Leakage)
*   **Theory:** `main` used per-entity scaling, while `development` moved toward global scaling. It was suspected that the `MinMaxScaler` on `main` was accidentally fitted on the entire dataset (including future peaks), providing a distributional hint to the model.
*   **Test:** Forced a manual "Global Fit" (concatenating all entities before fitting) on both target and feature scalers.
*   **Result:** **FALSIFIED.** MSLE remained at ~0.37. The scaling mode is not the source of the performance drop.

## 3. The "Smoking Gun": Scaling Calibration Bug
*   **Discovery:** The investigation moved to the `sweep_week-nbeats` branch (MSLE 1.7, y_hat_bar 0.01).
*   **Root Cause:** Two-fold architectural regression:
    1.  `Scaler` objects were missing `global_fit=True`, causing inconsistent per-entity scaling for a global model.
    2.  `_inverse_transform_target_scaler` was hardcoded to use only the first entity's fitted parameters (`fitted_params[0]`) for all predictions.
*   **Fix:** Enabled `global_fit=True` and refactored the inverse transform to correctly map scaling parameters to entities.
*   **Result:** **SUCCESS.** MSLE restored to **0.37** and y_hat_bar to **25**. 

## 4. Conclusion
The high performance on `main` (~0.36 MSLE) is **legitimate**. It is not a result of data leakage or "cheating." The deterioration seen in the refactored branches was a mathematical bug in the scaling inversion logic. 

**Recommendation:** Port the `global_fit=True` and robust `_inverse_transform_target_scaler` logic to `development` immediately.
