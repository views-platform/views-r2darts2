# Post-Mortem: The Scaling Calibration Collapse
**Date:** February 11, 2026
**Status:** Resolved
**Impact:** Model MSLE deteriorated from 0.36 to 1.7; predicted fatalities ($\bar{\hat{y}}$) collapsed to near-zero (0.01).

---

## 1. Executive Summary
During the transition from the `main` branch to the `development` branch, a critical regression was introduced in the time-series scaling pipeline. This resulted in a total loss of model calibration, where the model essentially "forgot" how to map its internal normalized predictions back to real-world fatality counts. Initially suspected as a "fix" for data leakage, rigorous testing proved the high performance on `main` was legitimate and the new deterioration was a mathematical inversion bug.

## 2. The Investigation Timeline
*   **22:35 (Feb 10):** A run on `main` confirms 0.36 MSLE (Healthy).
*   **00:15 (Feb 11):** Merged `development` code results in 1.6+ MSLE (Catastrophic).
*   **Discovery Phase:** We hypothesized two paths of "cheating" on `main` (Feature Look-Ahead and Scaling Peeks).
*   **Falsification Phase:** 
    *   **Test A:** Slicing features to the training end-date did not break `main` (MSLE remained 0.36).
    *   **Test B:** Forcing a manual Global Fit on `main` did not break it (MSLE remained 0.37).
    *   **Conclusion:** `main` was NOT cheating. The problem was a new bug in the refactor.
*   **The Breakthrough:** Identified that the new `_inverse_transform_target_scaler` was hardcoded to use the first entity's scaling parameters (`fitted_params[0]`) for all countries.

## 3. Root Cause Analysis
The collapse was caused by a two-fold architectural failure in the scaling refactor:

### A. Missing `global_fit=True`
Global models (N-BEATS, TFT) require consistent scaling. The new code instantiated `Scaler` objects without the `global_fit=True` flag. This defaulted to per-entity scaling, meaning the model was trained on 191 different "dialects" of fatalities where `1.0` meant something different for every country.

### B. The "First-Entity Trap" (The Smoking Gun)
The `_inverse_transform_target_scaler` method attempted to handle probabilistic samples manually. However, its implementation was:
```python
# The BUG
sklearn_scaler = self.target_scaler._fitted_params[0] 
# (Always took the first country, e.g., Afghanistan)
```
During prediction, the model took Sweden's tiny fatality predictions and "un-scaled" them using Afghanistan's massive range. This resulted in predictions that were mathematically tiny (0.01), leading to the 1.7 MSLE.

## 4. The "Explosion" Clue
A key diagnostic moment was removing scaling entirely, which resulted in an expected fatality count ($\bar{\hat{y}}$) of **33 billion**. This confirmed that the model architecture was divergent without normalization, but the near-zero result (0.01) with scaling confirmed the **inversion** was the bottleneck.

## 5. Resolution & Recovery
The following fixes were implemented on the `sweep_week-nbeats` branch:
1.  **Mandatory Global Fit:** Added `global_fit=True` to all `Scaler` instantiations in `_instantiate_scaler`.
2.  **Robust Inversion:** Refactored `_inverse_transform_target_scaler` to correctly map fitted parameters to series indices, ensuring every country uses its own scaler (even if global fit is toggled off).
3.  **Result:** MSLE restored to **0.37** and $\bar{\hat{y}}$ to **25**.

## 6. Preventative Measures
To ensure scaling never kills the pipeline silently again, we added `tests/test_scaling_robustness.py`:
*   **Multi-Entity Consistency Test:** Verifies that a peaceful and a violent country can both be scaled and restored correctly in the same batch.
*   **Global Fit Guard:** Programmatically checks that models in the `Catalog` are using `global_fit=True`.
*   **Leakage Prevention Test:** Ensures that the act of "transforming" data never updates the internal state of the scaler (transductive leak check).

## 7. Lessons Learned
*   **Trust the Baseline:** If a model drops 4x below a simple moving average baseline, it is almost certainly a bug in the data/scaling math, not a change in model "honesty."
*   **Scaling is Part of the Architecture:** For global models, the scaler is not just a preprocessing step; it is a weight-sharing constraint.
*   **Beware of `[0]`:** Manual indexing of fitted parameters in a multi-entity environment is an anti-pattern that leads to "First-Entity Traps."
