# ADR-012: Scaling Pipeline and Calibration Integrity

**Status:** Accepted  
**Date:** 2026-02-11  
**Deciders:** Simon Polichinel von der Maase  
**Revised:** 2026-09-10 — re-derived against the 0.2.x codebase (`development` @ `fe7e681`). Original decision unchanged unless stated.  

---

## Context

Incorrect scaler management is a leading cause of calibration collapse in probabilistic forecasting. Early versions of this repository used custom `ChainedScaler` objects that failed to correctly broadcast sample dimensions during inverse transforms. This resulted in models that appeared to have zero uncertainty.

Additionally, using local fitting (where each time series is scaled relative to itself) can mask important cross-sectional signals in conflict data.

---

## Decision

1.  **Standardized Pipeline:** All transformations (chained or single) must use the **Darts native `Pipeline`** or `Scaler`. Custom scaling wrappers are forbidden.
    *Compliance note (2026-09-10):* `views_r2darts2/transformers/inverse.py` is a bespoke inverse path that reaches into Darts' private `_fitted_params` / `_fit_called` to preserve the sample dimension, and `ViewsDataset._inverse_transform_numpy_predictions` bypasses `Pipeline.inverse_transform`. Whether these are a sanctioned exception or a violation is register **D-03**. Until ruled, they are the de facto implementation of decision 3 below.
2.  **Global Scaling Mandate:** All target scalers must use **`global_fit=True`**. This ensures the scaler learns the distribution across all countries/entities, preventing signal loss.
3.  **Dimension Preservation:** Target scalers must be applied such that the **sample dimension** (the third axis of the Darts tensor) is preserved and correctly transformed during `inverse_transform`.

---

## Rationale

- **Calibration:** Darts' `Pipeline` and `Scaler` objects are designed to handle the 3D nature of probabilistic forecasts. Standardizing on them reduces the risk of "zero-width" intervals.
- **Comparability:** Global fitting ensures that a count of "10" in Country A means the same thing as a count of "10" in Country B after scaling. Local fitting destroys this semantic link.
- **Maintainability:** Using native framework objects reduces the amount of specialized "Fortress" code we need to maintain.

---

## Consequences

### Positive
- Reliable uncertainty estimation (better Brier scores and calibration curves).
- Simplified logic in `DartsForecaster` — scaler ownership moved to `ViewsDataset` in 0.2.x.
- Improved cross-entity learning for models like N-BEATS.

### Negative
- Models may be more sensitive to extreme outliers in the training set (since they share a global scale).

---

## Implementation Notes

- **Enforcement:** `ViewsDataset.fit_scalers` (`views_r2darts2/dataset/base.py`) constructs every scaler through `ScalerSelector.instantiate_darts_scaler` (`views_r2darts2/transformers/scaler_selector.py`), which returns a Darts `Scaler` for a single spec and a Darts `Pipeline` for a chain.
- **Constraint:** Target scalers must never be instantiated with `global_fit=False`. Every construction site in `scaler_selector.py` hardcodes `global_fit=True`; this is honoured on 0.2.x.

---

## Validation & Monitoring

- **Tests:** `tests/test_scaler_selector.py` covers construction and `global_fit`; `tests/test_feature_scaler_manager.py` and `tests/test_parity_e2e.py` cover sample-dimension preservation through the inverse path.
- **Audit:** Any PR introducing a new scaler must include a Green Team test showing valid probabilistic ranges after transformation.
