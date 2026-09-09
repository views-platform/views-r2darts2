# ADR-012: Scaling Pipeline and Calibration Integrity

**Status:** Accepted  
**Date:** 2026-02-11  
**Deciders:** Simon Polichinel von der Maase  

---

## Context

Incorrect scaler management is a leading cause of calibration collapse in probabilistic forecasting. Early versions of this repository used custom `ChainedScaler` objects that failed to correctly broadcast sample dimensions during inverse transforms. This resulted in models that appeared to have zero uncertainty.

Additionally, using local fitting (where each time series is scaled relative to itself) can mask important cross-sectional signals in conflict data.

---

## Decision

1.  **Standardized Pipeline:** All transformations (chained or single) must use the **Darts native `Pipeline`**. Custom scaling wrappers are forbidden.
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
- Simplified logic in `DartsForecaster`.
- Improved cross-entity learning for models like N-BEATS.

### Negative
- Models may be more sensitive to extreme outliers in the training set (since they share a global scale).

---

## Implementation Notes

- **Enforcement:** `DartsForecaster._instantiate_scaler` must wrap all configs in `darts.dataprocessing.Pipeline`.
- **Constraint:** Target scalers must never be instantiated with `global_fit=False`.

---

## Validation & Monitoring

- **Tests:** `tests/test_scaling_robustness.py` verifies that `global_fit` is active and that sample dimensions are preserved.
- **Audit:** Any PR introducing a new scaler must include a Green Team test showing valid probabilistic ranges after transformation.
