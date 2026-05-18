# ADR-010: Numerical Precision and Semantic Thresholds

**Status:** Accepted  
**Date:** 2026-02-11  
**Deciders:** Simon Polichinel von der Maase  

---

## Context

In conflict forecasting, small numerical discrepancies can lead to divergent results across different hardware environments. Mixed precision (using `float64` for targets and `float32` for features) often causes subtle stochastic drift during training and inference. 

Additionally, models trained on count data often produce "noisy" continuous predictions (e.g., `0.23` instead of `0.0`). There is a temptation to "clean" these outputs with hardcoded semantic thresholds (e.g., "anything less than 1.0 is 0.0"). Such hardcoding creates "silent lies" where the model's actual performance is masked by a hidden heuristic, violating the principle of authority (ADR-003).

---

## Decision

1.  **Universal Precision:** All data tensors (targets and covariates) must be explicitly downcast to **`float32`** before entering any model or loss function. 
2.  **Raw Output Mandate:** Models must return their "raw" expected values. 
3.  **Prohibition of Semantic Floors:** No hardcoded thresholding logic (clipping to 0 or rounding to integers) is permitted within the `model/` or `data/` layers. 
4.  **Separation of Metrics:** Semantic definitions of "Zero" belong in the **Evaluation Layer** (during metric calculation) or must be declared as a **gene** in the DNA manifest (e.g., `prediction_floor: 1.0`).

---

## Rationale

- **Reproducibility:** `float32` is the standard for deep learning on GPUs. Enforcing it globally eliminates precision-based drift between CPU and GPU paths.
- **Intentionality:** If we tell the world our model predicted `0.23`, and we evaluate it as `0.0`, the evaluation metrics must account for that choice explicitly. Hardcoding it in the pipeline makes the system's behavior opaque.
- **Correctness:** Count data models predict *expected values*, which are naturally continuous. Rounding or flooring them prematurely destroys the information needed for nuanced evaluation.

---

## Consequences

### Positive
- Predictable behavior across different machines and libraries.
- Clearer debugging: what the model says is what you see in the results.
- Robustness: The system won't break if we switch to non-count targets (like probabilities).

### Negative
- Predictions may look "messy" (non-integers) until processed by evaluation scripts.
- Minor increase in memory usage when targets are small but stored as `float32`.

---

## Implementation Notes

- **Enforcement:** `DartsForecaster._process_predictions` must use a minimal numerical epsilon (`1e-8`) only for mathematical stability, not for semantic filtering.
- **DNA Extension:** If a floor is scientifically required, add `prediction_floor` to the DNA Genome for the specific model.

---

## Validation & Monitoring

- **Tests:** `tests/test_scaling.py` verifies that transforms maintain ~1e-4 precision.
- **Audits:** `ReproducibilityGate` verifies that input data is not `float64`.
