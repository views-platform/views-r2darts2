# ADR-015: Artifact-Prediction Timestamp Contract

**Status:** Accepted
**Date:** 2026-05-19
**Deciders:** Simon Polichinel von der Maase
**Consulted:** views-pipeline-core ADR-052 (central contract)

---

## Context

The VIEWS pipeline requires that prediction filenames carry the timestamp of the **trained model artifact** that produced them, not the wall-clock time of prediction generation (views-pipeline-core ADR-013, ADR-052). This contract enables:

1. **Traceability:** Any prediction file can be linked back to the exact artifact that produced it.
2. **Ensemble resolution:** The ensemble manager matches constituent predictions to artifacts by timestamp. A mismatch causes silent fallback to subprocess re-execution or failure.

A violation of this contract was discovered in views-baseline in May 2026, where `datetime.now()` overwrote the artifact timestamp. An audit of all model-specific repos confirmed that views-r2darts2 implements the contract correctly via `DartsForecastingModelManager`.

This ADR documents the existing correct implementation for the record and to ensure it is preserved during future refactors.

---

## Decision

`DartsForecastingModelManager` (in `views_r2darts2/engines/darts_forecasting_model_manager.py`) correctly implements the artifact-prediction timestamp contract:

- `_evaluate_model_artifact()` resolves the latest artifact path, extracts the 15-character timestamp from the filename stem into a local variable, and persists it via `self._config_manager.add_config({"timestamp": timestamp})`.
- `_forecast_model_artifact()` follows the same pattern.

No code changes are required. This ADR is documentation-only.

---

## Rationale

- **Defensive documentation.** The contract is subtle (the `config` property trap makes the wrong approach silently fail). Documenting the correct implementation guards against regression during refactors.
- **Cross-repo consistency.** All model-specific repos now have a local ADR referencing the central contract in views-pipeline-core ADR-052.

---

## Considered Alternatives

Not applicable — the implementation is already correct. This ADR documents existing behavior.

---

## Consequences

### Positive

- The timestamp contract is documented in this repo, reducing the risk of accidental regression.
- Future contributors can reference this ADR when modifying artifact loading or prediction generation.

### Negative

- None.

---

## Implementation Notes

- **No code changes required.**
- **Actual pattern** (in `_evaluate_model_artifact()` and `_forecast_model_artifact()`):
  ```python
  timestamp = path_artifact.stem[-15:]
  self._config_manager.add_config({"timestamp": timestamp})
  ```
- **Key invariant:** No method in the evaluate/forecast flow should overwrite the timestamp with a value other than the one extracted from the artifact filename.

---

## Validation & Monitoring

- **Failure signal:** Ensemble evaluation failing with "prediction file not found" for r2darts2 constituents would indicate a regression.

---

## Open Questions

- None.

---

## References

- views-pipeline-core ADR-052: Artifact-Prediction Timestamp Contract (central)
- views-pipeline-core ADR-013: Prediction Naming Convention
- views-hydranet ADR-026: Model Artifact Fetcher Specification
- views-baseline ADR-016: Artifact-Prediction Timestamp Contract (bugfix record)
