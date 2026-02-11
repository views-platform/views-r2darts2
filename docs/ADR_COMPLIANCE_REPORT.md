# Codebase ADR Compliance and Upgrade Report

**Date:** 2026-02-11  
**Status:** Pending Execution  
**Context:** This report evaluates the current `views-r2darts2` codebase against the newly established Architectural Decision Records (ADRs 000-009).

---

## 1. Governance & Documentation (ADR-006)

### Issue: Missing Explicit Intent Contracts
**Observation:** Core classes have descriptive docstrings, but they do not strictly follow the "Purpose, Non-Goals, Guarantees, Failure Behavior" structure mandated by ADR-006.
- **Affected Classes:**
  - `ReproducibilityGate`
  - `ModelCatalog`
  - `DartsForecaster`
  - `DartsForecastingModelManager`
  - `_ViewsDatasetDarts`

**Upgrade Plan:**
- **How:** Refactor class-level docstrings to include the four mandatory sections. Ensure the "Failure Behavior" section explicitly references the exceptions raised.

---

## 2. Ontology & Redundancy (ADR-001)

### Issue: Redundant `ChainedScaler`
**Observation:** `utils/scaling.py` contains a custom `ChainedScaler` class. However, `FeatureScalerManager` and `DartsForecaster` have been upgraded to use Darts' native `Pipeline` object for chaining, which better handles probabilistic sample dimensions.
- **Affected File:** `views_r2darts2/utils/scaling.py`

**Upgrade Plan:**
- **How:** Remove the `ChainedScaler` class and its associated tests in `tests/test_scaling.py`. Verify that all chaining logic now flows through the `_instantiate_scaler` methods using Darts `Pipeline`.

---

## 3. Observability & Fail-Loud (ADR-008)

### Issue: Implicit Device State in Prediction
**Observation:** The recent fix for the GPU race condition (forcing `max_workers=1` on GPU) solved the crash, but the system still doesn't "Fail-Loud" if a model is accidentally moved to the CPU during a forward pass.
- **Affected File:** `views_r2darts2/model/forecaster.py`

**Upgrade Plan:**
- **How:** Add a check inside `DartsForecaster.predict` (or the underlying `_process_predictions`) that verifies the model's device matches the expected device. If a mismatch is detected, log a `CRITICAL` error and raise a `RuntimeError`.

---

## 4. Boundary Contracts (ADR-009)

### Issue: Loose Handshake in Data Handlers
**Observation:** `_ViewsDatasetDarts` performs some validation, but it doesn't explicitly audit the "Handshake" from the raw VIEWS dataframe as rigorously as the DNA manifest is audited.
- **Affected File:** `views_r2darts2/data/handlers.py`

**Upgrade Plan:**
- **How:** Implement a `ReproducibilityGate.Data.audit_dataframe_schema` method and call it during the initialization of `_ViewsDatasetDarts`. This ensures the dataframe contains the required time/entity indices before any Darts transformation occurs.

---

## 5. Summary of Mechanical Cleanups

- **Standardization:** audit `ModelCatalog` for any remaining singular `layer_width` or similar naming inconsistencies.
- **Dead Code:** Remove `repro_device_issue.py` and other temporary debug scripts.
- **Tests:** Ensure that removing `ChainedScaler` doesn't leave gaps in coverage for the `Pipeline`-based chaining.

---

## Conclusion

The codebase is structurally sound and follows the intended topology. The primary work required is **formalizing intent** via documentation and **hardening boundaries** via explicit validation gates. Executing these upgrades will bring the repository into full compliance with the "Fortress" mindset.
