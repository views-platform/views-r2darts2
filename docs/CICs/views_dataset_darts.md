# Class Intent Contract: _ViewsDatasetDarts

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-02-11  
**Related ADRs:** ADR-001, ADR-002, ADR-009, ADR-010  

---

## 1. Purpose

The `_ViewsDatasetDarts` class acts as the **Data Airlock** for the repository. Its primary purpose is to transform generic, multi-indexed VIEWS pandas dataframes into the structured `TimeSeries` collections required by the Darts ecosystem.

> **It ensures that the high-dimensional data produced by the upstream pipeline is correctly mapped onto the temporal and spatial (entity) axes of the forecasting models.**

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** perform temporal slicing (delegated to `DartsForecaster`).
- This class does **not** perform scaling or log-transformations (delegated to `DartsForecaster`).
- This class does **not** handle file I/O or database queries.
- This class does **not** modify the values of the data (e.g., no interpolation or gap-filling).

---

## 3. Responsibilities and Guarantees

- **Guarantees Schema Integrity:** Immediately audits the incoming dataframe to ensure all targets and features declared in the DNA manifest physically exist (ADR-009).
- **Enforces Entity Mapping:** Guarantees that the data is correctly grouped by the entity index, preventing "Cross-Country Contamination" during tensor construction.
- **Ensures Multi-Index Preservation:** Guarantees that the semantic meaning of `month_id` and `country_id` is preserved during the transition to Darts `TimeSeries`.
- **Numerical Handshake:** Audits the dataframe for numerical sanity (no NaNs in raw input) before it reaches the models.

---

## 4. Inputs and Assumptions

- **Raw Dataframe:** Assumes a pandas MultiIndex DataFrame with levels matching the VIEWS standard (e.g., `month_id`, `country_id`).
- **DNA Components:** Assumes a list of `targets` and `features` provided by the experiment configuration.
- **Broadcast features:** Assumes `broadcast_features=True` if covariates are shared across targets.

---

## 5. Outputs and Side Effects

- **Darts TimeSeries:** Produces a collection of `darts.TimeSeries` objects, grouped by the second level of the MultiIndex (entity).
- **Side Effects:** None. This class is designed to be a pure transformation layer.

---

## 6. Failure Modes and Loudness

- **Missing Columns:** Raises `KeyError` if a required DNA column is missing from the source dataframe.
- **Invalid Index:** Raises `TypeError` or `ReproducibilityError` if the dataframe index does not comply with the `month_id`/`country_id` standard.
- **Numerical Insanity:** Raises `NumericalSanityError` (via Gate) if raw data contains prohibited NaNs or Infs.

---

## 7. Boundaries and Interactions

- **Upstream:** Consumes dataframes from `views_pipeline_core`.
- **Physical Zen:** Lives in `views_r2darts2/data/views_dataset_darts.py`.
- **Downstream:** Provides data to `DartsForecaster`.
- **Gatekeeper:** Invokes `ReproducibilityGate.Data` during initialization to verify the "Handshake" (ADR-009).

---

## 8. Examples of Correct Usage

```python
# Create airlock
dataset = _ViewsDatasetDarts(source=df, targets=["ged_sb"], features=["wdi_gdp"])

# Convert to Darts format
ts_collection = dataset.as_darts_timeseries()
```

---

## 9. Examples of Incorrect Usage

- **Manual Slicing:** Passing a pre-sliced dataframe to the constructor (violates the "Raw Source" principle).
- **Value Mutation:** Attempting to use this class to fill missing values before Darts conversion.

---

## 10. Test Alignment

- **Beige Team:** `tests/contract_verification/verify_prediction_schema.py` (Index alignment).
- **Red Team:** `tests/test_reproducibility_infra.py` (Data poisoning injection).

---

## 11. Evolution Notes

### Known Deviations / Technical Debt
- **Numerical Type Enforcement:** While ADR-010 mandates `float32`, the current implementation relies on the parent class's behavior. This should be refactored to explicitly downcast to `float32` within `as_darts_timeseries()` to ensure total compliance.

---

## End of Contract

This document defines the **intended meaning** of `_ViewsDatasetDarts`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
