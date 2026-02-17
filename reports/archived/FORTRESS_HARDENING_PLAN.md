# Fortress Hardening Plan: Post-CIC Refinement

**Date:** 2026-02-11  
**Status:** PROPOSED  
**Objective:** Resolve the 5 critical and minor architectural shortcomings revealed during Class Intent Contract (CIC) creation.

---

## Phase 1: Eradicating "The Silent Lie" (CRITICAL)
**Target:** `views_r2darts2/model/forecaster.py` -> `_process_predictions`  
**The Problem:** `np.nan_to_num` masks numerical instability by converting NaNs to 0.0.

### Step-by-Step:
1.  **Remove** the `np.nan_to_num` call from `_process_predictions`.
2.  **Implement** a numerical audit using `ReproducibilityGate.Data.audit_numerical_sanity(pred_values)`.
3.  **Raise** `NumericalSanityError` if NaNs or Infs are detected post-inference.
4.  **Verification:** Create a Red Team test that mocks a model returning NaNs and verify the run halts loudly.

---

## Phase 2: Sealing the Data Airlock
**Target:** `views_r2darts2/data/handlers.py` -> `as_darts_timeseries`  
**The Problem:** No explicit enforcement of `float32` precision at the entry boundary.

### Step-by-Step:
1.  **Modify** `as_darts_timeseries` to apply `.astype(np.float32)` to the dataframe *before* it is converted to a `darts.TimeSeries` object.
2.  **Refactor** `ReproducibilityGate.Data.audit_dataframe_schema` to also verify the dtypes of the incoming dataframe levels and columns.
3.  **Verification:** Test with a `float64` source dataframe and assert the resulting Darts object is `float32`.

---

## Phase 3: Scaling Integrity Enforcement
**Target:** `views_r2darts2/utils/scaling.py` -> `FeatureScalerManager`  
**The Problem:** `global_fit` is implicit, relying on factory defaults rather than Fortress law.

### Step-by-Step:
1.  **Modify** `_instantiate_scaler` in `FeatureScalerManager` to explicitly pass `global_fit=True` to the `darts.dataprocessing.transformers.Scaler` constructor.
2.  **Apply** the same explicit enforcement to `DartsForecaster._instantiate_scaler` for target scalers.
3.  **Verification:** Run `tests/test_scaling_robustness.py`.

---

## Phase 4: Structural DRYness (Catalog Refactor)
**Target:** `views_r2darts2/model/catalog.py`  
**The Problem:** Duplicated logic for mapping chunk lengths and common training parameters.

### Step-by-Step:
1.  **Implement** a private helper `_get_common_model_args(self)` that returns a dictionary containing:
    - `input_chunk_length`, `output_chunk_length`, `output_chunk_shift`
    - `batch_size`, `n_epochs`, `random_state`
    - `loss_fn`, `pl_trainer_kwargs`, `optimizer_cls`, `optimizer_kwargs`
    - `lr_scheduler_cls`, `lr_scheduler_kwargs`
2.  **Update** all 10 model getter methods to use `**self._get_common_model_args()`, overriding only what is specific to the architecture.
3.  **Verification:** Run `tests/test_catalog.py` to ensure bit-identical model instantiation.

---

## Phase 5: Pure Orchestration (Manager Cleanup)
**Target:** `views_r2darts2/manager/model.py`  
**The Problem:** Manager is performing direct file I/O, coupling it to the dataframe storage format.

### Step-by-Step:
1.  **Refactor** the dataframe loading logic into a static factory method `_ViewsDatasetDarts.from_views_path(path, dna)`.
2.  **Update** the Manager to simply call this factory, passing the path provided by the `ModelPathManager`.
3.  **Verification:** Verify that `ForecastingModelManager` (parent) remains unaware of the internal data format.

---

## Final Verification
1.  Run `ruff check . --fix`.
2.  Execute full test suite (382 tests).
3.  Ensure code is 100% compliant with the established CICs.
