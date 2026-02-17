# N-BEATS Reproducibility Investigation: Fault Line Registry

This document tracks potential causes for the divergence between **Sweep** and **Single-Run** execution paths in the `views-r2darts2` repository.

## 🟥 Config Object Inconsistency (Physical vs. Logical Identity)
*   **Issue:** Divergence between `self.config` and `self.configs`.
*   **Status:** **CONFIRMED** (Structural Fault)
*   **Mechanism:** Confirmed by dev as "legacy leftover." Both properties call `ConfigurationManager.get_combined_config()`, which returns a **brand new dictionary instance** on every access.
*   **Impact:** 
    1.  **Mutation Loss:** Any key added to the config during `_train_model_artifact` (which uses `self.config`) is physically lost when `_evaluate_model_artifact` pulls a fresh instance from `self.configs`.
    2.  **Fragility:** The code uses `self.config` and `self.configs` interchangeably, making it impossible to rely on object identity for any config-derived state.
*   **Verification:** `audit_identity.py` proved `id(self.config) != id(self.configs)`.

## 🟧 The "Reloading Gap" (In-Memory vs. Disk)
*   **Issue:** Models in sweeps are kept in-memory; single-runs are saved and reloaded.
*   **Location:** `views_r2darts2/manager/model.py`, `_evaluate_sweep` vs `_evaluate_model_artifact`.
*   **Mechanism:** `DartsForecaster.load_model` restores weights but uses Darts' `load()` which might not perfectly restore `pl_trainer_kwargs` or other non-state-dict properties.
*   **Impact:** 
    1.  **RNG State:** In-memory models carry the RNG state forward from training. Reloaded models start from a fresh framework state.
    2.  **Callback Persistence:** Training callbacks like `NaNDetectionCallback` remain attached to the model in sweeps, but are lost/re-initialized in single-runs.
*   **Status:** [ ] Open

## 🟨 Missing Prediction Hyperparameters in Sweeps
*   **Issue:** `_evaluate_sweep` fails to pass critical hyperparameters to `predict()`.
*   **Location:** `views_r2darts2/manager/model.py`, line 354.
*   **Status:** **CRITICAL FAULT IDENTIFIED**
*   **Mechanism:** `_evaluate_model_artifact` passes `num_samples`, `n_jobs`, and `mc_dropout` to `forecaster.predict()`. However, `_evaluate_sweep` calls `model.predict()` with **only** `sequence_number` and `output_length`.
*   **Why it causes divergence:** If the configuration specifies `mc_dropout: True` or `num_samples > 1`, the single-run will perform a stochastic/probabilistic prediction, while the sweep iteration will perform a **default deterministic prediction** (1 sample, no dropout). This results in systematic divergence in predicted values and metrics.
*   **Verification:** Check if `num_samples` or `mc_dropout` are non-default in the diverging configurations.

## 🟨 Global Side-Effects in `get_device`
*   **Issue:** Hidden modification of global PyTorch precision.
*   **Location:** `views_r2darts2/model/forecaster.py`, Line 311.
*   **Mechanism:** `torch.set_default_dtype(torch.float32)` is called if MPS is available.
*   **Why it causes divergence:** If this is called *inside* a sweep iteration worker but not in the main single-run process before initialization, weight initialization precision could differ.
*   **Status:** [ ] Open
*   **Verification:** Print `torch.get_default_dtype()` at the start of `ModelCatalog.get_model` in both paths.

## 🟩 Config Mutation during Evaluation
*   **Issue:** Injection of state into the configuration object.
*   **Location:** `views_r2darts2/manager/model.py`, Line 183.
*   **Mechanism:** `self._config_manager.add_config({"timestamp": ...})`
*   **Why it causes divergence:** `timestamp` is used to index artifacts. If evaluation in a single run uses a model-specific timestamp while a sweep uses a run-specific one, any artifact-naming logic will diverge.
*   **Status:** [ ] Open
*   **Verification:** Check for the existence of a `timestamp` key in `self.config` *before* training starts in a sweep iteration.

## 🟦 Derived Hyperparameter Logic
*   **Issue:** Calculation of `output_chunk_length` vs Forecast Horizon.
*   **Location:** `views_r2darts2/model/catalog.py`, Line 348.
*   **Mechanism:** `output_chunk_length = len(self.config["steps"])`.
*   **Why it causes divergence:** If `steps` is passed as a list in one path but a range in another, or if `len(steps)` (model chunk) is different from `max(steps)` (predict horizon), N-BEATS switches from direct to auto-regressive prediction.
*   **Status:** [ ] Open
*   **Verification:** Log `type(self.config["steps"])`, `len(steps)`, and `max(steps)` in both paths.

## 🟫 Data Loader Initialization Timing
*   **Issue:** `ViewsDataLoader` is initialized with stale config.
*   **Location:** `views_pipeline_core/managers/model/model.py`, `__init__`.
*   **Mechanism:** `self._data_loader` is created using `self._config_hyperparameters` *before* `update_for_sweep_run` is called.
*   **Why it causes divergence:** In a sweep, `steps` might be overridden. The model will be created with the new `steps`, but the `DataLoader` (which was created with the default `steps`) might have already calculated its internal partition ranges (especially for `forecasting` run type).
*   **Status:** [ ] Open
*   **Verification:** In a sweep iteration, check if `self._data_loader.steps` matches `len(self.configs["steps"])`.
