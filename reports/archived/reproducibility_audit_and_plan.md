# Reproducibility Audit Report & Resolution Plan

## 1. Audit Report: The State of the Fault

### 1.1 Configuration "Black Hole" (Property Divergence)
*   **Issue**: `self.config` and `self.configs` in `ModelManager` are non-stable properties. Every access triggers `ConfigurationManager.get_combined_config()`, which returns a **brand new dictionary instance**.
*   **Why**: Confirmed by original developers as "legacy leftover" logic.
*   **How Verified**: `audit_identity.py` proved `id(self.config) != id(self.configs)`. Mutations to `self.config` do not persist across lines of code or between train/eval phases.
*   **Impact**: Any state derived during training and stored in the config is lost before evaluation starts.

### 1.2 Stale Data Loader (Temporal Misalignment)
*   **Issue**: `ViewsDataLoader` uses a `steps` parameter that becomes stale during sweeps.
*   **Why**: The loader is initialized in the constructor using default values. When a sweep overrides `steps`, the `ModelCatalog` builds a model for the new horizon, but the `DataLoader` continues using the old horizon to calculate data partitions.
*   **How Verified**: `ruthless_audit.py` showed `loader.steps` remained at **3** while `configs['steps']` was overridden to **6**.
*   **Impact**: Sweeps train/evaluate on a different temporal window than single runs, leading to "hallucinated" performance metrics.

### 1.3 Partial Prediction Interface (Missing Kwargs)
*   **Issue**: The sweep evaluation path (`_evaluate_sweep`) fails to pass critical prediction hyperparameters.
*   **Why**: `_evaluate_model_artifact` passes `num_samples`, `n_jobs`, and `mc_dropout`, but `_evaluate_sweep` does not.
*   **How Verified**: Static analysis of `views_r2darts2/manager/model.py` Line 354.
*   **Impact**: Models revert to **deterministic default mode** in sweeps (ignoring dropout/sampling), causing systematic divergence from single runs where these parameters are correctly applied.

### 1.4 Implicit Architecture Coupling
*   **Issue**: Model architecture (specifically `output_chunk_length`) is implicitly derived from the `steps` list length.
*   **Why**: `output_chunk_length = len(self.config["steps"])` in `ModelCatalog`.
*   **How Verified**: Confirmed in `views_r2darts2/model/catalog.py` Line 348.
*   **Impact**: Changing the forecast horizon in a sweep silently changes the neural network's physical structure, making hyperparameter comparisons invalid across different `steps` counts.

---

## 2. Resolution Plan: The "Glass House" Protocol

### 2.1 Configuration Unification & Stability
*   **Action**: Refactor `DartsForecastingModelManager` to avoid internal use of `self.config`/`self.configs`.
*   **Method**: Capture a single `active_config` dictionary at the start of a run and pass it explicitly to all functional units.
*   **Constraint**: No mutations allowed. Derived values must be stored in explicit variables or a dedicated `run_context` object.

### 2.2 Explicit Data Loader Re-Sync
*   **Action**: Create an explicit `sync_data_loader(config)` method in the manager.
*   **Method**: Call this method immediately after `update_for_sweep_run` or `update_for_single_run`.
*   **Constraint**: If `steps` is missing from the config, raise a `KeyError`. Do not default to 36.

### 2.3 Unified Prediction Interface
*   **Action**: Implement `_get_predict_kwargs(config)` to centralize parameter extraction.
*   **Method**: Use this helper in both `_evaluate_sweep` and `_evaluate_model_artifact`.
*   **Constraint**: Raise `ValueError` if mandatory parameters (`num_samples`, `mc_dropout`) are missing. No framework-level defaults permitted.

### 2.4 Strict Architecture Contracts
*   **Action**: Require `output_chunk_length` as an explicit hyperparameter.
*   **Method**: Remove implicit derivation from `steps`. Add validation logic to ensure `len(steps)` is compatible with `output_chunk_length`.
*   **Constraint**: The model must fail to initialize if these are not explicitly defined and mathematically compatible.
