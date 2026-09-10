# Class Intent Contract: DartsForecaster

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-09-10  
**Related ADRs:** ADR-001, ADR-002, ADR-006, ADR-008, ADR-011, ADR-012, ADR-016  

---

## 1. Purpose

The `DartsForecaster` is a slim orchestrator that couples one Darts model to one `ViewsDataset` and one partition. It trains, predicts, saves and loads; every data transformation is delegated to the dataset.

> **Its primary goal is to ensure that predictions come back on the original "raw" scale, by driving the dataset's fitted scalers in the right order rather than by owning them.**

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** manage Weights & Biases logging (delegated to the Manager).
- This class does **not** perform model architecture selection (delegated to the Catalog).
- This class does **not** handle high-level rolling-origin logic (delegated to the Manager).
- This class does **not** own scalers or log-transforms — `ViewsDataset` does; the forecaster calls `fit_scalers` and `get_scaled_darts_timeseries`.
- This class **does** clip predictions to non-negative on the streaming path — unconditionally, `darts_forecaster.py:515`, with no opt-out — while the ingest path's `clip_negatives=True` default lives on `ViewsDataset`. Both are register D-05 / ADR-016; this contract records the behaviour, not a decision.

---

## 3. Responsibilities and Guarantees

- **Guarantees Scaler Coupling:** The scalers fitted on the dataset during `train()` are the ones applied in inverse during `predict()`; `save_model`/`load_model` persist and restore them together with the weights, and `load_model` raises on a target-scaler config mismatch.
- **Enforces Numerical Precision:** All series reach the model as `float32` (ADR-016).
- **Preserves Probabilistic Calibration:** Target scalers are `global_fit=True`, enforced at construction by `ScalerSelector` (ADR-012).
- **Device Self-Healing:** `_ensure_model_on_device()` runs before every prediction and moves weights back to the resolved device if Darts drifted them to CPU (ADR-011). *On failure it warns and continues — see §6 and D-01.*
- **Entropy Lock:** Calls `ReproducibilityGate.Data.lock_entropy(random_state)` before every prediction so probabilistic samples are reproducible.
- **Non-negative Output:** Predictions are clipped to `>= 0` on ingest into the dataset (the code's own Intent Contract states this as a guarantee; see D-05).

---

## 4. Inputs and Assumptions

- **Dataset:** Assumes a `ViewsDataset` instance (Zarr-backed, with `targets`/`features` resolved and `validate_indices()` passed).
- **Model:** Assumes a `TorchForecastingModel` instance (already instantiated).
- **Partition:** Assumes a validated partition dictionary provided by the Manager.
- **Fitted State:** Assumes `predict()` will only be called after `train()` or `load_model()` has successfully fitted the scalers.

---

## 5. Outputs and Side Effects

- **Transformed Data:** Obtains scaled `TimeSeries` from `ViewsDataset.get_scaled_darts_timeseries`; builds the validation window itself (`_build_validation_set`).
- **Raw Predictions:** `predict()` returns `dict[str, views_frames.PredictionFrame]` — one per target, on the original data scale, sample dimension preserved. DataFrame conversion is on demand via `transformers/darts_bridge.py`.
- **State Mutation:** Mutates the dataset's scaler state through `fit_scalers`; holds the model and device.
- **Persistence:** Saves and loads comprehensive artifacts containing both model weights and scaler pipelines.

---

## 6. Failure Modes and Loudness

- **Unfitted Predict:** Raises `RuntimeError` if prediction is attempted without fitted scalers.
- **Device Failure:** *Does not raise.* If `_ensure_model_on_device` cannot move the model off CPU it logs a `WARNING` and continues on CPU. This contradicts ADR-008/ADR-011 and is register **D-01**; recorded here as the current behaviour, not the intended one.
- **Numerical Insanity:** Fails loudly (`NumericalSanityError`) if NaNs or Infs reach the ingest path.
- **Scaler Config Mismatch on Load:** Raises `ValueError` if the artifact recorded a non-null `target_scaler` config that differs from the current one (`:682`; an artifact that recorded `None` is not compared). The feature-scaler config is *not* checked (register C-09).

---

## 7. Boundaries and Interactions

- **Upstream:** Managed by `DartsForecastingModelManager`.
- **Physical Zen:** Lives in `views_r2darts2/engines/darts_forecaster.py`.
- **Downstream:** Drives `ViewsDataset` (which owns `FeatureScalerManager`/`ScalerSelector` internally), `transformers/frame_builder.py` for streaming prediction output, and one Darts `TorchForecastingModel`.
- **Validator:** Deeply coupled with `ReproducibilityGate` for boundary auditing.

---

## 8. Examples of Correct Usage

```python
# Instantiate and train
forecaster = DartsForecaster(dataset=ds, model=m, partition_dict=p, random_state=42)   # random_state is mandatory
forecaster.train()

# Predict; returns dict[target -> PredictionFrame] on the raw scale
frames = forecaster.predict(sequence_number=0)
```

---

## 9. Examples of Incorrect Usage

- **Manual Scaling:** Fitting a scaler outside the forecaster and then passing scaled data to `train()`.
- **Direct Model Fit:** Calling `forecaster.model.fit()` directly, bypassing the gate-protected `forecaster.train()` method.
- **Inferred Device:** Manually moving the model to a device without updating the `forecaster.device` property.

---

## 10. Test Alignment

- **Green Team:** `tests/test_darts_forecaster.py` (construction guards, `predict` at `sequence_number=0`, save/load round-trip). *No test calls `train()`; no rolling-origin sequence > 0 is exercised.*
- **Adjacent, not this class:** `tests/test_parity_e2e.py` exercises `ViewsDataset.ingest_darts_predictions` and the inverse path directly; it never constructs `DartsForecaster`.
- **Green Team:** `tests/test_streaming_predict_builder.py` (streaming prediction path).
- **Red Team:** `tests/test_reproducibility_gate.py` (the gates this class invokes).
- **Not covered:** device-restore failure (D-01), `parallel_workers > 1` reproducibility (C-24).

---

## 11. Evolution Notes

### Known Deviations / Technical Debt
- **Device failure is not fail-loud** — D-01.
- **Feature-scaler config not validated on load** — C-09.
- **`_predict_streaming` is 160 lines** — the largest method in the engine (C-19).

### Resolved in 0.2.x
- Scaler ownership, log transforms and prediction processing moved to `ViewsDataset`; the 0.1.x `_preprocess_timeseries` entity-alignment defect (C-20) no longer exists — targets and covariates are split once, in lockstep, by `ViewsDataset._split_targets_covariates`.

---

## End of Contract

This document defines the **intended meaning** of `DartsForecaster`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
