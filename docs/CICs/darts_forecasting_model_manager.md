# Class Intent Contract: DartsForecastingModelManager

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-09-10  
**Related ADRs:** ADR-001, ADR-002, ADR-003, ADR-009, ADR-011  

---

## 1. Purpose

The `DartsForecastingModelManager` is the high-level orchestrator for the forecasting lifecycle. Its primary purpose is to coordinate the transition of an experiment through its four stages: **Handshake**, **Training**, **Evaluation**, and **Forecasting**.

> **It acts as the single point of entry for experiment execution, ensuring that orchestration logic never leaks into the mathematical or data-handling layers.**

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** implement model architectures or loss functions (delegated to `ModelCatalog`).
- This class does **not** manage low-level data transformations or scaling (delegated to `ViewsDataset`).
- This class does **not** contain scientific business logic or thresholds (ADR-016).
- This class does **not** directly interface with databases or external APIs (except via `ModelPathManager`).

---

## 3. Responsibilities and Guarantees

- **Guarantees the Handshake:** Ensures `ReproducibilityGate.Config` is audited before any execution begins.
- **Orchestrates Lifecycle:** Manages the sequential flow from raw data reading to artifact persistence.
- **Enforces Temporal Integrity:** Guarantees that partitions are resolved and audited for continuity (t+1) before training starts.
- **Manages Hardware Safety:** Explicitly controls concurrency (e.g., forcing sequential GPU prediction) to prevent framework-level race conditions (ADR-011).
- **Snapshot Integrity:** Captures immutable snapshots of the configuration to prevent mutation during long-running jobs.

---

## 4. Inputs and Assumptions

- **Config Snapshot:** Assumes a merged configuration dictionary containing both `CORE` and `ALGORITHM` genomes.
- **Path Manager:** Requires a `ModelPathManager` to resolve standardized directory structures.
- **Data Availability:** Assumes a raw parquet (or cached frame) exists at the path resolved by `_resolve_raw_parquet_path` from the Path Manager.
- **Parent Class:** Requires `views_pipeline_core` (the optional `manager` extra). Without it the module still imports — the class inherits from `object` — but `__init__` raises `ImportError`.

---

## 5. Outputs and Side Effects

- **Artifacts:** Produces persistent `.pt` model artifacts containing weights and coupled scaler states.
- **Predictions:** Produces `dict[str, PredictionFrame]` per sequence (evaluation) or per run (forecast); DataFrame conversion only on demand via `_predictions_to_dataframe`.
- **Logging:** Emits structured logs via `WandbLogger` and standard logging for lifecycle events.
- **Monkeypatching:** Performs a controlled override of `torch.load` to handle Darts serialization requirements.

---

## 6. Failure Modes and Loudness

- **Configuration Gap:** Raises `MissingHyperparameterError` if the DNA is incomplete.
- **Temporal Gap:** Raises `TemporalDiscontinuityError` if the test set is not contiguous with training.
- **Hardware Drift:** *Does not raise* — the forecaster warns and continues on CPU (register D-01).
- **Horizon Violation:** Raises `PredictionHorizonError` if a forecast is attempted beyond ground truth.

---

## 7. Boundaries and Interactions

- **Upstream:** Interacts with `views_pipeline_core` for configuration and lifecycle control.
- **Physical Zen:** Lives in `views_r2darts2/engines/darts_forecasting_model_manager.py`.
- **Downstream:** Depends on `ModelCatalog` (for instantiation) and `DartsForecaster` (for execution).
- **Airlock:** Interacts with `ReproducibilityGate` to validate all boundaries.

---

## 8. Examples of Correct Usage

```python
# Standard training workflow
manager = DartsForecastingModelManager(model_path=my_paths)
manager._train_model_artifact()

# Standard evaluation workflow
predictions = manager._evaluate_model_artifact(eval_type="standard")
```

---

## 9. Examples of Incorrect Usage

- **Direct Tensor Manipulation:** Accessing `forecaster.model.model` to manually zero weights.
- **Bypassing the Gate:** Instantiating a model without calling `ReproducibilityGate.Config.audit_manifest`.
- **Mixing Tiers:** Adding a new loss function directly into a manager method.

---

## 10. Test Alignment

- **Green Team:** `tests/test_darts_forecasting_model_manager.py` (`_resolve_total_sequence_number` boundary and failure cases, lazy-import path, lifecycle wiring).
- **Red Team:** `tests/test_reproducibility_gate.py` (the gates this class invokes at the handshake).
- **Beige Team:** `tests/test_model_catalog.py` (catalog integration).
- **Not covered:** no test executes a real `model.fit()` — CI runs on a host without CUDA and `"accelerator": "gpu"` is hardcoded (C-21).

---

## 11. Evolution Notes

### Known Deviations / Technical Debt
- **Timestamp by fixed slice:** `path_artifact.stem[-15:]` with no validation (C-29; ADR-015 governs the contract).
- **`parallel_workers > 1` races on the global RNG** that `lock_entropy` reseeds (C-24).
- **`torch.load` is patched process-wide** on construction via `apply_all_patches()` (C-28).

### Resolved in 0.2.x
- Sequence count is derived by `_resolve_total_sequence_number(partition, max_steps)` with a fail-loud guard (C-01/C-02).
- Data loading goes through `ViewsDataset`, not a direct `read_dataframe` call.

---

## End of Contract

This document defines the **intended meaning** of `DartsForecastingModelManager`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
