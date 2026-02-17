# Class Intent Contract: DartsForecastingModelManager

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-02-11  
**Related ADRs:** ADR-001, ADR-002, ADR-003, ADR-009, ADR-011  

---

## 1. Purpose

The `DartsForecastingModelManager` is the high-level orchestrator for the forecasting lifecycle. Its primary purpose is to coordinate the transition of an experiment through its four stages: **Handshake**, **Training**, **Evaluation**, and **Forecasting**.

> **It acts as the single point of entry for experiment execution, ensuring that orchestration logic never leaks into the mathematical or data-handling layers.**

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** implement model architectures or loss functions (delegated to `ModelCatalog`).
- This class does **not** manage low-level data transformations or scaling (delegated to `DartsForecaster`).
- This class does **not** contain scientific business logic or heuristic "clipping" (ADR-010).
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
- **Data Availability:** Assumes raw VIEWS dataframes exist at the paths provided by the Path Manager.

---

## 5. Outputs and Side Effects

- **Artifacts:** Produces persistent `.pt` model artifacts containing weights and coupled scaler states.
- **Predictions:** Produces lists of `pd.DataFrame` results for evaluation or forecasting.
- **Logging:** Emits structured logs via `WandbLogger` and standard logging for lifecycle events.
- **Monkeypatching:** Performs a controlled override of `torch.load` to handle Darts serialization requirements.

---

## 6. Failure Modes and Loudness

- **Configuration Gap:** Raises `MissingHyperparameterError` if the DNA is incomplete.
- **Temporal Gap:** Raises `TemporalDiscontinuityError` if the test set is not contiguous with training.
- **Hardware Drift:** Raises `RuntimeError` if GPU restoration fails during prediction.
- **Horizon Violation:** Raises `PredictionHorizonError` if a forecast is attempted beyond ground truth.

---

## 7. Boundaries and Interactions

- **Upstream:** Interacts with `views_pipeline_core` for configuration and lifecycle control.
- **Physical Zen:** Lives in `views_r2darts2/manager/darts_forecasting_model_manager.py`.
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

- **Red Team:** `tests/test_reproducibility_infra.py` (Temporal injections, DNA poisoning).
- **Green Team:** `tests/test_model.py` (Lifecycle verification).
- **Beige Team:** Verified via `ModelCatalog` integration tests.

---

## 11. Evolution Notes

### Known Deviations / Technical Debt
- **Direct Data Loading:** Currently uses `read_dataframe` directly. This should eventually be delegated to a dedicated data-orchestration utility to keep the manager purely about lifecycle.
- **Fixed Seq Numbers:** Some methods still hardcode `total_sequence_number = 12`. This should be moved to the DNA manifest.

---

## End of Contract

This document defines the **intended meaning** of `DartsForecastingModelManager`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
