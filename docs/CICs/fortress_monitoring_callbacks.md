# Class Intent Contract: GradientHealthCallback

**Status:** Active
**Owner:** Core Engineering
**Last reviewed:** 2026-09-10
**Related ADRs:** ADR-003, ADR-008

---

## 1. Purpose

The `GradientHealthCallback` provides the **Observability Layer** for model training. Its primary purpose is to detect numerical decay in gradients before it compromises the scientific integrity of an experiment.

> **It acts as the "Internal Sensor" of the Fortress, ensuring that gradient failure is never silent.**

*Scope note (2026-09-10):* `views_r2darts2/infrastructure/callbacks.py` defines fifteen callbacks plus one private patch helper (`_PatchedTrainingStep`); this contract covers `GradientHealthCallback` in detail and names the others. Eleven callbacks carry their own `Intent Contract:` docstring in code. `NaNDetectionCallback` (halts after `patience` consecutive NaN losses) and `GradientHealthCallback` are the two fail-loud kill-switches; both are attached to every trainer by `ModelCatalog._get_common_pl_trainer_kwargs`, alongside `TrainingStepPatchCallback` (must be first), `WeightNormCallback`, `RevINMonitorCallback`, `PredictionSanityCallback`, `LossStabilityCallback`, `EpochTimingCallback`, `YHatBarCallback`, `ValMetricsCallback`, `InputBatchMonitorCallback`, `LossComponentCallback`, `RichLossDiagnosticsCallback`, and `LossGradientDiagnosticsCallbackV2`.

---

## 2. Non-Goals (Explicit Exclusions)

- This callback does **not** modify model weights or gradients (it is a read-only observer).
- This callback does **not** calculate metrics or loss values (it consumes existing outputs).
- This callback does **not** manage device placement.
- This callback does **not** log to external databases directly (it uses the configured `logging` and `logger` interfaces).

---

## 3. Responsibilities and Guarantees

### GradientHealthCallback
- **Guarantees Per-Step Snapshot, Per-Epoch Verdict:** Snapshots gradient statistics in `on_before_optimizer_step` (gated by `log_every_n_epochs`); the verdict and the halt live in `on_train_epoch_end`. `trainer.should_stop = True` fires **only on NaN/Inf gradients** (`callbacks.py:271-274`); an exploding norm changes the status string (`:245`) and does not halt.
- **Detects Vanishing/Exploding Gradients:** Provides high-visibility status messages (`✅ healthy` vs `🚨 exploding`) based on configurable thresholds.
- **Exposes Sparsity:** Reports the ratio of zero gradients, identifying potentially "dead" neurons or bottlenecks.

---

## 4. Inputs and Assumptions

- **Trainer State:** Assumes access to the `pytorch_lightning.Trainer` and `LightningModule` objects.
- **Gradients:** Assumes that `param.grad` is populated (i.e., called during or after the backward pass).

---

## 5. Outputs and Side Effects

- **Logs:** Emits structured status updates to the standard logging stream.
- **Side Effects:** Halts training on NaN/Inf gradients; pushes scalar metrics to the PL logger (`:258`). Does not touch weights or gradients.

---

## 6. Failure Modes and Loudness

- **Silent Success:** If gradients are healthy, it logs an `INFO` message (unless logging frequency is reduced).
- **Fail-Loud Mandate (aspirational):** If this callback cannot access gradients, it should fail loudly. *It does not:* with no populated gradients it returns early (`:231`) or reports a zero-statistics "vanishing" verdict at `INFO`. Register C-47.

---

## 7. Boundaries and Interactions

- **Upstream:** Orchestrated by `ModelCatalog` which attaches it to the Trainer.
- **Physical Zen:** Lives in `views_r2darts2/infrastructure/callbacks.py`.
- **Downstream:** Closely monitors the model weights and training loop outputs.
- **Abstractions:** Treats the Darts/Lightning training loop as an opaque source of tensors.

---

## 8. Examples of Correct Usage

```python
# Typically added automatically by ModelCatalog
callbacks = [
    GradientHealthCallback(log_every_n_epochs=1)
]
trainer = pl.Trainer(callbacks=callbacks)
```

---

## 9. Examples of Incorrect Usage

- **Weight Mutation:** Attempting to manually clip gradients inside the `GradientHealthCallback` (violates the "Read-Only Observer" goal).
- **Metric Dependency:** Relying on validation metrics to trigger a halt (this callback focuses on raw numerical health, not predictive performance).

---

## 10. Test Alignment

- **None.** No test under `tests/` references `callbacks`, `NaNDetectionCallback`, or `GradientHealthCallback` — the `should_stop` kill-switches have zero coverage (C-13). `tests/test_model_catalog.py` verifies only that they are attached.

---

## 11. Evolution Notes

### Known Deviations / Technical Debt
- **Shared Thresholds:** The exploding-gradient threshold (`explode_threshold`, default `500.0`) is a constructor default, not a DNA gene.
- **Zero test coverage** of the halt paths (C-13).

### Correction (2026-09-10)
An earlier revision of this contract stated that `NaNDetectionCallback` had been removed. That was false: it is defined at `callbacks.py:96` and attached to every trainer. The paragraph has been deleted.

---

## End of Contract

This document defines the **intended meaning** of `GradientHealthCallback`.
Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
