# Class Intent Contract: GradientHealthCallback

**Status:** Active
**Owner:** Core Engineering
**Last reviewed:** 2026-03-15
**Related ADRs:** ADR-003, ADR-008

---

## 1. Purpose

The `GradientHealthCallback` provides the **Observability Layer** for model training. Its primary purpose is to detect numerical decay in gradients before it compromises the scientific integrity of an experiment.

> **It acts as the "Internal Sensor" of the Fortress, ensuring that gradient failure is never silent.**

---

## 2. Non-Goals (Explicit Exclusions)

- This callback does **not** modify model weights or gradients (it is a read-only observer).
- This callback does **not** calculate metrics or loss values (it consumes existing outputs).
- This callback does **not** manage device placement.
- This callback does **not** log to external databases directly (it uses the configured `logging` and `logger` interfaces).

---

## 3. Responsibilities and Guarantees

### GradientHealthCallback
- **Guarantees Per-Epoch Auditing:** Audits the gradient norms of every trainable parameter at the end of each epoch.
- **Detects Vanishing/Exploding Gradients:** Provides high-visibility status messages (`✅ healthy` vs `🚨 exploding`) based on configurable thresholds.
- **Exposes Sparsity:** Reports the ratio of zero gradients, identifying potentially "dead" neurons or bottlenecks.

---

## 4. Inputs and Assumptions

- **Trainer State:** Assumes access to the `pytorch_lightning.Trainer` and `LightningModule` objects.
- **Gradients:** Assumes that `param.grad` is populated (i.e., called during or after the backward pass).

---

## 5. Outputs and Side Effects

- **Logs:** Emits structured status updates to the standard logging stream.
- **Side Effects:** None. This is designed to be non-intrusive.

---

## 6. Failure Modes and Loudness

- **Silent Success:** If gradients are healthy, it logs an `INFO` message (unless logging frequency is reduced).
- **Fail-Loud Mandate:** If this callback cannot access gradients due to framework changes, it must fail loudly rather than silently assuming everything is fine.

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

- **Red Team:** `tests/test_reproducibility_infra.py` (Numerical poisoning tests).
- **Green Team:** Verified via standard training integration tests in `tests/test_catalog.py`.

---

## 11. Evolution Notes

### Known Deviations / Technical Debt
- **Shared Thresholds:** Thresholds for "exploding" gradients (currently `100.0`) are hardcoded. These should ideally be moved to the DNA manifest to allow architecture-specific sensitivity tuning.

### Historical Changes
- **`NaNDetectionCallback` removed:** Previously defined alongside `GradientHealthCallback` but was never used in production. NaN detection in loss is handled by custom loss functions via `NumericalSanityError` guards.

---

## End of Contract

This document defines the **intended meaning** of `GradientHealthCallback`.
Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
