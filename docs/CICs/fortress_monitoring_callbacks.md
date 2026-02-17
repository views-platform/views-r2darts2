# Class Intent Contract: Fortress Monitoring Callbacks

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-02-11  
**Related ADRs:** ADR-003, ADR-008  

---

## 1. Purpose

The Monitoring Callbacks (`NaNDetectionCallback` and `GradientHealthCallback`) provide the **Observability Layer** for model training. Their primary purpose is to detect and halt numerical decay before it compromises the scientific integrity of an experiment.

> **They act as the "Internal Sensors" of the Fortress, ensuring that training failure is never silent.**

---

## 2. Non-Goals (Explicit Exclusions)

- These callbacks do **not** modify model weights or gradients (they are read-only observers).
- These callbacks do **not** calculate metrics or loss values (they consume existing outputs).
- These callbacks do **not** manage device placement.
- These callbacks do **not** log to external databases directly (they use the configured `logging` and `logger` interfaces).

---

## 3. Responsibilities and Guarantees

### NaNDetectionCallback
- **Guarantees Immediate Halt:** Ensures that training is terminated upon detection of persistent NaN values in the loss function.
- **Provides Diagnostic Feedback:** Guarantees that the error log contains actionable suggestions (e.g., "lower learning rate") when NaNs are detected.

### GradientHealthCallback
- **Guarantees Per-Epoch Auditing:** Audits the gradient norms of every trainable parameter at the end of each epoch.
- **Detects Vanishing/Exploding Gradients:** Provides high-visibility status messages (`✅ healthy` vs `🚨 exploding`) based on configurable thresholds.
- **Exposes Sparsity:** Reports the ratio of zero gradients, identifying potentially "dead" neurons or bottlenecks.

---

## 4. Inputs and Assumptions

- **Trainer State:** Assumes access to the `pytorch_lightning.Trainer` and `LightningModule` objects.
- **Outputs Dictionary:** Assumes that the training step returns a dictionary containing a `'loss'` key.
- **Gradients:** Assumes that `param.grad` is populated (i.e., called during or after the backward pass).

---

## 5. Outputs and Side Effects

- **Logs:** Emits structured status updates to the standard logging stream.
- **Control Signal:** `NaNDetectionCallback` sets `trainer.should_stop = True` to terminate execution.
- **Side Effects:** None. These are designed to be non-intrusive.

---

## 6. Failure Modes and Loudness

- **Silent Success:** If gradients are healthy, they log an `INFO` message (unless logging frequency is reduced).
- **Loud Failure:** If NaNs are detected, they emit an `ERROR` and a `CRITICAL` alert.
- **Fail-Loud Mandate:** If these callbacks cannot access the loss or gradients due to framework changes, they must fail loudly rather than silently assuming everything is fine.

---

## 7. Boundaries and Interactions

- **Upstream:** Orchestrated by `ModelCatalog` which attaches them to the Trainer.
- **Physical Zen:** Lives in `views_r2darts2/utils/callbacks.py`.
- **Downstream:** Closely monitors the model weights and training loop outputs.
- **Abstractions:** Treat the Darts/Lightning training loop as an opaque source of tensors.

---

## 8. Examples of Correct Usage

```python
# Typically added automatically by ModelCatalog
callbacks = [
    NaNDetectionCallback(patience=1),
    GradientHealthCallback(log_every_n_epochs=1)
]
trainer = pl.Trainer(callbacks=callbacks)
```

---

## 9. Examples of Incorrect Usage

- **Weight Mutation:** Attempting to manually clip gradients inside the `GradientHealthCallback` (violates the "Read-Only Observer" goal).
- **Metric Dependency:** Relying on validation metrics to trigger a halt (these callbacks focus on raw numerical health, not predictive performance).

---

## 10. Test Alignment

- **Red Team:** `tests/test_reproducibility_infra.py` (Numerical poisoning tests).
- **Green Team:** Verified via standard training integration tests in `tests/test_catalog.py`.

---

## 11. Evolution Notes

### Known Deviations / Technical Debt
- **Shared Thresholds:** Thresholds for "exploding" gradients (currently `100.0`) are hardcoded. These should ideally be moved to the DNA manifest to allow architecture-specific sensitivity tuning.

---

## End of Contract

This document defines the **intended meaning** of the Fortress Monitoring Callbacks.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
