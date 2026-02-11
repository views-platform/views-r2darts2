# ADR-008: Observability and Explicit Failure

**Status:** Accepted  
**Date:** 2026-02-11  
**Deciders:** Simon Polichinel von der Maase  

---

## Context

In deep learning pipelines, failure is often silent. A model can train for days with NaN gradients, or a prediction can run on the CPU while the model is on the GPU, leading to massive performance degradation or race conditions. These are "structural lies" that compromise research.

Stack traces are insufficient for post-hoc auditability of long-running training jobs. Failures must be explicitly raised and persistently recorded in logs (e.g., W&B, local logs).

---

## Decision

This repository adopts the invariant: **Structural failures must be logged persistently and raised explicitly.**

### 1. The Fail-Loud Mandate
- **Device Integrity:** If a component is expected on `cuda` but found on `cpu`, the system must log the mismatch and raise an error immediately.
- **Numerical Sanity:** We use callbacks like `NaNDetectionCallback` and `GradientHealthCallback`. If NaNs are detected, the training must be terminated explicitly, not allowed to "zero-out" and continue.
- **Gate Failures:** Violations of `ReproducibilityGate` (temporal holes, DNA missing) must raise a `ReproducibilityError`. Swallowing these errors as warnings is forbidden.

### 2. Persistent Observability
- **Error Level:** Structural failures must be logged at the `ERROR` level or higher.
- **Contextual Logging:** Logs must include the offending tensor shape, device, and relevant DNA parameters at the moment of failure.
- **Redundancy:** Logging is not a substitute for raising. We raise to stop the lie; we log to explain it.

---

## Observability Patterns in `views-r2darts2`

- **The Configuration Summary:** Every run must log a "DNA Manifest Summary" to stdout/W&B before training starts.
- **The Health Check:** Long-running models should log gradient norms and loss values at every epoch.
- **The Audit Log:** Every `ReproducibilityGate` check must emit a log entry if it encounters an adversarial outlier, even if it doesn't terminate the run.

---

## Consequences

### Positive
- **Reduced Debugging Entropy:** You don't have to guess why a model failed 5 hours into a run.
- **Auditability:** Researchers can review the gradient health of past experiments.
- **Robustness:** Prevents race conditions (like the GPU/CPU thread conflict) from silently corrupting results.

### Negative
- Increases the amount of logging boilerplate.
- Can create "noisy" logs if thresholds for warnings (like outlier detection) are set too low.

---

## Notes

Observability must support understanding. Failure must never be silent. This ADR reinforces the "Fail-Loud" principle established in ADR-003.
