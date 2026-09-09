# ADR-011: Hardware Integrity and Parallelism

**Status:** Accepted  
**Date:** 2026-02-11  
**Deciders:** Simon Polichinel von der Maase  
**Revised:** 2026-09-10 — re-derived against the 0.2.x codebase (`development` @ `fe7e681`). Original decision unchanged unless stated.  

---

## Context

Darts models exhibit an undocumented behavior where they shift to the CPU during `teardown()` after training or prediction. In multi-threaded environments (like rolling-window evaluations on GPU), this induces **race conditions**: Thread A may be performing a forward pass on `cuda:0` while Thread B's teardown operation is moving the weights to `cpu`. 

This leads to `RuntimeError` (Device Mismatch) or silent, massive performance degradation if the model continues to run on the CPU while tensors remain on the GPU.

---

## Decision

1.  **Device Self-Healing:** The `DartsForecaster` must implement a **Verify-and-Restore** pattern before every prediction. It must audit the current device of the model weights and move them back to the target device if drift is detected.
2.  **GPU Serialization:** Parallel prediction (via `ThreadPoolExecutor`) is **forbidden** for GPU-resident models. `max_workers` must be forced to `1` for all GPU prediction jobs.
3.  **CPU Parallelism:** Parallel prediction is permitted and encouraged for CPU-only models to maximize hardware utilization.

---

## Rationale

- **Robustness:** Device-shifting is a known weakness in the current Darts/PyTorch Lightning integration. Self-healing is the only way to guarantee the "Fortress" doesn't crash during long evaluations.
- **Reliability:** GPU contexts are not designed for the type of rapid device-switching that Darts' teardown induces. Forcing sequential execution eliminates the race condition entirely.
- **Performance:** While sequential prediction is slower than parallel, a crash is infinitely slower.

---

## Consequences

### Positive
- Eliminated "Device Mismatch" crashes during evaluation sweeps.
- Predictable GPU memory management.
- Safer multi-GPU utilization (once implemented).

### Negative
- Evaluation runs on GPUs will take longer than CPU equivalents for models with light computational overhead.

---

## Implementation Notes

- **Enforcement:** `DartsForecaster.predict()` calls `_ensure_model_on_device()` before every forecast; the device itself is resolved once by `get_device()` in `views_r2darts2/infrastructure/device.py`.
- **Orchestration:** `DartsForecastingModelManager` inspects `forecaster.device` (a `str`: `"cuda"`, `"mps"` or `"cpu"`) and sets `max_workers=1` unless it equals `"cpu"`.

---

## Validation & Monitoring

- **Failure Mode:** If self-healing fails to move the model from CPU to GPU, the system must **fail-loud** and abort the prediction (ADR-008).
  *Compliance note (2026-09-10):* the current code logs a `WARNING` and continues on CPU. See register **D-01** — this line states intent until that ruling is made.
- **Logs:** Successful restorations should be logged at the `INFO` level.
