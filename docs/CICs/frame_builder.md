# Class Intent Contract: frame_builder (module contract)

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-09-10  
**Related ADRs:** ADR-001, ADR-008  
**Last reviewed:** 2026-09-29 (scratch-directory ownership, issue #54)  

---

## 1. Purpose

`transformers/frame_builder.py` is the read path complementing `DatasetBuilder`: it streams predictions out of a Zarr store into row-major `(N, S)` memmap files in entity-aligned blocks and wraps them as `views_frames.PredictionFrame`s whose `values` is a read-only `np.memmap`.

> **Peak memory is one entity block, and the returned frame occupies ~0 RAM until touched.**

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** compute or transform predictions.
- Does **not** keep the Zarr store — see §3.
- `PredictionScratch` does **not** free its directory on garbage collection, unlike `ZarrStore`. See §3.

---

## 3. Responsibilities and Guarantees

- Reads in entity blocks aligned to the Zarr entity-chunk size (default block 1024, a multiple of 256), so every chunk is read exactly once.
- **Verify, then delete:** after writing and flushing the memmaps it verifies shape, target names, and a readback. Only then is the Zarr store deleted (`zarr_cleanup=True`, the default; pass `zarr_cleanup=False` to keep it). **On any verification failure the Zarr store is kept and `PredictionFrameVerificationError` is raised** — the failure mode is never data loss.
- **`out_dir` has an owner:** `PredictionScratch` creates the directory and removes it exactly once — on `close()`, or at interpreter exit if nobody closed it. Before 0.2.4 the directory had no owner and was never removed; on the full grid that is tens of gigabytes per `predict()` call.
- **Cleanup is deliberately not tied to garbage collection.** `ZarrStore` frees its directory via `weakref.finalize` and `__del__`, which is safe because nobody else holds a handle to its bytes. The frames built here hand the caller `np.memmap` views *into* the directory, and those outlive the owner — so collection-time cleanup would delete files still mapped by a live frame. The `atexit` registration holds a strong reference on purpose.
- **The consumer decides when to close.** Only the caller knows whether the values have been copied out. `DartsForecaster.release_prediction_scratch` is that seam; the manager calls it once predictions have been converted to DataFrames, and skips it when frames are handed out as frames.

---

## 4. Inputs and Assumptions

- A `ViewsDataset` in prediction mode; a destination directory for the memmap files.

---

## 5. Outputs and Side Effects

- `dict[str, PredictionFrame]`, memmap-backed.
- **Side effects:** writes memmap files; deletes the source Zarr directory on success; registers an `atexit` hook per `PredictionScratch`, unregistered on `close()` so a long-running sweep does not accumulate them.

---

## 6. Failure Modes and Loudness

- `KeyError` — requested target not in the dataset.
- `PredictionFrameVerificationError(RuntimeError)` — shape, name, or readback mismatch; Zarr retained.
- **Silent, by design:** `PredictionScratch.close()` uses `ignore_errors=True`, so a directory that cannot be removed leaves no error. The `atexit` backstop then retries at exit.
- **Known limitation:** when predictions are handed out as frames rather than DataFrames, no consumer signals that it has finished reading them, so those directories live until the process exits.

---

## 7. Boundaries and Interactions

- **Physical Zen:** `views_r2darts2/transformers/frame_builder.py`.
- Imports `views_frames`, `numpy`; nothing from the package. Called by `DartsForecaster._predict_streaming`.

---

## 8. Examples of Correct Usage

```python
frames = build_prediction_frames_from_dataset(ds, target_names, out_dir)   # ds's Zarr is gone afterwards
```

---

## 9. Examples of Incorrect Usage

- Reading `ds` after calling this — its store has been deleted.
- Catching `PredictionFrameVerificationError` and deleting the Zarr anyway.

---

## 10. Test Alignment

- **Green:** `tests/test_frame_builder.py` (shape/name/readback verification).
- **Red:** `tests/test_zarr_cleanup.py::test_failed_readback_keeps_zarr_and_raises` (failure retention) and the delete-exactly-once interplay.
- **Lifecycle:** `tests/test_prediction_scratch_cleanup.py` (`close()` idempotence, the `atexit` backstop in a child process, frames still readable before release, and the manager's format guard — that releasing is skipped when frames are handed out).

---

## End of Contract

This document defines the **intended meaning** of `frame_builder (module contract)`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
