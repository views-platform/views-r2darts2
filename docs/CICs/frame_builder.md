# Class Intent Contract: frame_builder (module contract)

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-09-10  
**Related ADRs:** ADR-001, ADR-008  

---

## 1. Purpose

`transformers/frame_builder.py` is the read path complementing `DatasetBuilder`: it streams predictions out of a Zarr store into row-major `(N, S)` memmap files in entity-aligned blocks and wraps them as `views_frames.PredictionFrame`s whose `values` is a read-only `np.memmap`.

> **Peak memory is one entity block, and the returned frame occupies ~0 RAM until touched.**

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** compute or transform predictions.
- Does **not** keep the Zarr store — see §3.

---

## 3. Responsibilities and Guarantees

- Reads in entity blocks aligned to the Zarr entity-chunk size (default block 1024, a multiple of 256), so every chunk is read exactly once.
- **Verify, then delete:** after writing and flushing the memmaps it verifies shape, target names, and a readback. Only then is the Zarr store deleted (`zarr_cleanup=True`, the default; pass `zarr_cleanup=False` to keep it). **On any verification failure the Zarr store is kept and `PredictionFrameVerificationError` is raised** — the failure mode is never data loss.

---

## 4. Inputs and Assumptions

- A `ViewsDataset` in prediction mode; a destination directory for the memmap files.

---

## 5. Outputs and Side Effects

- `dict[str, PredictionFrame]`, memmap-backed.
- **Side effects:** writes memmap files; deletes the source Zarr directory on success.

---

## 6. Failure Modes and Loudness

- `KeyError` — requested target not in the dataset.
- `PredictionFrameVerificationError(RuntimeError)` — shape, name, or readback mismatch; Zarr retained.

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

---

## End of Contract

This document defines the **intended meaning** of `frame_builder (module contract)`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
