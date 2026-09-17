# Class Intent Contract: ZarrStore

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-09-10  
**Related ADRs:** ADR-001, ADR-008  

---

## 1. Purpose

`ZarrStore` owns one scratch directory holding one or more Zarr groups and guarantees it is removed **exactly once**.

> **It is the reason a `ViewsDataset` can be disk-resident without leaking temp directories.**

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** materialise array data; it only routes `xarray.Dataset` writes and reads to the directory.
- Does **not** manage durable stores — an absolute path given to `DatasetBuilder` bypasses it.

---

## 3. Responsibilities and Guarantees

- Cleanup on `close()`, context-manager exit, `__del__`, or interpreter shutdown (`atexit` is the backstop), via `weakref.finalize`, and never more than once.
- Every default store name is UUID-suffixed (`dataset_<hex>.zarr`), so two stores in one process cannot silently overwrite each other — the legacy fixed `dataset.zarr` name did exactly that.

---

## 4. Inputs and Assumptions

- Optional `prefix` and `base_dir` for `tempfile.mkdtemp`.

---

## 5. Outputs and Side Effects

- `sink_zarr(ds, path=None)` writes; `open_zarr(path=None)` reads. `path`/`closed` properties.
- **Side effects:** filesystem only.

---

## 6. Failure Modes and Loudness

- `RuntimeError` — `sink_zarr` or `open_zarr` after `close()`.

---

## 7. Boundaries and Interactions

- **Physical Zen:** `views_r2darts2/dataset/zarr_store.py` (97 lines).
- Used by `ViewsDataset` and `DatasetBuilder`. Imports nothing from the package.

---

## 8. Examples of Correct Usage

```python
with ZarrStore() as store:
    path = store.sink_zarr(xr_dataset)
    ds = store.open_zarr(path)
# directory is gone here
```

---

## 9. Examples of Incorrect Usage

- Caching `store.path` and reading it after the store is closed.
- Two datasets sharing one store and assuming they get separate default names *without* the UUID suffix.

---

## 10. Test Alignment

- **Lifecycle:** `tests/test_zarr_cleanup.py` (exactly-once cleanup, verify-then-delete interplay with `frame_builder`).
- **Indirect:** `tests/test_builder.py`, `tests/test_frame_builder.py`.

---

## End of Contract

This document defines the **intended meaning** of `ZarrStore`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
