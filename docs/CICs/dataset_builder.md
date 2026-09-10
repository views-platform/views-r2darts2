# Class Intent Contract: DatasetBuilder

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-09-10  
**Related ADRs:** ADR-001, ADR-002, ADR-008  

---

## 1. Purpose

`DatasetBuilder` scaffolds a Zarr store from a declared level-of-analysis and coordinate set, then fills it batch by batch, so prediction tensors far larger than RAM can be assembled with peak memory of one batch.

> **It is the write path for streaming predictions; `ViewsDataset` ingestion is for sources that already exist whole.**

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** compute predictions — it stores what it is given.
- Does **not** infer coordinates from batches; they are declared up front.
- Does **not** read the store back (that is `transformers/frame_builder.py`).

---

## 3. Responsibilities and Guarantees

- Declares time/entity dimension names from the `loa` code (`pgm` → `month_id`/`priogrid_id`, `cm` → `month_id`/`country_id`, `pgy`/`cy` the year variants).
- Pre-allocates a NaN-filled Zarr skeleton — metadata only, nothing materialised — and scatter-writes via `converters.GridWriter`.
- Two write entry points: `write_batch(times, entities, columns)` for scattered `(time, entity)` rows, and `write_time_slice(time, columns)` for one full `(E, sample_size)` slice at a time; each has its own shape rules and its own `strict` duplicate check.
- Coordinates are sorted and unique; every batch is validated against them and **fails loud naming the offending values**.
- Never-written cells stay NaN. `build(require_complete=True)` fails loud on any unwritten cell (requires `strict=True` or `track_coverage=True`).
- Overwrites are last-write-wins by default; `strict=True` raises on any duplicate `(time, entity)` write.
- `path=None` writes into a self-cleaning scratch store; an absolute `path` survives `close()`.
- The built dataset is a real `ViewsDataset` subclass for the `loa`, indistinguishable from an ingested one.

---

## 4. Inputs and Assumptions

- `loa`, `times`, `entities`, `variables` (name → spec), `sample_size >= 1`, `targets`.
- Batches arrive as `(times, entities, columns)` with shapes matching the declared coordinates.

---

## 5. Outputs and Side Effects

- `build()` returns a lazy `ViewsDataset`. `coverage` reports the written fraction when tracking is on.
- **Side effects:** creates a Zarr directory (scratch or durable).

---

## 6. Failure Modes and Loudness

- `ValueError` on: non-1-D or empty coordinates, duplicates, unknown coordinate values in a batch, no columns, element-count mismatch, duplicate write under `strict=True`, incomplete grid under `require_complete=True`.
- `RuntimeError` on any write after `close()`.
- Two silent paths exist: `write_batch` returns without error on an empty batch (before column validation), and `_validate_column` checks `arr.size` only, so a transposed `(sample_size, n_rows)` array with the right element count is reshaped rather than rejected.

---

## 7. Boundaries and Interactions

- **Physical Zen:** `views_r2darts2/dataset/builder.py`.
- Imports `dataset.base`, `dataset.converters`, `dataset.zarr_store` at top level; `base` imports it lazily in `ViewsDataset.builder()` — that deferral breaks a cycle and must stay.

---

## 8. Examples of Correct Usage

```python
with ViewsDataset.builder(loa="pgm", times=np.arange(528, 540), entities=pg_ids,
                          variables={"pred_ln_sb_best": "num3"}, sample_size=32,
                          targets=["pred_ln_sb_best"]) as b:
    for t, ents, values in batches():
        b.write_batch(times=t, entities=ents, columns={"pred_ln_sb_best": values})
    ds = b.build(require_complete=True)
```

---

## 9. Examples of Incorrect Usage

- Writing entities that were not declared, expecting them to be appended.
- Relying on the default last-write-wins to "fix" a batch that should have been rejected.

---

## 10. Test Alignment

- **Green + Red:** `tests/test_builder.py` (36 tests), `tests/test_frame_builder.py`, `tests/test_streaming_predict_builder.py`.

---

## End of Contract

This document defines the **intended meaning** of `DatasetBuilder`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
