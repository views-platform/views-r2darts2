# Class Intent Contract: DataFrameConverter / PredictionFrameConverter / FeatureFrameConverter / ParquetConverter / GridWriter

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-09-10  
**Related ADRs:** ADR-001, ADR-009, ADR-013 (D-04), ADR-016  

---

## 1. Purpose

The converter family is the **Data Airlock** of 0.2.x: one converter per input kind, each streaming its source into a Zarr store on disk with a shared schema.

> **Nothing enters a `ViewsDataset` except through one of these.**

---

## 2. Non-Goals (Explicit Exclusions)

- Do **not** decide targets or features — they receive the roles and validate them.
- Do **not** scale, log-transform, or clip.
- Do **not** hold the result — they write it and return the path.

---

## 3. Responsibilities and Guarantees

- `DataFrameConverter`, `PredictionFrameConverter`, `FeatureFrameConverter`: sources already in RAM; build one `xarray.Dataset` and sink it in one shot.
- `ParquetConverter`: the only genuinely out-of-core path; scans in Arrow batches and scatter-writes into a pre-allocated skeleton via `GridWriter`, so peak memory is one batch.
- `GridWriter`: pre-allocates a Zarr skeleton and scatter-writes dense grid regions; shared by `ParquetConverter` and `DatasetBuilder`.
- `build_schema_attrs` resolves column roles (`num2`/`num3`/`text`, `pred_*` detection) and is the single place the store's `.attrs` schema is defined.
- All numeric output is `np.float32` (`_FLOAT`). `ReproducibilityGate.Data.audit_frame_schema` raises `NumericalSanityError` on anything else — a defensive guard that is unreachable in practice, and in any case never invoked on the production path (C-44) (ADR-016).
- **One bypass:** `ViewsDataset.create_empty` (`base.py:938`) hand-builds the `.attrs` dict and enters via the `"dataset"` source kind, skipping `build_schema_attrs`.

---

## 4. Inputs and Assumptions

- A source of the matching kind; declared `targets`; `sample_size`; time/entity dimension names.

---

## 5. Outputs and Side Effects

- A Zarr group on disk inside a `ZarrStore` (or a durable path).

---

## 6. Failure Modes and Loudness

- `ValueError` — targets not found among columns; unsupported spec; shape conflicts.
- Pandas is imported here at module level (`:18`) — this, `readers.py` and one local import in `dataset/base.py` are the exceptions to the package's pandas-free claim.

---

## 7. Boundaries and Interactions

- **Physical Zen:** `views_r2darts2/dataset/converters.py` — **five classes in one file.** This is a deliberate homogeneous family and is the subject of register **D-04** against ADR-013's 1-class-1-file invariant.
- Imports `dataset.readers` only.

---

## 8. Examples of Correct Usage

Callers do not use converters directly; `ViewsDataset._ingest` dispatches on `readers.detect_source_type`.

---

## 9. Examples of Incorrect Usage

- Constructing a Zarr group by hand and pointing `ViewsDataset` at it, bypassing `build_schema_attrs`.
- Adding a sixth input kind without a converter and without teaching `detect_source_type` about it.

---

## 10. Test Alignment

- **Indirect only:** `tests/test_builder.py`, `tests/test_frame_builder.py`, `tests/test_parquet_loader.py`, `tests/test_views_dataset.py` exercise every converter through `ViewsDataset`. No test imports `converters` directly.

---

## End of Contract

This document defines the **intended meaning** of `DataFrameConverter / PredictionFrameConverter / FeatureFrameConverter / ParquetConverter / GridWriter`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
