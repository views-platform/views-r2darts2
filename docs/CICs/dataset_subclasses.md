# Class Intent Contract: PGDataset / PGMDataset / PGYDataset / CDataset / CMDataset / CYDataset

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-09-10  
**Related ADRs:** ADR-001, ADR-003, ADR-013 (D-04)  

---

## 1. Purpose

Six thin subclasses of `ViewsDataset`, one per VIEWS level of analysis, each adding exactly one invariant on the store's dimension names.

> **A `PGMDataset` is a `ViewsDataset` that refuses to exist unless its entity axis is `priogrid_id` and its time axis is `month_id`.**

---

## 2. Non-Goals (Explicit Exclusions)

- Add **no** behaviour beyond `validate_indices`.
- Do **not** infer the level from data — `_loa_to_class(loa)` maps a declared code to a class, and an unknown code falls back to the base class.

---

## 3. Responsibilities and Guarantees

- `PGDataset` — entity must be `priogrid_id`. `CDataset` — entity must be `country_id`.
- `PGMDataset(PGDataset)`, `CMDataset(CDataset)` — time must be `month_id`.
- `PGYDataset`, `CYDataset` — time must be `year_id` (and the entity of their parent).
- Invariants are layered through inheritance so one `validate_indices` chain checks both axes — except `PGYDataset`, which inherits `ViewsDataset` directly and re-checks the entity inline (`subclasses.py:36`), an asymmetry with `CYDataset(CDataset)`. Behaviour is identical.

---

## 4. Inputs and Assumptions

- Same as `ViewsDataset`; the `loa` code is declared by the caller (`ViewsDataset.for_loa`, `DatasetBuilder(loa=...)`).

---

## 5. Outputs and Side Effects

- None beyond the base class.

---

## 6. Failure Modes and Loudness

- `ValueError` naming the expected and actual dimension, raised at construction.

---

## 7. Boundaries and Interactions

- **Physical Zen:** `views_r2darts2/dataset/subclasses.py` — six classes in 81 lines; register **D-04** (ADR-013).
- Imports `dataset.base` at top level; `base._loa_to_class` imports these lazily — the deferral breaks the cycle and must stay.

---

## 8. Examples of Correct Usage

```python
ds = ViewsDataset.for_loa("cm", source=frame)   # -> CMDataset, or ValueError
```

---

## 9. Examples of Incorrect Usage

- Subclassing one of these to add data logic — that belongs on `ViewsDataset`.

---

## 10. Test Alignment

- **Red:** none — no test triggers a subclass `validate_indices` `ValueError`; `tests/test_views_dataset.py` only asserts `for_loa` routing (`isinstance`) for two subclasses.
- **Green:** `tests/test_builder.py` (built datasets have the declared subclass).

---

## End of Contract

This document defines the **intended meaning** of `PGDataset / PGMDataset / PGYDataset / CDataset / CMDataset / CYDataset`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
