# Class Intent Contract: darts_bridge (module contract)

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-09-10  
**Related ADRs:** ADR-001, ADR-002, ADR-009  

---

## 1. Purpose

`transformers/darts_bridge.py` is the **only module in the package allowed to import pandas.** It converts `views_frames` data to Darts `TimeSeries` and back, because Darts requires `pandas.Index`/`pandas.DataFrame` for its time index and static covariates.

> **When Darts drops pandas, or this package switches backend, only this file changes.**

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** scale, slice, or clip.
- Does **not** persist anything.
- Is **not** a class — four functions with one shared invariant, which is why it carries a module-level contract.

---

## 3. Responsibilities and Guarantees

- `build_entity_timeseries(...)` — one `TimeSeries` per entity from frame-shaped arrays, with the entity id as a static covariate.
- `prediction_frame_from_darts(...)`, `prediction_frames_from_darts(...)` — Darts predictions (2-D or 3-D) back to `views_frames.PredictionFrame`(s), sample dimension preserved.
- **Not currently on the production path:** `build_entity_timeseries` is imported by `dataset/base.py:1597` and never called; `ViewsDataset.to_darts_timeseries` constructs `TimeSeries` directly at `base.py:1702` with its own local `import pandas`. The bridge's inbound half is bypassed (register C-45).
- `prediction_frames_to_dataframe(...)` — the on-demand DataFrame view; never called implicitly.
- Darts `TimeSeries` are short-lived views; the memmap-backed frame stays the source of truth.

---

## 4. Inputs and Assumptions

- Integer time ids and entity ids; Darts series carrying the entity id in `static_covariates`.

---

## 5. Outputs and Side Effects

- None beyond return values. Stateless.

---

## 6. Failure Modes and Loudness

- `ValueError` — empty prediction list; component mismatch or multi-target series; entity id missing from static covariates. *(No time-index comparison is performed.)*

---

## 7. Boundaries and Interactions

- **Physical Zen:** `views_r2darts2/transformers/darts_bridge.py`.
- Imports `darts`, `views_frames`, `pandas`, `numpy`; nothing from the package. Imported by `ViewsDataset` (lazily) and `DartsForecastingModelManager`.
- Any new `import pandas` elsewhere in the package is a violation of this contract. Recorded exceptions: `dataset/converters.py:18` (module-level), `dataset/readers.py:30` (function-local), `dataset/base.py:1659` (function-local, inside `to_darts_timeseries`).

---

## 8. Examples of Correct Usage

```python
frames = prediction_frames_from_darts(predictions=darts_preds, target_columns=ds.targets, ...)   # keyword-only
df = prediction_frames_to_dataframe(frames)   # only when a DataFrame is actually needed
```

---

## 9. Examples of Incorrect Usage

- Calling `TimeSeries.pd_dataframe()` in the engines to get a DataFrame.
- Building a `TimeSeries` in `dataset/base.py` directly instead of through `build_entity_timeseries` — *which is what production currently does (C-45); this line states the intent, not the state.*

---

## 10. Test Alignment

- **Green + Red:** `tests/test_darts_bridge.py` (round-trips, mismatches, empty inputs); `tests/test_darts_forecaster.py`, `tests/test_views_dataset.py` (integration).

---

## End of Contract

This document defines the **intended meaning** of `darts_bridge (module contract)`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
