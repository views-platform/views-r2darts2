# Class Intent Contract: StaticCovariateConfig / StaticCovariateStats / compute_static_covariates

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-09-10  
**Related ADRs:** ADR-001, ADR-003, ADR-013 (D-04)  

---

## 1. Purpose

`transformers/static_covariates.py` computes the per-entity "fingerprint" — `mu`, `sigma`, `max`, `trend` (OLS slope over integer position), `sparsity` (fraction of zeros) — for each target, optionally transformed (elementwise, then cross-entity), for injection as Darts static covariates.

> **It is a numpy-only, pandas-free replacement for a `groupby().agg().apply()` chain, and it claims bit-for-bit parity with that chain.**

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** decide *whether* stats are injected — the caller does.
- Does **not** read data; it takes arrays.

---

## 3. Responsibilities and Guarantees

- The five statistics use the legacy reduction semantics exactly (`ddof=1`, `Σ(t−t̄)(y−ȳ)/Σ(t−t̄)²`, `(y == 0).mean()`).
- `stat_time_range` is a **required** keyword: stats are computed over the training window the caller names. Passing `None` computes over the full frame and **warns** — this is the register C-08 seam, now off the live path (see §11).
- Elementwise transforms (`AsinhTransform`, `LogTransform`, `SqrtTransform`, `FourthRootTransform`) apply to `mu`/`sigma`/`max`/`trend`, never to `sparsity`; cross-entity `MaxAbsScaler` / `StandardScaler` follow the legacy zero-guards.

---

## 4. Inputs and Assumptions

- Sorted entity array, time array, one value array per target; a `StaticCovariateConfig`.

---

## 5. Outputs and Side Effects

- A `StaticCovariateStats` with `row_for_entity` and `column_names`. Stateless.

---

## 6. Failure Modes and Loudness

- `ValueError` — unknown transform step or stat name (`StaticCovariateConfig.__post_init__`).
- `KeyError` — `row_for_entity` for an unknown entity.
- `WARNING` — `stat_time_range=None` (leakage-with-warning; ADR-003 says this should be fail-loud).

---

## 7. Boundaries and Interactions

- **Physical Zen:** `views_r2darts2/transformers/static_covariates.py` — two dataclasses plus one function in one file (D-04).
- Imports nothing from the package.

---

## 8. Examples of Correct Usage

```python
stats = compute_static_covariates(entities, times, {"ged_sb": y},
                                  config=StaticCovariateConfig(transform="AsinhTransform->MaxAbsScaler"),
                                  stat_time_range=(train_start, train_end))
```

---

## 9. Examples of Incorrect Usage

- `stat_time_range=None` on data that includes the test window.
- Transforming `sparsity`.

---

## 10. Test Alignment

- **Green + Red:** `tests/test_static_covariates.py` (420 lines: parity per stat, transform chains, guards).

---

## 11. Evolution Notes

- **`compute_static_covariates` has zero production callers on 0.2.x.** `ViewsDataset.to_darts_timeseries` attaches only the entity id as a static covariate. The module is complete and tested but currently dead (register C-36). The bit-for-bit parity claim is against an implementation that no longer exists.

---

## End of Contract

This document defines the **intended meaning** of `StaticCovariateConfig / StaticCovariateStats / compute_static_covariates`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
