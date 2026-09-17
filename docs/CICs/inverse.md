# Class Intent Contract: inverse (module contract)

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-09-10  
**Related ADRs:** ADR-002, ADR-012 (D-03)  

---

## 1. Purpose

`transformers/inverse.py` is the single place where the package reaches into Darts `Scaler` internals (`_fitted_params`, `_fit_called`) to run inverse transforms that preserve the sample dimension of probabilistic forecasts.

> **The 0.1.x codebase had two copies of this logic. This module exists so there is one, and so the fragile coupling has one address.**

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** choose or fit scalers.
- Does **not** handle `Pipeline` vs `Scaler` dispatch for the caller — callers pick the helper.

---

## 3. Responsibilities and Guarantees

- `extract_fitted_sklearn_scaler(scaler)` tolerates every `_fitted_params` layout seen across Darts versions: list of 1 (`global_fit=True`), list of N, tuple, dict with `"fitted"`, nested list.
- `inverse_transform_probabilistic_subset` — 3-D `(time, features, samples)` → 2-D → sklearn inverse → 3-D, exact legacy semantics.
- `inverse_transform_deterministic_subset` — 2-D direct.
- `fit_scaler_on_concatenated_subset`, `transform_subset_via_darts`, `inverse_transform_subset_via_darts` — the forward-side helpers `FeatureScalerManager` uses.

---

## 4. Inputs and Assumptions

- A fitted Darts `Scaler` or `Pipeline`; `TimeSeries` or arrays with the expected dimensionality.

---

## 5. Outputs and Side Effects

- Transformed arrays/series. Stateless.

---

## 6. Failure Modes and Loudness

- **Silent passthrough:** when `extract_fitted_sklearn_scaler` returns `None` *and* the scaler's transformer has no `inverse_func` (`:130-141`, `:170-177`), both inverse helpers fall through to `# Last resort: passthrough` and return the *unscaled* values **with no log line** (`inverse.py:143`, `:179`). The 0.1.x code at least warned. This is register **C-10**, escalated.

---

## 7. Boundaries and Interactions

- **Physical Zen:** `views_r2darts2/transformers/inverse.py`.
- Imports `darts` only. Used by `FeatureScalerManager` and `ViewsDataset`.
- Its existence is the subject of register **D-03** against ADR-012's "custom scaling wrappers are forbidden".

---

## 8. Examples of Correct Usage

Callers do not use this module directly; `FeatureScalerManager` and `ViewsDataset` do.

---

## 9. Examples of Incorrect Usage

- Accessing `scaler._fitted_params` anywhere else in the package.
- Treating a passthrough result as a successful inverse.

---

## 10. Test Alignment

- **Indirect only:** `tests/test_feature_scaler_manager.py`, `tests/test_parity_e2e.py`. No test imports `inverse` directly, and no test exercises the passthrough branch.

---

## End of Contract

This document defines the **intended meaning** of `inverse (module contract)`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
