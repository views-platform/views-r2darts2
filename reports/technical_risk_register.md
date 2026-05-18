# Technical Risk Register

| Register Info     | Details                              |
|-------------------|--------------------------------------|
| Project           | views-r2darts2                       |
| Owner             | Simon Polichinel von der Maase       |
| Last Updated      | 2026-04-27                           |
| Total Concerns    | 5                                    |
| Open Concerns     | 0                                    |
| Resolved Concerns | 5                                    |
| Governed by       | ADR-014                              |

---

## Tier Definitions

| Tier | Severity | Description |
|------|----------|-------------|
| 1 | Critical | Silent data corruption or model output correctness risk. Requires immediate attention. |
| 2 | High | Structural fragility that will cause failures under realistic change scenarios. |
| 3 | Medium | Maintainability or coupling issues that increase cost of change. |
| 4 | Low | Code quality concerns that do not affect correctness or reliability. |

---

## Causal Clusters

### Evaluation sequencing fragility — *resolved 2026-04-11*

- **Entries:** ~~C-01, C-02~~ (both resolved, see Resolved Concerns)
- **Root cause:** Rolling-origin sequencing math (`total_sequence_number` derivation, `predict()` end-index) was spread across multiple inline call sites in `views_r2darts2/engines/` without regression tests. Drift between sites had already caused one production bug (`f78bbf7`).
- **Resolution:** PR #10 extracted `DartsForecastingModelManager._resolve_total_sequence_number(partition, max_steps)` with a `ValueError` guard when `test_len < max_steps`, replaced all three inline copies, and added regression tests covering (a) the `predict(sequence_number=0)` base-origin convention and (b) the `test_len < max_steps` silent-failure mode flagged by Copilot.

---

## Open Concerns

(All concerns resolved as of 2026-04-27. See Resolved Concerns below.)

---

## Disagreements

(No disagreements registered yet.)

---

## Resolved Concerns

### C-05: Feature scaler instantiated before empty-features guard in `DartsForecaster.__init__` — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-05 |
| Resolved | 2026-04-27 |
| Resolution | Restructured `__init__` feature-scaler block as an `if/elif/else`: the empty-features guard now runs first, so `FeatureScalerManager` and `_instantiate_scaler` are never called when `features=[]`. |

---

### C-01 — No regression test for `DartsForecaster.predict()` rolling-origin end index *(resolved 2026-04-11)*

- **Tier:** 2
- **Source:** review-diff + pr-review (PR #10)
- **Location:** `views_r2darts2/engines/darts_forecaster.py:558`
- **Resolution:** Added `test_predict_rolling_origin_sequence_zero_ends_at_test_start_minus_one` and `test_predict_rolling_origin_advances_one_month_per_sequence` in `tests/test_forecaster.py`. Both tests capture the `_preprocess_timeseries` call kwargs and assert the base-origin convention (`end == test_start - 1 + sequence_number`), so a future refactor of either `predict()` or `_preprocess_timeseries` cannot silently re-introduce the off-by-one fixed in `f78bbf7`. Resolved as part of the C-01/C-02 cluster PR.

---

### C-02 — `total_sequence_number` formula duplicated at three call sites *(resolved 2026-04-11)*

- **Tier:** 2 *(escalated from Tier 3 after Copilot reviewer surfaced a silent-failure mode: when `max(steps) > test_len`, Python's `[None] * -1 == []` and `range(-1) == []` produced an empty prediction batch with no error signal — matching the Tier 1/2 silent-corruption criterion)*
- **Source:** pr-review (PR #10) + Copilot comment on `darts_forecasting_model_manager.py:253`
- **Location:** Previously at `darts_forecasting_model_manager.py:253`, `:434`, `:475`
- **Resolution:** Extracted `DartsForecastingModelManager._resolve_total_sequence_number(partition, max_steps)` as a `@staticmethod` with a `ValueError` guard that fails loudly when `test_len < max_steps` (instead of silently returning zero or negative, which previously produced an empty prediction batch). All three call sites now delegate to the helper. Unit tests in `tests/test_darts_forecasting_model_manager.py` cover the standard case, the `test_len == max_steps` boundary (returns 1), and the `test_len < max_steps` failure mode Copilot flagged.

---

### C-03 — Chain-spec construction divergence in `ScalerSelector.instantiate_darts_scaler` *(resolved 2026-04-11)*

- **Tier:** 3
- **Source:** pr-review (PR #10) + Copilot comment on `scaler_selector.py:179`
- **Location:** Previously at `views_r2darts2/transformers/scaler_selector.py:135-195`
- **Resolution:** Collapsed the divergent list / dict-chain-list / dict-chain-str code paths into a single `ScalerSelector._build_chain_or_single(scaler_names)` helper. All chain-spec forms — string, list, dict-with-string-chain, dict-with-list-chain — now route through one definition, so adding a new scaler step or changing chain semantics is a single edit. The helper also fixes a bug Copilot surfaced separately: single-element dict-chain inputs (`{"chain": ["StandardScaler"]}` and `{"chain": "StandardScaler"}`) previously returned a one-element `Pipeline`, inconsistent with the list-form `["StandardScaler"]` which returned a bare `Scaler`; all single-element forms now return a bare `Scaler`, and empty lists / non-string elements raise `ValueError`/`TypeError` instead of silently producing malformed pipelines. Regression tests in `tests/test_scaling.py::TestInstantiateDartsScalerConsolidation` cover all four equivalent chain forms, the single-element collapse, and the empty/invalid-element error paths.

---

### C-04 — `FeatureScalerManager._instantiate_scaler` silently propagates `None` *(resolved 2026-04-11)*

- **Tier:** 2
- **Source:** Copilot comment on `feature_scaler_manager.py:87` (PR #10)
- **Location:** Previously at `views_r2darts2/transformers/feature_scaler_manager.py:83-88`
- **Resolution:** `_instantiate_scaler(None)` used to return `None`, which was then stored in `self._scalers` and later dereferenced as a Darts `Scaler`/`Pipeline` at fit time, producing an `AttributeError` on `scaler.transformer` that was hard to trace back to the misconfigured group. The fix makes `FeatureScalerManager._instantiate_scaler` raise `ValueError` at parse time when `scaler_cfg is None`, so the misconfiguration fails loudly at manager construction rather than at fit time. `DartsForecaster._instantiate_scaler` retains its `None → None` pass-through because a forecaster legitimately supports having no target or feature scaler — the narrow fix applies only to the manager where `None` is always a misconfiguration. Regression test in `tests/test_scaling.py::TestFeatureScalerManagerRejectsNoneScalerConfig` covers the direct `None` rejection and the named-group-without-scaler-and-no-default reproducer.
- **Note:** Registered and resolved in the same commit — this was surfaced by a second Copilot comment during PR #10 review after the initial register (7b92bfa). Tier 2 because the silent propagation reached downstream fit code and only surfaced as a non-obvious `AttributeError`, meeting the "structural fragility with clear trigger" criterion.

---

## Register Conventions

- **ID format:** `C-xx` for concerns, `D-xx` for disagreements. IDs are permanent — gaps indicate merged or resolved entries.
- **Sources:** `repo-assimilation`, `expert-review`, `test-review`, `falsification-audit`, `clean-architecture-review`, `pr-review`, `review-diff`, `tech-debt-audit`, `incident`.
- **Resolution:** Move to "Resolved Concerns" with resolution date and one-line summary when addressed. Do not delete.
- **Header counts:** Manually maintained — update whenever a concern is added or resolved.
- **Governed by:** ADR-014.
