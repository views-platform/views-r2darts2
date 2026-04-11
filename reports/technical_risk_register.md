# Technical Risk Register

| Register Info     | Details                              |
|-------------------|--------------------------------------|
| Project           | views-r2darts2                       |
| Owner             | Simon Polichinel von der Maase       |
| Last Updated      | 2026-04-11                           |
| Total Concerns    | 3                                    |
| Open Concerns     | 1                                    |
| Resolved Concerns | 2                                    |
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

### C-03 — Chain-spec construction divergence in `ScalerSelector.instantiate_darts_scaler`

- **Tier:** 3
- **Source:** pr-review (PR #10)
- **Location:** `views_r2darts2/transformers/scaler_selector.py:135-195` (new `instantiate_darts_scaler` method)
- **Trigger:** When adding a new scaler type, a new transformer wrapper, or a new normalization step to the chain-construction logic — verify that the behavior is identical across all four input forms (`"A->B"`, `["A", "B"]`, `{"chain": "A->B"}`, `{"chain": ["A", "B"]}`). Adding a step to one code path and forgetting the other is the expected failure mode.
- **Narrative:** `instantiate_darts_scaler` accepts four equivalent ways to specify a scaler chain, but routes them through two different constructors: string chains (`"A->B"`) go through `ScalerSelector.get_chained_scaler()`, while list and dict-chain forms go through a local `_make_pipeline()` helper that builds the `darts.Pipeline` inline. The two paths should produce semantically identical objects today, but a future change (e.g. adding `global_fit=True` handling, adding an intermediate `InvertibleDataTransformer` wrap, or changing how scalers are named) must be made in both places. A reviewer cannot tell by inspection that the two paths are equivalent — they are structurally different. Prefer routing list/dict-chain inputs through `get_chained_scaler` (after joining with `->`), or invert the delegation so `get_chained_scaler` calls `_make_pipeline`.

---

## Disagreements

(No disagreements registered yet.)

---

## Resolved Concerns

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

## Register Conventions

- **ID format:** `C-xx` for concerns, `D-xx` for disagreements. IDs are permanent — gaps indicate merged or resolved entries.
- **Sources:** `repo-assimilation`, `expert-review`, `test-review`, `falsification-audit`, `clean-architecture-review`, `pr-review`, `review-diff`, `tech-debt-audit`, `incident`.
- **Resolution:** Move to "Resolved Concerns" with resolution date and one-line summary when addressed. Do not delete.
- **Header counts:** Manually maintained — update whenever a concern is added or resolved.
- **Governed by:** ADR-014.
