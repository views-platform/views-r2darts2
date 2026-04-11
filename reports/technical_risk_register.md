# Technical Risk Register

| Register Info     | Details                              |
|-------------------|--------------------------------------|
| Project           | views-r2darts2                       |
| Owner             | Simon Polichinel von der Maase       |
| Last Updated      | 2026-04-11                           |
| Total Concerns    | 3                                    |
| Open Concerns     | 3                                    |
| Resolved Concerns | 0                                    |
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

### Evaluation sequencing fragility

- **Entries:** C-01, C-02
- **Root cause:** Rolling-origin sequencing math (`total_sequence_number` derivation, `predict()` end-index) is spread across multiple inline call sites in `views_r2darts2/engines/` without regression tests. Drift between sites has already caused one production bug (`f78bbf7`).
- **Recommended action:** Single PR that (a) adds regression tests for `DartsForecaster.predict()` asserting `sequence_number=0` returns earliest month `== partition["test"][0]`, and (b) extracts `_resolve_total_sequence_number(config)` as the authoritative helper, replacing the three inline copies at `darts_forecasting_model_manager.py:253`, `:434`, `:475`.

---

## Open Concerns

### C-01 — No regression test for `DartsForecaster.predict()` rolling-origin end index

- **Tier:** 2
- **Source:** review-diff + pr-review (PR #10)
- **Location:** `views_r2darts2/engines/darts_forecaster.py:558` (and missing `tests/test_forecaster.py` coverage)
- **Trigger:** When refactoring the rolling-origin slicing logic in `DartsForecaster.predict()` — or when touching `_preprocess_timeseries` — verify that `sequence_number=0` returns a `DataFrame` whose earliest month equals `partition_dict["test"][0]` (not `test[0] + 1`).
- **Narrative:** Commit `f78bbf7` fixed an off-by-one where `end = test_start + sequence_number` caused the model to observe month `test_start` and forecast one month too late (e.g. 446–481 instead of 445–480 for sequence 0). The fix aligns with the views-pipeline-core convention `base_origin = test[0] - 1`, but the slicing math is now split between `predict()` (sets `end`) and `_preprocess_timeseries` (slices to `end + 1`, right-exclusive) — two places that must agree. The commit message explicitly acknowledges the missing unit test as a TODO. This bug was only caught in production by `_assert_predictions_in_step_window()` pre-flight in views-pipeline-core; without a local regression test, a future refactor of either site can silently re-introduce the same drift, and the pre-flight is the only remaining safety net.
- **Cross-refs:** Related to C-02 (same file, same evaluation path, same absence of regression coverage). Part of cluster *Evaluation sequencing fragility*.

---

### C-02 — `total_sequence_number` formula duplicated at three call sites

- **Tier:** 3
- **Source:** pr-review (PR #10)
- **Location:** `views_r2darts2/engines/darts_forecasting_model_manager.py:253` (`_evaluate_model_artifact`), `:434` (sweep `HORIZON LOCKDOWN` inline), `:475` (`_evaluate_sweep`)
- **Trigger:** When modifying the sequence-count contract (e.g. adding a new eval type, changing partition semantics, or adjusting the relationship between `steps`, `test_len`, and `MAX_SHIFT_COUNT`), update all three sites *and* consider extracting to `_resolve_total_sequence_number(config)` to eliminate drift.
- **Narrative:** Commit `872ebaa` replaced a hardcoded `total_sequence_number = 12` with the partition-derived formula `test_end - test_start + 1 - time_steps + 1` at three call sites: `_evaluate_model_artifact`, the sweep `HORIZON LOCKDOWN` block, and `_evaluate_sweep`. The formula is correct but is inlined three times with slightly different variable names and wrapping styles. This is the same copy-paste drift pattern that caused the `_has_evaluation_metrics` bug (`47d13dd` / `a8646ed`) — where the subclass re-implemented a 7-key `any([...])` block instead of delegating to the base class, then silently fell out of sync. A helper method `_resolve_total_sequence_number(config)` would give one place to test and one place to change.
- **Cross-refs:** Related to C-01 (both concern the evaluation path and both lack unit tests). Part of cluster *Evaluation sequencing fragility*.

---

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

(No resolved concerns yet.)

---

## Register Conventions

- **ID format:** `C-xx` for concerns, `D-xx` for disagreements. IDs are permanent — gaps indicate merged or resolved entries.
- **Sources:** `repo-assimilation`, `expert-review`, `test-review`, `falsification-audit`, `clean-architecture-review`, `pr-review`, `review-diff`, `tech-debt-audit`, `incident`.
- **Resolution:** Move to "Resolved Concerns" with resolution date and one-line summary when addressed. Do not delete.
- **Header counts:** Manually maintained — update whenever a concern is added or resolved.
- **Governed by:** ADR-014.
