# Technical Risk Register

| Register Info     | Details                              |
|-------------------|--------------------------------------|
| Project           | views-r2darts2                       |
| Owner             | Simon Polichinel von der Maase       |
| Last Updated      | 2026-09-09                           |
| Total Concerns    | 34                                   |
| Open Concerns     | 29                                   |
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

> The "silent-acceptance seam" concerns (C-07, C-08, C-09, C-10, and now C-20, C-22, C-23, C-24)
> share a root theme: seams where the otherwise fail-loud pipeline silently accepts a wrong,
> leaky, or duplicated configuration. A `review-rr` strategic pass may wish to group these into
> a formal causal cluster.

### C-08 — `stat_time_range=None` leaks test data into static-covariate stats with only a warning

- **Tier:** 1 *(silent target leakage into features → optimistic, incorrect forecasts with no error signal; meets the silent-corruption criterion)*
- **Source:** repo-assimilation (Phase 4 — transformers)
- **Trigger:** A new evaluation path or downstream consumer calls `as_darts_timeseries()` without passing `stat_time_range`.
- **Location:** `views_r2darts2/transformers/views_dataset_darts.py:161-176`
- **Narrative:** Static-covariate stats (`target_mu/sigma/max/trend`) are computed over the full frame, including the test period, when `stat_time_range` is omitted. The engine currently passes the correct range (`darts_forecaster.py:574-580`), but the dataset API defaults to leakage-with-warning rather than fail-loud — a structural mismatch with the fail-loud reproducibility gate that governs the rest of the pipeline. Leakage inflates apparent skill silently.
- **Cross-refs:** C-07, C-09, C-10 (silent-acceptance seams).

---

### C-06 — Production & experimental loss family (Spotlight/Prism/Sentinel + 5 variants) has zero dedicated tests

- **Tier:** 2 *(structural fragility: the production scoring function `SpotlightLoss` v37 ships with no behavioral/gradient/edge-case tests; a clear trigger exists at every training/sweep run)*
- **Source:** repo-assimilation (Phase 6 — test coverage)
- **Trigger:** A sweep or training run selects `SpotlightLoss`/`PrismLoss`/`SentinelLoss`, or someone edits the DRO / spectral-STFT / DC-AC decomposition math.
- **Location:** `views_r2darts2/math/spotlight_loss.py`, `prism_loss.py`, `sentinel_loss.py`, `spotlight_focal_loss.py`, `spotlight_loss_{asinh,huber,logcosh,power_law}.py`; no corresponding files under `tests/`.
- **Narrative:** ~2,000 LOC of the most mathematically intricate code (Barron base, KL-DRO weighting, multi-resolution STFT, DC/AC decomposition) has only catalog-instantiation coverage. Tier-1/2 losses each have a dedicated test file; Tier-3/4 advanced losses have none. A silent gradient or NaN regression in the production loss would not be caught by CI.
- **test-review (2026-06-10) corroboration:** Confirmed zero references to `spotlight`/`prism`/`sentinel` anywhere in `tests/`. The red-team test named `test_loss_numerical_purge_red_team_all_losses` (`tests/test_fortress_hardening.py:23-38`) is misnamed — it iterates only the **8 classic losses**, so the advanced family has no green (gradcheck/golden), beige, *or* red (NaN-fail-loud) coverage of any color.

---

### C-07 — Silent static-covariate drop for N-BEATS / N-HiTS / Transformer / TCN

- **Tier:** 2 *(config is accepted but silently not honored, producing a wrong mental model of what the model consumes and wasted fingerprint computation; structural fragility with a clear trigger)*
- **Source:** repo-assimilation (Phase 1 — catalogs)
- **Trigger:** Configuring `use_static_covariates=True` for any of these four architectures.
- **Location:** `views_r2darts2/catalogs/model_catalog.py:284-299` (NBEATS), `:301-317` (NHiTS), `:319-331` (TCN), `:348-363` (Transformer) — none pass the flag, unlike TFT `:280`, TSMixer `:259`, TiDE `:407`, BlockRNN `:344`, NLinear `:373`, DLinear `:385`.
- **Narrative:** Static-covariate fingerprints are computed and attached (cost plus a leakage surface) but never reach the model weights for these four architectures, and no warning is emitted at instantiation. Documented in `README.md:54-59` but not enforced in code. Partly a Darts library constraint, but the silent honoring of the config flag is the fragility.
- **test-review (2026-06-10) corroboration:** No test asserts a warning/raise when `use_static_covariates=True` is set on these four architectures (Leveson "inadequate-feedback" gap) — the silent drop is unguarded by the suite as well as by the code.
- **Cross-refs:** C-08, C-09, C-10 (silent-acceptance seams); C-11 (same catalog).

---

### C-09 — Feature-scaler config mismatch not validated on artifact load

- **Tier:** 2 *(silent train/eval scaler divergence → miscalibrated predictions with no error; structural fragility with a clear trigger)*
- **Source:** repo-assimilation (Phase 3 — engines)
- **Trigger:** Loading a saved artifact for evaluation after the `feature_scaler` config changed between the train run and the eval run.
- **Location:** `views_r2darts2/engines/darts_forecaster.py:874-875` (feature scaler silently overwritten) vs `:881-887` (target scaler config *is* checked and raises on mismatch).
- **Narrative:** Target-scaler config mismatch raises on load; the feature-scaler config has no equivalent guard and is silently overwritten, so an inconsistent feature scaler produces wrong-scale predictions without any error signal.
- **test-review (2026-06-10) corroboration:** No test exercises the artifact-load path with a changed `feature_scaler` config, so neither the (present) target-scaler guard nor the (absent) feature-scaler guard is regression-protected.
- **Cross-refs:** C-07, C-08, C-10 (silent-acceptance seams).

---

### C-10 — Probabilistic inverse-transform silently passes through unscaled on fitted-param miss

- **Tier:** 2 *(predictions returned on the wrong scale with no raise at the point of failure; structural fragility with a clear trigger)*
- **Source:** repo-assimilation (Phases 3 & 4 — engines, transformers)
- **Trigger:** A probabilistic (multi-sample) prediction run where a scaler's `_fitted_params` is missing/malformed or `param_idx` is out of bounds.
- **Location:** `views_r2darts2/engines/darts_forecaster.py:312-315`; `views_r2darts2/transformers/feature_scaler_manager.py:220-224`.
- **Narrative:** The 3D-reshape inverse path falls back to returning unscaled values (logging a warning) instead of raising. Raw z-scores can reach the output DataFrame; only the downstream NaN/range checks might catch it, and only if the unscaled values happen to violate those checks.
- **Cross-refs:** C-07, C-08, C-09 (silent-acceptance seams); distinct from resolved C-04 (different location and failure mode).

---

### C-11 — Catalog hub coupling to `ReproducibilityGate`; adding an architecture is a ≥3-site edit

- **Tier:** 3 *(maintainability/coupling: increases the cost and error-rate of change; affects multiple contributors)*
- **Source:** repo-assimilation (Phase 2 — dependency graph)
- **Trigger:** Adding a new model architecture, optimizer, scheduler, or loss.
- **Location:** `views_r2darts2/catalogs/model_catalog.py:95-106` (registry) + the `_get_*` factory + `views_r2darts2/infrastructure/reproducibility_gate.py:74-233` (genome registries); unknown algorithm names return `None` and crash with an opaque `TypeError` at `model_catalog.py:239`. **Live instance:** `MultiQueryTransformerModel` is registered with a 13-key genome at `reproducibility_gate.py:206-220` but has no `ModelCatalog` factory and no corresponding Darts class.
- **Narrative:** All registrations are hardcoded static dicts with no discovery mechanism, and the gate is imported by 12 files. A forgotten genome entry fails loud (good), but an unknown algorithm name returns `None` and crashes opaquely if the manifest audit is bypassed. Adding an architecture requires synchronized edits in at least three places.
- **repo-assimilation (2026-09-09) corroboration:** The registry desynchronisation is no longer hypothetical. `MultiQueryTransformerModel` appears *only* in `ALGORITHM_GENOMES` — nowhere else in the repo, and it is not a Darts class. A config naming it therefore **passes** `audit_manifest` (its 13 hyperparameters are validated as present and non-`None`), and only then does `get_model` execute `self.models.get(name)()` and raise `TypeError: 'NoneType' object is not callable`. The genome's presence actively signals to a reader that the architecture is supported.
- **Cross-refs:** C-07 (same catalog); C-32 (same catalog layer, dead cross-layer import).

---

### C-12 — Monkey-patch ordering invariant (RevIN before TiDE) is convention-only

- **Tier:** 3 *(latent fragility: correct today, but unguarded at runtime; no current silent-corruption, so not Tier 2)*
- **Source:** repo-assimilation (Phase 5 — infrastructure)
- **Trigger:** Calling individual `apply_*` patch functions out of order, or refactoring `apply_all_patches()`.
- **Location:** `views_r2darts2/infrastructure/patches.py:487-493`.
- **Narrative:** TiDE construction depends on RevIN already being patched; the order is guaranteed only by the sequential calls in `apply_all_patches()`, with no runtime assertion. Individual patches also lack exception handling, so a Darts-internal change would surface as an opaque crash. Pinning `darts==0.40.0` mitigates the version risk but not the ordering risk.

---

### C-13 — Fortress monitoring callbacks' fail-loud kill-switches are untested

- **Tier:** 2 *(a numerical-safety control with zero coverage: if the NaN/Inf-gradient halt silently stops working, training proceeds through numerical corruption undetected — structural fragility with a clear trigger)*
- **Source:** test-review (Nygard / Leveson lenses; CIC contract alignment)
- **Trigger:** Editing `callbacks.py` — e.g. refactoring `GradientHealthCallback`/`NaNDetectionCallback` thresholds or their `on_*` hooks — or upgrading the PyTorch-Lightning callback API.
- **Location:** `views_r2darts2/infrastructure/callbacks.py:95-130` (NaNDetectionCallback), `:137-274` (GradientHealthCallback); CIC `docs/CICs/fortress_monitoring_callbacks.md` §3 guarantees 0/3 tested.
- **Narrative:** Nine callbacks (~1,262 LOC) have no dedicated unit tests. Critically, the `trainer.should_stop=True` halt-on-NaN/Inf path — the Fortress's last-line numerical monitor — is never triggered by a test, so it could silently no-op after a refactor and let corrupt training runs proceed. The CIC's §10 also cites a non-existent `repro_phase1_gate.py` (stale reference).
- **Cross-refs:** C-12 (same infrastructure layer; patches/callbacks both lightly tested).

---

### C-14 — `_ViewsDatasetDarts` correctness guarantees (entity isolation, schema, index) untested

- **Tier:** 2 *(entity cross-contamination or silent acceptance of a malformed schema would corrupt per-country forecasts with no error signal; structural fragility with a clear trigger)*
- **Source:** test-review (Kleppmann lens; CIC contract alignment)
- **Trigger:** Modifying `as_darts_timeseries()` grouping/index logic, or feeding a DataFrame with missing DNA columns or a malformed MultiIndex.
- **Location:** `views_r2darts2/transformers/views_dataset_darts.py` (entity grouping `:304-311`, schema audit `:31-35`); CIC `docs/CICs/views_dataset_darts.md` §3 guarantees 1/4 tested.
- **Narrative:** No dedicated test verifies entity isolation (no cross-country contamination), multi-index semantic preservation, or a fail-loud raise on missing columns. Coverage is only indirect via reproducibility NaN-poisoning tests. Distinct from C-08, which is the `stat_time_range=None` leakage default in the same file. The CIC's §10 cites a non-existent `tests/contract_verification/verify_prediction_schema.py` (stale reference).
- **Cross-refs:** C-08 (same file, different failure mode).

---

### C-15 — Loss-family structural duplication and version archaeology

- **Tier:** 3 *(maintainability/coupling affecting multiple contributors; no current correctness impact, so not Tier 2)*
- **Source:** tech-debt-cleanup (survey)
- **Trigger:** Editing shared loss logic (`_spectral_loss`, `_log_cosh`, DRO weighting) and needing to apply the change consistently across the Spotlight/Prism family.
- **Location:** `views_r2darts2/math/spotlight_loss.py`, `spotlight_loss_{asinh,huber,logcosh,power_law}.py`, `spotlight_focal_loss.py`, `prism_loss.py`; stale version string at `spotlight_loss.py:11` (v37) vs `:179` (v36); vestigial `alpha` at `spotlight_loss.py:175`, `spotlight_loss_huber.py:66`, `spotlight_loss_power_law.py:41`; **false architecture claim** in the `spotlight_loss_huber.py` and `spotlight_loss_power_law.py` docstrings.
- **Narrative:** `_spectral_loss`/`_log_cosh`/`__repr__`/`__init__` are copy-pasted ~6× with no shared base class; docstrings carry embedded v35→v37 changelogs and a self-contradictory version label (class says v37, warning says v36). Competing versions (v36/v37/v46) coexist with no canonical marker. A fix to the shared math must be hand-applied to each copy. Refactor is blocked on C-06 (no tests).
- **graphify (2026-09-09) corroboration:** Graph extraction over the loss family produced **45 `semantically_similar_to` duplication edges** between the copy-pasted helpers, confirming the duplication is pervasive rather than incidental. It also surfaced a documentation defect not previously recorded: the `SpotlightLossHuber` and `SpotlightLossPowerLaw` docstrings both claim an *"identical architecture as SpotlightLossLogcosh ... KL-DRO"*, but `SpotlightLossLogcosh` is at v46 and uses **per-series sqrt DRO**, not KL-DRO. A reader trusting either docstring will form a wrong model of what the loss actually optimizes. Version strings v33/v35/v36/v37/v46 all coexist across the family with no canonical marker.
- **Cross-refs:** C-06 (same modules, testing dimension); C-33 (a stale replica in the test suite, same drift-between-copies root cause).

---

### C-16 — `apply_nbeats_patch` is disabled in `apply_all_patches` but still in the public API

- **Tier:** 3 *(footgun/maintainability: a caller can re-enable behavior the maintainers deliberately disabled — could change model outputs, but only via explicit opt-in, so not Tier 2)*
- **Source:** tech-debt-cleanup (survey)
- **Trigger:** A user or downstream pipeline calls `views_r2darts2.apply_nbeats_patch()` directly, having found it in `__all__`.
- **Location:** `views_r2darts2/infrastructure/patches.py:133-283` (151-LOC function), disabled at `:492`; exported at `views_r2darts2/__init__.py:8, 19`.
- **Narrative:** The function patches N-BEATS dropout forwarding but was intentionally commented out of `apply_all_patches`. It remains exported, so the public surface advertises a disabled, separately-maintained 151-LOC code path that no automated flow exercises — dead-but-callable code that can silently alter N-BEATS behavior if invoked.
- **Cross-refs:** C-12 (patch-layer fragility).

---

### C-17 — Non-executing test guards: dead file, uncollected nested test, assertion-free test, skipped suite

- **Tier:** 3 *(false-coverage signal: the suite reports green on guards that never execute or never assert; misleads refactoring decisions. Escalated in scope 2026-09-09 from a single dead file to a four-location pattern — tier held at 3 because the impact is a wrong belief about coverage, not a direct correctness path.)*
- **Source:** tech-debt-cleanup (survey) + test-review (CIC alignment) + graphify (2026-09-09, test-coverage edge extraction)
- **Trigger:** A developer relies on any of these four guards — or on the manager CIC §10 — believing the behaviour is regression-tested, before refactoring the production code beneath it.
- **Location:** `tests/test_model.py` (598 lines, 476 commented, **0 collected**), referenced in `docs/CICs/darts_forecasting_model_manager.md` §10; `tests/test_model_catalog.py` — `test_init_with_invalid_loss_raises_error` is **nested inside another test function**, so pytest never collects it; `tests/test_reproducibility_infra.py` — `test_stochastic_parity_serialization` contains **only a skip guard and asserts nothing**; `tests/test_loss.py` — `TestTweedieLoss` is `@pytest.mark.skip`'d as obsolete.
- **Narrative:** Four distinct mechanisms produce the same outcome: a guard that a reader (or a CIC) counts as coverage, but which contributes no assertion to any run. `test_model.py` imports and collects zero tests while the CIC presents it as "Green Team: lifecycle verification." The nested test in `test_model_catalog.py` is invisible to collection — the invalid-loss error path it was written to protect is therefore unguarded, and nothing in CI reports the omission. The assertion-free `test_stochastic_parity_serialization` passes unconditionally, so the stochastic-parity property it names is not verified. The skipped `TestTweedieLoss` is at least explicit. Together these inflate the apparent size of the suite while covering nothing, which is why the raw test-LOC figure (7,529 lines) overstates real protection.
- **graphify (2026-09-09) corroboration:** Surfaced during test→production coverage-edge extraction; the three additional locations were merged into this entry on the same date. The mock-heavy files (`test_forecaster.py`, `test_mc_dropout_entropy_lock.py`, `test_model_catalog.py`, `test_reproducibility_infra.py`, `test_scaling_robustness.py`) are a separate and weaker signal — they do assert, but against mocks — and are already covered by C-21's analysis of why real `model.fit()` cannot run in CI.
- **Cross-refs:** C-33 (a guard that runs but validates a stale replica — the fourth variant of the same false-coverage family); C-13, C-14, C-30 (real coverage gaps this false signal helps conceal); C-21 (structural cause of the mocking).

---

### C-18 — Patch layer validated against the wrong Darts version (pinned 0.40.0, dev/test on 0.38.0)

- **Tier:** 2 *(the monkey-patches target 0.40.0 internals; the passing suite runs on 0.38.0, so the patch layer's correctness against the pinned runtime is unverified — silent divergence between tested and shipped behavior)*
- **Source:** tech-debt-cleanup (survey) + test-review (F-7)
- **Trigger:** Running the test suite, or deploying, on the pinned `darts==0.40.0` after development/CI validation occurred on 0.38.0.
- **Location:** `pyproject.toml` (`darts = "=0.40.0"`) vs installed `0.38.0`; `views_r2darts2/infrastructure/patches.py:284-718` (RevIN/TiDE/TCN patches reaching into Darts internals).
- **Narrative:** The 450-green baseline was produced on 0.38.0. RevIN/TiDE/TCN patches reach into version-specific Darts internals; if 0.40.0's internals differ, a patch could fail to apply or misbehave, and the suite as currently run would not catch it. The pinned version that ships is the one *not* exercised by tests.
- **Cross-refs:** C-12 (patch ordering/fragility).

---

### C-19 — God-methods in engines/infrastructure reduce testability

- **Tier:** 3 *(maintainability: long methods mixing concerns are hard to unit-test in isolation, compounding the C-13/C-14 coverage gaps; no direct correctness impact)*
- **Source:** tech-debt-cleanup (survey)
- **Trigger:** Adding or modifying a step inside one of these methods (e.g. a new preprocessing stage in `train()` or a new audit in `audit_manifest`).
- **Location:** `views_r2darts2/engines/darts_forecaster.py` `__init__` (154L), `train` (150L), `predict` (121L); `views_r2darts2/infrastructure/callbacks.py` `on_train_epoch_end` (144L); `views_r2darts2/infrastructure/reproducibility_gate.py` `audit_manifest` (116L); `views_r2darts2/transformers/views_dataset_darts.py` `as_darts_timeseries` (113L).
- **Narrative:** Several methods exceed ~110 lines and bundle multiple responsibilities, making isolated characterization tests hard to write — part of why the C-13/C-14 gaps persist. Splitting is deferred (high-risk on partially-tested code) but the debt should be tracked.
- **Cross-refs:** C-13, C-14 (testability gaps these methods aggravate).

---

### C-20 — `_preprocess_timeseries` overwrites its own entity-alignment fix for `past_covariates`

- **Tier:** 2 *(structural fragility with a clear, already-logged trigger. **Escalates to Tier 1** if Darts 0.40's `fit()` does not validate `len(past_covariates) == len(series)` — in that case entity-misaligned covariate history reaches training with no error signal. See the verification note below.)*
- **Source:** repo-assimilation (2026-09-09) (Phases 3 & 5 — engines, invariant 14)
- **Trigger:** Any training run on a multivariate config (`dataset.features` non-empty) that logs `Training filter: N/M entities passed` with N < M — i.e. where at least one entity's *training slice* is shorter than `input_chunk_length + output_chunk_length`.
- **Location:** `views_r2darts2/engines/darts_forecaster.py:442-451` (aligned pairs built) vs `:463-469` (unconditional overwrite).
- **Narrative:** The `train_mode` branch builds `paired` `(target_slice, past_cov_slice)` tuples filtered together on target length, specifically to fix a documented entity-misalignment bug — the 15-line comment at `:427-441` explains that pairing by list index meant "every entity after the first filtered-out one was trained on the wrong covariate history." Twelve lines later, `if self.dataset.features:` rebuilds `past_cov` from the **full, unfiltered** `timeseries_float` and rebinds the name, discarding the filtered list from `:451`. In every multivariate run the alignment fix is dead code. `targets` then holds N elements while `past_cov` holds M > N, and `model.fit(series=targets, past_covariates=past_cov)` receives mismatched lists. Note that `audit_boundary_integrity` and `audit_sequence_contiguity` are applied to targets only, never to `past_covariates`, so no Fortress gate covers this path.
- **Verification note (open):** Whether this raises loudly or corrupts silently depends on whether `darts==0.40.0`'s `TorchForecastingModel.fit` validates that the two lists are the same length. `darts` is not installed on any environment on the assimilation machine and no virtualenv exists for this project, so this could not be determined. **If Darts validates:** training crashes whenever any short-history entity is filtered, and the documented fix is merely dead. **If it does not:** silent per-entity covariate misalignment → Tier 1. Resolve this question before scheduling the fix; a single working install settles it.
- **Cross-refs:** C-18 (the same missing-runtime problem blocks verification here); C-19 (`_preprocess_timeseries` sits inside the god-methods flagged there); silent-acceptance seam group (C-07–C-10).

---

### C-21 — `accelerator="gpu"` is hardcoded, making CPU/MPS training impossible and end-to-end training untestable in CI

- **Tier:** 2 *(structural fragility: a realistic and routine scenario — running on any non-CUDA host — fails at Trainer construction, and the same constraint structurally prevents CI from ever exercising a real `model.fit()`)*
- **Source:** repo-assimilation (2026-09-09) (Phases 3 & 6)
- **Trigger:** Running a real (non-mocked) training or sweep on a host without CUDA — a developer laptop, an MPS Mac, or the GitHub Actions runner — or adding a CI job intended to exercise `model.fit()` end-to-end.
- **Location:** `views_r2darts2/catalogs/model_catalog.py:167` (`"accelerator": "gpu"`), vs `views_r2darts2/engines/darts_forecaster.py:388-399` (`get_device()` supports `mps`/`cuda`/`cpu`) and `views_r2darts2/engines/darts_forecasting_model_manager.py:339-345` (branches on `forecaster.device == "cpu"`).
- **Narrative:** `_get_common_pl_trainer_kwargs` returns `"accelerator": "gpu"` unconditionally for every architecture. PyTorch Lightning raises `MisconfigurationException` at Trainer construction when no GPU is present, so training cannot start on a CPU or MPS host. Two parts of the codebase contradict this: `get_device()` explicitly supports `mps` and `cpu`, and the manager has a CPU-only parallel-prediction branch that a CPU-trained model could never reach. The consequence is not only portability — `.github/workflows/run_pytest.yml` runs on `ubuntu-latest` with no GPU, so **no test in the suite can call a real `model.fit()`**. This is the structural cause of the mock density in the training-path tests (`test_forecaster.py` alone carries 158 `Mock`/`patch` references) and therefore a root cause behind the C-13 / C-14 / C-20 coverage gaps.
- **Cross-refs:** C-13, C-14 (coverage gaps this constraint enforces); C-18 (both concern divergence between what is tested and what ships); C-20 (the alignment bug that an end-to-end training test would have caught).

---

### C-22 — Cyclic time encoders are injected twice, from two independent resolution-inference sites

- **Tier:** 2 *(the model silently receives each cyclic channel twice at doubled input width; two duplicated inference sites can diverge on a non-standard configuration, with no error signal — structural fragility with a clear trigger)*
- **Source:** repo-assimilation (2026-09-09) (Phases 3 & 4)
- **Trigger:** Setting `use_cyclic_encoders: True` in a sweep or model config; or changing `config['level']` or the dataset's `_time_id` such that the two resolution-inference sites can disagree.
- **Location:** `views_r2darts2/transformers/views_dataset_darts.py:118-129` (dataset columns) and `views_r2darts2/catalogs/model_catalog.py:222-231` (`_resolve_add_encoders`).
- **Narrative:** With the flag on, `as_darts_timeseries` appends `month_sin`/`month_cos` as real columns to `df_reset` and to `self.features`, so they enter the model as explicit `past_covariates`. Independently, `ModelCatalog._resolve_add_encoders` returns an `add_encoders` dict wiring the *same* callables into Darts' own encoder machinery for both `past` and `future`, plus `position: relative`. The model therefore receives each cyclic channel twice. The two sites also infer temporal resolution differently — the dataset from `self._time_id.split("_")[0][0]`, the catalog from `config["level"][-1]` — so they agree today only because both resolve to `"m"` for the current monthly configurations, and can silently diverge for a non-standard `level`/`time_id` pairing (e.g. a `cy`/`month_id` combination). Neither site logs the presence of the other.
- **Cross-refs:** C-23 (the downstream scaling consequence of the same `self.features` mutation); silent-acceptance seam group (C-07–C-10).

---

### C-23 — Dataset feature-list mutation makes cyclic-encoder scaling depend on which feature-scaler style is configured

- **Tier:** 2 *(the same config yields different model inputs depending on an unrelated configuration choice, with no log line or error marking the difference — structural fragility with a clear trigger)*
- **Source:** repo-assimilation (2026-09-09) (Phase 4)
- **Trigger:** Switching a config between the `feature_scaler_map` style and the single `feature_scaler` style while `use_cyclic_encoders: True`; or comparing sweep results across the two styles.
- **Location:** `views_r2darts2/transformers/views_dataset_darts.py:124-125` (mutation of `self.features`); `views_r2darts2/engines/darts_forecaster.py:170-178` (`FeatureScalerManager` constructed from `dataset.features`).
- **Narrative:** `as_darts_timeseries` mutates `self.features` by appending encoder column names, but `FeatureScalerManager` is constructed in `DartsForecaster.__init__` with `all_features=self.dataset.features` — *before* any `as_darts_timeseries` call. Under the `feature_scaler_map` path the encoder columns are therefore absent from both `_feature_to_scaler` and the `_assign_default_scaler` fallback group, and pass through unscaled. Under the plain single-`feature_scaler` path they are scaled along with every other component. Encoders are already in [-1, 1] so unscaled pass-through is arguably correct, but the divergence is undocumented, silent, and order-dependent. The mutation also means the feature set implied by a saved `.scalers` artifact is a function of call ordering rather than of the config.
- **Cross-refs:** C-22 (same `self.features` mutation, injection dimension); C-09 (adjacent scaler-divergence failure mode on the artifact-load path); silent-acceptance seam group.

---

### C-24 — `parallel_workers > 1` races on the process-global RNG that `lock_entropy` reseeds

- **Tier:** 2 *(the Entropy Guardian's documented "bit-perfect identity" guarantee silently does not hold under the configuration the code explicitly offers; structural fragility with a clear trigger)*
- **Source:** repo-assimilation (2026-09-09) (Phases 3 & 5, invariant 11)
- **Trigger:** Setting `parallel_workers > 1` in a config for a CPU evaluation run; or removing/altering the GPU sequential-forcing branch at `darts_forecasting_model_manager.py:341-345`.
- **Location:** `views_r2darts2/engines/darts_forecasting_model_manager.py:339-372`; `views_r2darts2/engines/darts_forecaster.py:741` (and `:423`); `views_r2darts2/infrastructure/reproducibility_gate.py:695-708`.
- **Narrative:** `_evaluate_model_artifact` submits every rolling-origin sequence to a `ThreadPoolExecutor` sharing one `forecaster` and one model instance. Each `predict()` then calls `ReproducibilityGate.Data.lock_entropy(random_state)`, which reseeds `random`, `numpy.random`, and `torch` — all **process-global**. With more than one worker these reseeds interleave with other threads' sampling, so the bit-perfect reproducibility the `lock_entropy` docstring promises does not hold for MC-dropout or any probabilistic forecast. The GPU path is forced sequential, but only with a comment about Darts device-shifting race conditions; the reproducibility hazard on the CPU branch is documented nowhere. `self.min_length` (`darts_forecaster.py:423`) is likewise written on the shared instance from every thread.
- **Cross-refs:** C-21 (the CPU branch is currently unreachable for a locally-trained model, which masks but does not remove this); silent-acceptance seam group.

---

### C-25 — Production infrastructure imports the test package

- **Tier:** 3 *(dependency-direction violation that makes installed-library behaviour depend on whether a `tests` module happens to be importable; contrary to ADR-002's first topological invariant)*
- **Source:** repo-assimilation (2026-09-09) (Phase 2 — dependency graph)
- **Trigger:** Installing the published wheel into an environment where an unrelated `tests` package is importable; or adding a `lint-imports` / import-linter contract to enforce ADR-002.
- **Location:** `views_r2darts2/infrastructure/patches.py:32` (`from tests.conftest import CLEAN_TORCH_LOAD`), reached from `darts_forecasting_model_manager.py:55` via `apply_all_patches()`.
- **Narrative:** `apply_torch_load_patch` attempts `from tests.conftest import CLEAN_TORCH_LOAD` inside a `try/except (ImportError, ModuleNotFoundError)`. The intent is defensive — avoiding capture of a `Mock` during test runs — but it inverts the dependency direction, with shipped infrastructure reaching upward into the test package. ADR-002 states "Upward Imports are Forbidden" as its first topological invariant. Because the import is opportunistic, the installed library resolves `torch.__original_load__` differently depending on whether *any* importable `tests.conftest` exposing that name exists in the consumer's process.
- **Cross-refs:** C-26 (ADR-002 cannot currently be checked mechanically); C-12, C-16, C-28 (patch-layer concerns).

---

### C-26 — ADR-002's layer map names four directories that do not exist in the package

- **Tier:** 3 *(the ADR governing "who may depend on whom" is not mechanically checkable against the tree, raising the cost and error-rate of every topology review)*
- **Source:** repo-assimilation (2026-09-09) (Phases 1 & 2)
- **Trigger:** A contributor or reviewer checks a proposed import against ADR-002's layer rules; or someone attempts to derive an import-linter contract from the ADR.
- **Location:** `docs/ADRs/002_topology_and_dependancy_rules.md` (Layer 0 `utils/`, Layer 1 `data/`, Layer 2 `model/`, Layer 3 `manager/`) vs the actual package (`catalogs/`, `transformers/`, `engines/`, `infrastructure/`, `math/`).
- **Narrative:** The ADR that defines the dependency hierarchy describes a directory layout the repository does not have — the same stale module names that `tests/test_model.py`'s commented-out imports still reference (`views_r2darts2.model.forecaster`, `views_r2darts2.model.catalog`). Its examples also place `loss_catalog.py` and `reproducibility_gate.py` together in Layer 0 despite the catalogs now depending on the gate. No `lint-imports` contract or equivalent exists in the repo, so the rule is enforced by neither tooling nor a readable mapping. The underlying dependency graph is in fact a clean DAG with no cycles — the defect is in the governing document, not the code.
- **Cross-refs:** C-25 (a live violation the ADR cannot currently adjudicate); C-17 (stale governance references); C-27 (governance drift).

---

### C-27 — `ADR_COMPLIANCE_REPORT.md` asserts total compliance while the register records 14 open concerns

- **Tier:** 3 *(false-status signal at the level of the whole governance set: an unqualified "100% compliant" claim in `docs/` misleads orientation and refactoring decisions)*
- **Source:** repo-assimilation (2026-09-09) (Phases 1 & 6)
- **Trigger:** A new contributor or external reviewer orients via `docs/` and treats `ADR_COMPLIANCE_REPORT.md` as a current status statement.
- **Location:** `docs/ADR_COMPLIANCE_REPORT.md` (dated 2026-02-16) vs `reports/technical_risk_register.md` (dated 2026-06-10).
- **Narrative:** The report declares "**Peak Fortress State** … 100% compliant with the scientific integrity mandates defined in ADRs 000-013" and "411 tests passing", with no status qualifier, expiry, or scope caveat. It is nearly seven months stale and is contradicted by the register one directory away, which at the time of writing carried a Tier-1 open concern (C-08) and thirteen others — several of them, including C-07 and C-11, against precisely the ADRs the report certifies as compliant (ADR-003, ADR-009, ADR-013). Same false-signal family as C-17, but at the level of the whole documentation set rather than one CIC's coverage claim.
- **Cross-refs:** C-17 (false-coverage signal, CIC scope); C-26 (governance drift).

---

### C-28 — `torch.load` is monkey-patched process-wide to `weights_only=False`

- **Tier:** 3 *(a global default weakened for every consumer in the process, not just this package's own artifact loads; correctness impact is nil today, so not Tier 2, but the blast radius is out of proportion to the need)*
- **Source:** repo-assimilation (2026-09-09) (Phases 2 & 3)
- **Trigger:** Another package in the same pipeline process calls `torch.load` on externally-supplied or untrusted weights after `DartsForecastingModelManager` has been constructed.
- **Location:** `views_r2darts2/infrastructure/patches.py:22-48`, invoked unconditionally from `views_r2darts2/engines/darts_forecasting_model_manager.py:55`.
- **Narrative:** `apply_torch_load_patch` rebinds the global `torch.load` so that **every** deserialization in the process defaults to `weights_only=False` — including calls made by `views-pipeline-core`, other model packages, or third-party libraries sharing the interpreter. The stated need is real and narrow: this package's `.scalers` sidecar holds live sklearn/Darts objects that `weights_only=True` refuses. The narrower fix — passing `weights_only=False` explicitly at the two load sites in `darts_forecaster.load_model` — is not attempted, and the override is silent to other consumers. The `Mock`-avoidance branch also couples this to C-25.
- **Cross-refs:** C-25 (same function, test-package import); C-12, C-16 (patch-layer fragility).

---

### C-29 — Artifact timestamp is extracted by a fixed 15-character slice with no validation

- **Tier:** 3 *(a silently-wrong value in an ADR-governed cross-repo contract, but reachable only via a caller-supplied non-conforming artifact name, so not Tier 2)*
- **Source:** repo-assimilation (2026-09-09) (Phases 3 & 5, invariant 17)
- **Trigger:** Passing `--artifact_name` for a file whose stem does not end in the generated 15-character `YYYYMMDD_HHMMSS` timestamp — e.g. a hand-renamed, copied, or externally-produced artifact.
- **Location:** `views_r2darts2/engines/darts_forecasting_model_manager.py:269` and `:406` (`timestamp = path_artifact.stem[-15:]`).
- **Narrative:** The extracted string is written straight into the config via `_config_manager.add_config({"timestamp": timestamp})` and thereby into every prediction filename. ADR-015 documents this implementation as the *correct* discharge of the cross-repo artifact-prediction timestamp contract (views-pipeline-core ADR-052), and it is correct for names produced by `generate_model_file_name`. But there is no length check, no format check, and no error path on the read side, while `artifact_name` is explicitly caller-supplied (`_evaluate_model_artifact(eval_type, artifact_name=None)`). ADR-015 itself states that a timestamp mismatch causes "silent fallback to subprocess re-execution or failure" in the ensemble manager — so the failure mode is documented while the guard is absent. The contract is enforced by convention at the write site and by nothing at the read site.
- **Cross-refs:** ADR-015 (the governing contract); silent-acceptance seam group.

---

### C-30 — Untested infrastructure outside the existing coverage entries: scheduler catalog, cyclic encoders, custom LR schedulers

- **Tier:** 3 *(three modules performing non-trivial config translation and phase arithmetic with zero test references; increases the cost and risk of change for multiple contributors, but no identified silent-corruption path, so not Tier 2)*
- **Source:** repo-assimilation (2026-09-09) (Phase 6 — test coverage)
- **Trigger:** Selecting a non-default `lr_scheduler_cls` (especially `WarmupCAWR`); adding an entry to `SchedulerCatalog._KWARG_MAP` or `_STATIC_KWARGS`; or changing the `(idx - 1) % period` phase convention in `encoders.py`.
- **Location:** `views_r2darts2/catalogs/scheduler_catalog.py` (125 LOC), `views_r2darts2/infrastructure/encoders.py` (115 LOC), `views_r2darts2/math/warmup_cawr.py` (106 LOC) — zero references to any of them anywhere under `tests/`.
- **Narrative:** C-06, C-13, and C-14 cover the untested loss family, callbacks, and dataset respectively; these three modules fall outside all three. `SchedulerCatalog` performs non-trivial work that is entirely unverified: config-key→torch-kwarg remapping via `_KWARG_MAP`, pass-through of the nested `lr_scheduler_kwargs` block, lazy loading of `_CUSTOM_SCHEDULERS`, and last-wins injection of `_STATIC_KWARGS` that deliberately overrides config values (`scheduler_catalog.py:121-123`). `encoders.py` encodes a phase convention — `(idx - 1) % period`, with `month_id` 1 = January 1980 — whose off-by-one would shift every seasonal signal by one month with no visible symptom in any metric.
- **graphify (2026-09-09) note:** Graph clustering linked this concern to `5.0 Integrity Harness — tests/losses/harness.py replicating the (B,T,K) scaling pipeline` in `docs/roadmap/FORTRESS_ROADMAP.md`. The roadmap already specifies a harness of the shape this gap needs; the work may be partly designed rather than wholly new.
- **Cross-refs:** C-06, C-13, C-14 (adjacent coverage gaps); C-22 (encoders' second consumer); C-21 (the structural reason integration coverage is hard here).

---

### C-31 — `math/warmup_cosine.py` is unreachable dead code

- **Tier:** 4 *(localized: 100 LOC that cannot be selected by any config; misleads a reader but affects no running path)*
- **Source:** repo-assimilation (2026-09-09) (Phase 2 — dependency graph)
- **Trigger:** A contributor reads `math/warmup_cosine.py` and configures `lr_scheduler_cls: "WarmupCosine"`; or adds it to `_CUSTOM_SCHEDULERS` without a matching `SCHEDULER_GENOMES` entry.
- **Location:** `views_r2darts2/math/warmup_cosine.py` (100 LOC).
- **Narrative:** The module has zero importers anywhere in the repo. It is absent from `math/__init__.py`'s 17-entry `__all__`, from `SchedulerCatalog._CUSTOM_SCHEDULERS` (which registers only `WarmupCAWR`), and from `ReproducibilityGate.Config.SCHEDULER_GENOMES`. Configuring `lr_scheduler_cls: "WarmupCosine"` therefore fails the Fortress whitelist check before the class is ever reached — correctly, but only by accident of non-registration. Distinct from C-16, which concerns dead-but-*callable* exported code; this is dead-and-unreachable.
- **Cross-refs:** C-16 (dead patch code, callable variant); C-30 (its sibling `warmup_cawr.py` is reachable but untested).

---

### C-32 — Dead state holds open the only `catalogs → engines` import; `SchedulerCatalog` missing from the public API

- **Tier:** 4 *(two localized asymmetries in one layer; no correctness or reliability impact)*
- **Source:** repo-assimilation (2026-09-09) (Phase 2 — dependency graph)
- **Trigger:** Removing the `catalogs → engines` edge to satisfy a topology contract; or a downstream consumer attempting `from views_r2darts2 import SchedulerCatalog`.
- **Location:** `views_r2darts2/catalogs/model_catalog.py:30` (import) and `:108` (`self.device = DartsForecaster.get_device()`, never read); `views_r2darts2/__init__.py:10-21` (`__all__`).
- **Narrative:** `catalogs/model_catalog.py` imports `engines/darts_forecaster` for exactly one line — `self.device = DartsForecaster.get_device()` — and `self.device` is never referenced again in the class. Deleting that line removes the sole catalogs→engines edge, simplifying the dependency graph at zero behavioural cost. Separately, `LossCatalog`, `OptimizerCatalog`, and `ModelCatalog` are all re-exported from `views_r2darts2/__init__.py`; `SchedulerCatalog`, added later, is not — so four sibling catalogs present two different public faces, and the newest is reachable only by its full internal path.
- **Cross-refs:** C-11 (same catalog layer, registry coupling); C-26 (the topology this edge would be checked against).

---

### C-33 — `test_rinorm_curvature_patch.py` validates a local replica that has diverged from the production RevIN patch

- **Tier:** 2 *(a 288-line guard over the repo's most mathematically intricate production code asserts against a copy whose formula no longer matches what ships; the suite reports green while the real patch is unverified — structural fragility with a clear trigger)*
- **Source:** graphify (2026-09-09) (semantic extraction, test→production coverage edges)
- **Trigger:** Editing `apply_rinorm_compression_patch()` in `patches.py` — changing the centering space, the σ clamp, the ±50 pre-`sinh` clamp, or the likelihood scale-parameter branch — and relying on `test_rinorm_curvature_patch.py` passing as evidence the change is safe.
- **Location:** `tests/test_rinorm_curvature_patch.py` (288 lines, class `FakeRawSpaceRINorm`, imports no production module); `views_r2darts2/infrastructure/patches.py:284-407` (`apply_rinorm_compression_patch`, hybrid-space v3).
- **Narrative:** The test file imports nothing from `views_r2darts2`. It defines its own `FakeRawSpaceRINorm` and asserts curvature, Jensen-bias, and z-range properties against that local class. The replica implements **pure raw-space RevIN** — the v2 formula — while production has since moved to **hybrid-space v3** (`z = asinh(sinh(x − μ_asinh) / σ_c)`, centering in asinh space and normalizing variance in raw space). `patches.py:162-198` documents v2's systematic positive mean bias as the explicit reason v3 replaced it. The test therefore passes by validating the exact behaviour the production code was rewritten to abandon: a green run is evidence about a formula that no longer ships. This is the most consequential instance of the C-17 false-coverage family, because RevIN sits directly on the inverse path from model output to raw counts, and because C-18 already establishes that the patch layer is unverified against the pinned Darts version.
- **Cross-refs:** C-17 (same false-coverage family — this is the "runs but proves nothing" variant); C-12, C-18 (patch-layer fragility and version divergence); C-15 (drift between copies, same root cause in the loss family).

---

### C-34 — `DartsForecaster.get_device()` mutates global torch state from inside a getter

- **Tier:** 4 *(a process-wide default-dtype mutation triggered as a side effect of a device query; latent today because the codebase is float32 throughout and the MPS branch is unreachable while C-21 stands)*
- **Source:** graphify (2026-09-09) (surfaced via the archived reproducibility-investigation cluster, confirmed against source)
- **Trigger:** Running on an MPS host after C-21 is fixed, in a process where another library depends on `torch.get_default_dtype()` being float64 — or calling `ModelCatalog(config)` purely to inspect the catalog, which invokes `get_device()` as a side effect.
- **Location:** `views_r2darts2/engines/darts_forecaster.py:388-399` (`torch.set_default_dtype(torch.float32)` inside the MPS branch); called from `darts_forecaster.py:183` and `views_r2darts2/catalogs/model_catalog.py:108`.
- **Narrative:** `get_device()` is a `@staticmethod` whose name and docstring promise a pure query — "Returns the device type for model training" — but the MPS branch calls `torch.set_default_dtype(torch.float32)`, a process-global mutation, before returning. Any caller that merely asks which device is available silently changes global tensor defaults for every other library in the interpreter. The effect is benign in this repo (ADR-010 mandates float32 everywhere), and the MPS branch cannot currently be reached during training because C-21 hardcodes a GPU accelerator — but the side effect is invisible at the call site and one of those two mitigations is a defect rather than a design. `ModelCatalog.__init__` calls it for a value it never reads (C-32), so a catalog instantiation alone can trip it.
- **Cross-refs:** C-28 (same pattern: process-global torch mutation as a side effect); C-32 (the dead `self.device` call site); C-21 (currently masks the MPS path).

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
