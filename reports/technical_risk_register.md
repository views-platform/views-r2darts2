# Technical Risk Register

| Register Info     | Details                              |
|-------------------|--------------------------------------|
| Project           | views-r2darts2                       |
| Owner             | Simon Polichinel von der Maase       |
| Last Updated      | 2026-09-10                           |
| Total Concerns    | 43                                   |
| Open Concerns     | 31                                   |
| Resolved Concerns | 12                                   |
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

> **Re-derived 2026-09-10 against `development` @ `fe7e681` (0.2.x).** Every open entry below was
> re-verified: 11 still present (line numbers refreshed), 2 moved, 9 changed (C-08 and C-22
> re-tiered down; C-12, C-18, C-32 retitled), 7 resolved (five by the rewrite, two by the
> `governance-0.2.x` branch), 9 registered (C-35..C-43), 5 disagreements opened (D-01..D-05).
> Two sub-claims are UNVERIFIED pending a `darts==0.46.1` install: C-12 (`_Block` guard collision)
> and C-18 (patch behaviour against 0.46.1 internals).
>
> The "silent-acceptance seam" concerns (C-07, C-09, C-10, C-24, C-29, plus latent C-08) share a
> root theme: seams where the otherwise fail-loud pipeline silently accepts a wrong, leaky, or
> stale configuration. A `review-rr` strategic pass may wish to group these into a formal causal
> cluster — and C-15 / C-17 / C-33 into a second ("a copy drifting from its original with no
> detector").

---

### C-08 — `stat_time_range=None` in `compute_static_covariates` warns instead of failing — now on a dead path

- **Tier:** 3 *(**re-tiered 1 → 3 on 2026-09-10.** The leakage-with-warning seam still exists in code, but `stat_time_range` is now a required keyword — a caller must pass `None` explicitly — and `compute_static_covariates` has zero production callers, so no silent-corruption path is reachable today. It is a latent ADR-003 violation in a dead module.)*
- **Source:** repo-assimilation (Phase 4 — transformers); re-derived 2026-09-10
- **Trigger:** Wiring `compute_static_covariates` back into `ViewsDataset.to_darts_timeseries` (the natural fix for C-36) without first changing the `stat_time_range=None` branch from warn to raise.
- **Location:** `views_r2darts2/transformers/static_covariates.py:140` (required kwarg), `:161-182` (full-frame branch, warning at `:178-182`); the only production static covariate on 0.2.x is the entity id, attached at `views_r2darts2/dataset/base.py:1706-1709`.
- **Narrative:** When `stat_time_range=None` is passed, the fingerprint stats are computed over the full frame, including the test period, with a warning rather than a raise — a structural mismatch with the fail-loud gate that governs the rest of the pipeline. On 0.1.x this sat on the live path with a default of `None`. On 0.2.x the parameter is mandatory and the function is never called in production (C-36). The seam is intact; the exposure is zero until someone reconnects the module.
- **Cross-refs:** C-36 (the dead module this lives in — fixing that one re-arms this one); C-07, C-09, C-10 (silent-acceptance seams).

---

### C-06 — Production & experimental loss family (Spotlight/Prism/Sentinel + 5 variants) has zero dedicated tests

- **Tier:** 2 *(structural fragility: the production scoring function `SpotlightLoss` v37 ships with no behavioral/gradient/edge-case tests; a clear trigger exists at every training/sweep run)*
- **Source:** repo-assimilation (Phase 6 — test coverage)
- **Trigger:** A sweep or training run selects `SpotlightLoss`/`PrismLoss`/`SentinelLoss`, or someone edits the DRO / spectral-STFT / DC-AC decomposition math.
- **Location:** `views_r2darts2/math/spotlight_loss.py` (288L), `prism_loss.py` (384L), `sentinel_loss.py` (228L), `spotlight_focal_loss.py` (235L), `spotlight_loss_asinh.py` (363L), `spotlight_loss_huber.py` (152L), `spotlight_loss_logcosh.py` (280L), `spotlight_loss_power_law.py` (277L) = **2,207 LOC**; plus `charbonnier_loss.py` (39L) and `logcosh_loss.py` (56L). Zero references under `tests/`. All selectable via `views_r2darts2/catalogs/loss_catalog.py:74-82` and `views_r2darts2/infrastructure/reproducibility_gate.py:352-360`.
- **Narrative:** ~2,000 LOC of the most mathematically intricate code (Barron base, KL-DRO weighting, multi-resolution STFT, DC/AC decomposition) has only catalog-instantiation coverage. Tier-1/2 losses each have a dedicated test file; Tier-3/4 advanced losses have none. A silent gradient or NaN regression in the production loss would not be caught by CI.
- **test-review (2026-06-10) corroboration:** Confirmed zero references to `spotlight`/`prism`/`sentinel` anywhere in `tests/`. The red-team test that iterated only the **8 classic losses** lived in a file since deleted; its live equivalent is the `LOSS_SPECS` list in `tests/losses/test_mathematical_integrity.py:11-56`, which still parametrizes exactly those eight, so the advanced family has no green (gradcheck/golden), beige, *or* red (NaN-fail-loud) coverage of any color.
- **re-derivation (2026-09-10, `development` @ `fe7e681`):** **Still present; scope grew.** All eight family modules exist unchanged in coverage; `charbonnier_loss.py` and `logcosh_loss.py` are now also selectable and untested.

---

### C-07 — Silent static-covariate drop for N-BEATS / N-HiTS / Transformer / TCN

- **Tier:** 2 *(config is accepted but silently not honored, producing a wrong mental model of what the model consumes and wasted fingerprint computation; structural fragility with a clear trigger)*
- **Source:** repo-assimilation (Phase 1 — catalogs)
- **Trigger:** Configuring `use_static_covariates=True` for any of these four architectures.
- **Location:** `views_r2darts2/catalogs/model_catalog.py:320-335` (NBEATS), `:337-353` (NHiTS), `:355-367` (TCN), `:384-399` (Transformer) — none pass the flag, unlike TSMixer `:295`, TFT `:316`, BlockRNN `:380`, NLinear `:409`, DLinear `:421`, TiDE `:443`.
- **Narrative:** The `use_static_covariates` flag is accepted for these four architectures but never reaches the constructor, and no warning is emitted at instantiation. *(The 2026-06 clause about wasted fingerprint computation and a leakage surface no longer applies — on 0.2.x the fingerprint is not computed at all; see C-36.)* Documented in `README.md:54-59` but not enforced in code. Partly a Darts library constraint, but the silent honoring of the config flag is the fragility.
- **test-review (2026-06-10) corroboration:** No test asserts a warning/raise when `use_static_covariates=True` is set on these four architectures (Leveson "inadequate-feedback" gap) — the silent drop is unguarded by the suite as well as by the code.
- **re-derivation (2026-09-10, `development` @ `fe7e681`):** **Still present**, line numbers refreshed. `README.md:54-59` still documents the drop without enforcing it.
- **Cross-refs:** C-08, C-09, C-10 (silent-acceptance seams); C-11 (same catalog).

---

### C-09 — Feature-scaler config mismatch not validated on artifact load

- **Tier:** 2 *(silent train/eval scaler divergence → miscalibrated predictions with no error; structural fragility with a clear trigger)*
- **Source:** repo-assimilation (Phase 3 — engines)
- **Trigger:** Loading a saved artifact for evaluation after the `feature_scaler` config changed between the train run and the eval run.
- **Location:** `views_r2darts2/engines/darts_forecaster.py:674` (feature scaler restored with no check) vs `:681-686` (target-scaler config *is* checked and raises on mismatch); `feature_scaler_cfg` / `feature_scaler_map_cfg` are persisted at `:660-661` and never read back.
- **Narrative:** Target-scaler config mismatch raises on load; the feature-scaler config has no equivalent guard and is silently overwritten, so an inconsistent feature scaler produces wrong-scale predictions without any error signal.
- **test-review (2026-06-10) corroboration:** No test exercises the artifact-load path with a changed `feature_scaler` config, so neither the (present) target-scaler guard nor the (absent) feature-scaler guard is regression-protected.
- **re-derivation (2026-09-10, `development` @ `fe7e681`):** **Still present.** The config strings needed for the check are already in the artifact; only the comparison is missing. Still no test exercises the load path with a changed feature-scaler config.
- **Cross-refs:** C-07, C-08, C-10 (silent-acceptance seams).

---

### C-10 — Probabilistic inverse-transform silently passes through unscaled on fitted-param miss

- **Tier:** 2 *(predictions returned on the wrong scale with no raise at the point of failure; structural fragility with a clear trigger. **Escalation candidate:** on 0.2.x the fallback no longer logs at all — the only thing keeping this from Tier 1 is that the trigger requires a Darts `_fitted_params` layout the extractor does not recognise, which is C-18 territory.)*
- **Source:** repo-assimilation (Phases 3 & 4 — engines, transformers)
- **Trigger:** A probabilistic (multi-sample) prediction run where a scaler's `_fitted_params` is missing/malformed or `param_idx` is out of bounds.
- **Location:** `views_r2darts2/transformers/inverse.py:143` (`# Last resort: passthrough`, 3-D probabilistic) and `:179` (2-D deterministic); reached from `FeatureScalerManager` and `ViewsDataset._inverse_transform_numpy_predictions` (`views_r2darts2/dataset/base.py:1459`).
- **Narrative:** Both inverse paths fall back to returning unscaled values **with no log line** — the 0.1.x code at least emitted a warning; the unification into `inverse.py` dropped it — instead of raising. Raw z-scores can reach the output DataFrame; only the downstream NaN/range checks might catch it, and only if the unscaled values happen to violate those checks.
- **re-derivation (2026-09-10, `development` @ `fe7e681`):** **Moved and worsened.** The two 0.1.x copies were unified into `transformers/inverse.py` (good) and the warning was lost in the process (bad). `extract_fitted_sklearn_scaler` tolerates five `_fitted_params` layouts; a sixth returns `None` and the caller silently passes through.
- **Cross-refs:** C-07, C-08, C-09 (silent-acceptance seams); distinct from resolved C-04 (different location and failure mode).

---

### C-11 — Catalog hub coupling to `ReproducibilityGate`; adding an architecture is a ≥3-site edit

- **Tier:** 3 *(maintainability/coupling: increases the cost and error-rate of change; affects multiple contributors)*
- **Source:** repo-assimilation (Phase 2 — dependency graph)
- **Trigger:** Adding a new model architecture, optimizer, scheduler, or loss.
- **Location:** `views_r2darts2/catalogs/model_catalog.py:104-114` (registry, 10 entries) + the `_get_*` factories + `views_r2darts2/infrastructure/reproducibility_gate.py` genome registries; unknown algorithm names return `None` and crash with an opaque `TypeError` at `model_catalog.py:275`. **Live instance:** `MultiQueryTransformerModel` is registered with a 13-key genome at `reproducibility_gate.py:249` but has no `ModelCatalog` factory and no corresponding Darts class.
- **Narrative:** All registrations are hardcoded static dicts with no discovery mechanism, and the gate is imported by 8 production modules. A forgotten genome entry fails loud (good), but an unknown algorithm name returns `None` and crashes opaquely if the manifest audit is bypassed. Adding an architecture requires synchronized edits in at least three places.
- **repo-assimilation (2026-09-09) corroboration:** The registry desynchronisation is no longer hypothetical. `MultiQueryTransformerModel` appears *only* in `ALGORITHM_GENOMES` — nowhere else in the repo, and it is not a Darts class. A config naming it therefore **passes** `audit_manifest` (its 13 hyperparameters are validated as present and non-`None`), and only then does `get_model` execute `self.models.get(name)()` and raise `TypeError: 'NoneType' object is not callable`. The genome's presence actively signals to a reader that the architecture is supported.
- **re-derivation (2026-09-10, `development` @ `fe7e681`):** **Still present**, including the phantom `MultiQueryTransformerModel`. Now four catalogs (`SchedulerCatalog` added) each holding a registry parallel to the gate's.
- **Cross-refs:** C-07 (same catalog); C-32 (same catalog layer, dead cross-layer import).

---

### C-12 — Monkey-patch ordering, forward-method overwrites, and shared guard attributes are convention-only

- **Tier:** 3 *(latent fragility: correct today, but unguarded at runtime; no current silent-corruption, so not Tier 2)*
- **Source:** repo-assimilation (Phase 5 — infrastructure)
- **Trigger:** Calling individual `apply_*` patch functions in a different order than `apply_all_patches()`; re-enabling `apply_tide_skip_layernorm_patch` (C-38); or bumping `darts` past `0.46.1`.
- **Location:** `views_r2darts2/infrastructure/patches.py:938-946` (`apply_all_patches`, no docstring); `_TideModule.forward` set at `:603` by `apply_tide_mc_dropout_patch` and would be overwritten at `:793` by the disabled `apply_tide_skip_layernorm_patch`; `apply_nhits_layernorm_patch:854` and `apply_nbeats_layernorm_patch:917` share the guard attribute name `_Block._views_ln_patch`; `patches.py:3` docstring says "Darts 0.45 internals" against a `==0.46.1` pin.
- **Narrative:** The original RevIN→TiDE dependency is no longer visible in code (`apply_tide_mc_dropout_patch` `:566-611` never touches RINorm). The same shape of hazard has moved: two patches assign the same `forward`, and two patches share one guard attribute — safe only if the N-HiTS and N-BEATS `_Block` classes are unrelated, which could not be verified here (`darts` not installed). There is still no runtime assertion and no per-patch exception handling, so a Darts-internal change surfaces as an opaque crash or a silent no-op. The version pin moved from `0.40.0` to `0.46.1` while the module docstring still names `0.45`.
- **re-derivation (2026-09-10, `development` @ `fe7e681`):** **Changed.** Original ordering dependency gone; two new same-shape hazards, one UNVERIFIED pending a `darts==0.46.1` install.

---

### C-13 — Fortress monitoring callbacks' fail-loud kill-switches are untested

- **Tier:** 2 *(a numerical-safety control with zero coverage: if the NaN/Inf-gradient halt silently stops working, training proceeds through numerical corruption undetected — structural fragility with a clear trigger)*
- **Source:** test-review (Nygard / Leveson lenses; CIC contract alignment)
- **Trigger:** Editing `callbacks.py` — e.g. refactoring `GradientHealthCallback`/`NaNDetectionCallback` thresholds or their `on_*` hooks — or upgrading the PyTorch-Lightning callback API.
- **Location:** `views_r2darts2/infrastructure/callbacks.py:96-136` (`NaNDetectionCallback`, `should_stop = True` at `:128`), `:138-280` (`GradientHealthCallback`, `should_stop = True` at `:274`); the file is now **1,992 LOC / 15 callback classes**. `git grep -l 'callbacks\|NaNDetection\|GradientHealth' development -- tests/` → zero hits.
- **Narrative:** Fifteen callbacks (1,992 LOC) have no dedicated unit tests. Critically, the `trainer.should_stop=True` halt-on-NaN/Inf path — the Fortress's last-line numerical monitor — is never triggered by a test, so it could silently no-op after a refactor and let corrupt training runs proceed. *(The CIC's §10 was corrected 2026-09-10 to say "None".)*
- **re-derivation (2026-09-10, `development` @ `fe7e681`):** **Still present, worse** — the file grew ~58% and gained six callbacks; coverage stayed at zero.
- **Cross-refs:** C-12 (same infrastructure layer; patches/callbacks both lightly tested).

---

### C-15 — Loss-family structural duplication and version archaeology

- **Tier:** 3 *(maintainability/coupling affecting multiple contributors; no current correctness impact, so not Tier 2)*
- **Source:** tech-debt-cleanup (survey)
- **Trigger:** Editing shared loss logic (`_spectral_loss`, `_log_cosh`, DRO weighting) and needing to apply the change consistently across the Spotlight/Prism family.
- **Location:** `_log_cosh` copies at `views_r2darts2/math/spotlight_loss.py:46`, `spotlight_focal_loss.py:127`, `spotlight_loss_logcosh.py:37`, `spotlight_loss_power_law.py:75`, `prism_loss.py:154`; `_spectral_loss` copies at `spotlight_loss.py:137`, `spotlight_loss_asinh.py:197`, `spotlight_focal_loss.py:132`, `spotlight_loss_power_law.py:80`, `prism_loss.py:163`; **false architecture claim** at `spotlight_loss_power_law.py:11-14` ("identical … KL-DRO" while its own DRO at `:185-196` is CV-based); version archaeology at `prism_loss.py:11,:99,:145` (v36/v33) and `spotlight_loss_asinh.py:39` (v49b) vs `:353` (v49) vs `:268,:271` (v49a-c/v47).
- **Narrative:** `_spectral_loss`/`_log_cosh`/`__repr__`/`__init__` are copy-pasted ~6× with no shared base class; docstrings carry embedded v35→v37 changelogs and a self-contradictory version label (class says v37, warning says v36). Competing versions (v36/v37/v46) coexist with no canonical marker. A fix to the shared math must be hand-applied to each copy. Refactor is blocked on C-06 (no tests).
- **graphify (2026-09-09) corroboration:** Graph extraction over the loss family produced **45 `semantically_similar_to` duplication edges** between the copy-pasted helpers, confirming the duplication is pervasive rather than incidental. It also surfaced a documentation defect not previously recorded: the `SpotlightLossHuber` and `SpotlightLossPowerLaw` docstrings both claim an *"identical architecture as SpotlightLossLogcosh ... KL-DRO"*, but `SpotlightLossLogcosh` is at v46 and uses **per-series sqrt DRO**, not KL-DRO. A reader trusting either docstring will form a wrong model of what the loss actually optimizes. Version strings v33/v35/v36/v37/v46 all coexist across the family with no canonical marker.
- **re-derivation (2026-09-10, `development` @ `fe7e681`):** **Partly changed.** Resolved: the `spotlight_loss.py` v37-vs-v36 contradiction, the vestigial `alpha` in `spotlight_loss.py` and `spotlight_loss_huber.py` (`spotlight_loss_power_law.py:41-47` now deprecates it with a warning), and the false KL-DRO claim in `spotlight_loss_huber.py`. Still present: the same claim in `spotlight_loss_power_law.py`, five copies each of `_log_cosh` and `_spectral_loss` with no base class, and a self-contradictory version label that moved from `spotlight_loss.py` to `spotlight_loss_asinh.py`.
- **Cross-refs:** C-06 (same modules, testing dimension); C-33 (a stale replica in the test suite, same drift-between-copies root cause).

---

### C-17 — Non-executing test guards: dead file, uncollected nested test, assertion-free test, skipped suite

- **Tier:** 3 *(false-coverage signal: the suite reports green on guards that never execute or never assert; misleads refactoring decisions. Escalated in scope 2026-09-09 from a single dead file to a four-location pattern — tier held at 3 because the impact is a wrong belief about coverage, not a direct correctness path.)*
- **Source:** tech-debt-cleanup (survey) + test-review (CIC alignment) + graphify (2026-09-09, test-coverage edge extraction)
- **Trigger:** A developer relies on any of these four guards — or on the manager CIC §10 — believing the behaviour is regression-tested, before refactoring the production code beneath it.
- **Location:** `tests/test_model_catalog.py:304` — `test_init_with_invalid_loss_raises_error` is **nested inside** `test_init_with_valid_loss_functions` (`:277`), so pytest never collects it; `tests/test_loss.py:448-451` — `TestTweedieLoss` is `@pytest.mark.skip`'d as obsolete. *(Two 0.1.x locations — the dead `test_model.py` and the assertion-free `test_stochastic_parity_serialization` — were deleted in the rewrite; see re-derivation bullet.)*
- **Narrative:** Guards that a reader counts as coverage but which contribute no assertion to any run. The nested test in `test_model_catalog.py` is invisible to collection — the invalid-loss error path it was written to protect is unguarded, and nothing in CI reports the omission. The skipped `TestTweedieLoss` is at least explicit. On 0.1.x there were four such locations; the rewrite deleted two of them *along with the files they lived in*, so the stochastic-parity property that `test_stochastic_parity_serialization` named is now simply unguarded rather than falsely guarded.
- **graphify (2026-09-09) corroboration:** Surfaced during test→production coverage-edge extraction; the three additional locations were merged into this entry on the same date. The mock-heavy files (`test_forecaster.py`, `test_mc_dropout_entropy_lock.py`, `test_model_catalog.py`, `test_reproducibility_infra.py`, `test_scaling_robustness.py`) are a separate and weaker signal — they do assert, but against mocks — and are already covered by C-21's analysis of why real `model.fit()` cannot run in CI.
- **re-derivation (2026-09-10, `development` @ `fe7e681`):** **Changed: 2 of 4 remain.** The dead `test_model.py` was deleted → resolved. `test_stochastic_parity_serialization` deleted with its file (`git grep stochastic_parity development` → 0) → resolved by deletion, property now unguarded. The nested test and the skipped Tweedie class are intact, verbatim. The manager CIC's §10 no longer cites the dead file (Stage 2). A benign sibling: `tests/test_shrinkage_loss.py:71` carries an explicit skip.
- **Cross-refs:** C-33 (a guard that runs but validates a stale replica — the fourth variant of the same false-coverage family); C-13, C-14, C-30 (real coverage gaps this false signal helps conceal); C-21 (structural cause of the mocking).

---

### C-18 — Patch layer reaches into Darts internals with no version assertion; module docstring names 0.45 against a 0.46.1 pin

- **Tier:** 2 *(the monkey-patches target version-specific Darts internals; a version bump can make a patch fail to apply or silently no-op with no test catching it — silent divergence between intended and shipped behaviour. **Rewritten 2026-09-10**: the original 0.40.0-vs-0.38.0 divergence is gone; the residual is the absence of any guard.)*
- **Source:** tech-debt-cleanup (survey) + test-review (F-7); re-derived 2026-09-10
- **Trigger:** Bumping `darts` in `pyproject.toml`; or editing any of the five `apply_*` patches without a `darts==0.46.1` environment to verify against.
- **Location:** `pyproject.toml:14` (`darts = "==0.46.1"`); `views_r2darts2/infrastructure/patches.py:3` (docstring: "Darts 0.45 internals"); `patches.py:204-946` (RevIN, TCN, TiDE MC-dropout, N-HiTS and N-BEATS layernorm patches, all reaching into Darts private classes). No test asserts `darts.__version__`. `.github/workflows/run_pytest.yml:31-33` installs the pin fresh each run (no committed lockfile).
- **Narrative:** CI now installs exactly the pinned version, so the tested/shipped split that motivated this entry is closed. What remains is structural: five patches rebind private Darts classes and methods, guarded only by ad-hoc attribute flags, with no `darts.__version__` assertion and no test that imports the patched classes and checks the rebinding took effect (the one RevIN test, C-33, tests a replica). The module docstring is already one version behind the pin. Whether the patches behave correctly against `0.46.1` internals could not be verified on the audit machine — `darts` is not installed.
- **Cross-refs:** C-12 (patch ordering/fragility); C-33 (the only patch test, against a replica); C-42 (CI seam).

---

### C-19 — God-methods in engines/infrastructure reduce testability

- **Tier:** 3 *(maintainability: long methods mixing concerns are hard to unit-test in isolation, compounding the C-13/C-14 coverage gaps; no direct correctness impact)*
- **Source:** tech-debt-cleanup (survey)
- **Trigger:** Adding or modifying a step inside one of these methods (e.g. a new preprocessing stage in `train()` or a new audit in `audit_manifest`).
- **Location:** (re-measured on `fe7e681`) `views_r2darts2/engines/darts_forecaster.py` — **`_predict_streaming` `:410-569` (160L, new)**, `predict` `:297-401` (105L), `_build_validation_set` `:214-293` (80L), `__init__` `:66-145` (80L, was 154L), `train` `:162-211` (~50L, was 150L); `views_r2darts2/infrastructure/callbacks.py` `:942-1056` (115L) and `:1678-1800` (123L); `views_r2darts2/infrastructure/reproducibility_gate.py` `audit_manifest` (~77L, was 116L); `views_r2darts2/dataset/base.py` — **`to_darts_timeseries` `:1573-1712` (140L, was 113L)**, `ingest_numpy_predictions` `:1355-1457` (103L, new).
- **Narrative:** Several methods exceed ~110 lines and bundle multiple responsibilities, making isolated characterization tests hard to write — part of why the C-13/C-14 gaps persist. Splitting is deferred (high-risk on partially-tested code) but the debt should be tracked.
- **re-derivation (2026-09-10, `development` @ `fe7e681`):** **Changed, net roughly flat.** The rewrite shrank `__init__`, `train`, `audit_manifest` and `GradientHealthCallback.on_train_epoch_end` (56L now), and introduced two new offenders of comparable size. The largest method in the package is now `ViewsDataset.to_darts_timeseries`.
- **Cross-refs:** C-13, C-30 (testability gaps these methods aggravate).

---

### C-21 — `accelerator="gpu"` is hardcoded, making CPU/MPS training impossible and end-to-end training untestable in CI

- **Tier:** 2 *(structural fragility: a realistic and routine scenario — running on any non-CUDA host — fails at Trainer construction, and the same constraint structurally prevents CI from ever exercising a real `model.fit()`)*
- **Source:** repo-assimilation (2026-09-09) (Phases 3 & 6)
- **Trigger:** Running a real (non-mocked) training or sweep on a host without CUDA — a developer laptop, an MPS Mac, or the GitHub Actions runner — or adding a CI job intended to exercise `model.fit()` end-to-end.
- **Location:** `views_r2darts2/catalogs/model_catalog.py:203` (`"accelerator": "gpu"`, inside `_get_common_pl_trainer_kwargs` `:132-207`), vs `views_r2darts2/infrastructure/device.py:15-25` (`get_device()` supports `mps`/`cuda`/`cpu`) and `views_r2darts2/engines/darts_forecasting_model_manager.py:335-337` (branches on `forecaster.device == "cpu"`). CI: `.github/workflows/run_pytest.yml:13` `ubuntu-latest`.
- **Narrative:** `_get_common_pl_trainer_kwargs` returns `"accelerator": "gpu"` unconditionally for every architecture. PyTorch Lightning raises `MisconfigurationException` at Trainer construction when no GPU is present, so training cannot start on a CPU or MPS host. Two parts of the codebase contradict this: `get_device()` explicitly supports `mps` and `cpu`, and the manager has a CPU-only parallel-prediction branch that a CPU-trained model could never reach. The consequence is not only portability — `.github/workflows/run_pytest.yml` runs on `ubuntu-latest` with no GPU, so **no test in the suite can call a real `model.fit()`**. This is the structural cause of the mock density in the training-path tests and therefore a root cause behind the C-13 / C-30 coverage gaps; `tests/test_darts_forecaster.py` (714 lines) is the 0.2.x successor of the old mock-heavy forecaster suite.
- **re-derivation (2026-09-10, `development` @ `fe7e681`):** **Still present**, unchanged in substance; `get_device()` moved to `infrastructure/device.py`.
- **Cross-refs:** C-13, C-30 (coverage gaps this constraint enforces); C-18 (both concern divergence between what is tested and what ships); C-42 (the other CI seam that hides an install-time failure).

---

### C-22 — Cyclic-encoder columns are computed then discarded before `fit`; two divergent resolution-inference sites remain; a test asserts the discarded behaviour

- **Tier:** 3 *(**re-tiered 2 → 3 on 2026-09-10.** The double injection is gone, so no doubled-input path exists. What remains is wasted computation, two inference sites that can disagree with no effect because one is dead, and a green test that documents a no-op — maintainability and misleading-signal cost, not a correctness path.)*
- **Source:** repo-assimilation (2026-09-09) (Phases 3 & 4); re-derived 2026-09-10
- **Trigger:** Changing `_split_targets_covariates` to include the encoder columns (which would re-create the double injection); or relying on `tests/test_views_dataset.py:269-291` as evidence that cyclic encoders reach the model.
- **Location:** `views_r2darts2/dataset/base.py:1649-1657` (encoder columns appended to a local `feature_columns_ext`) vs `:1226-1235` (`_split_targets_covariates` selects `ts[self.features]`, dropping them); `views_r2darts2/catalogs/model_catalog.py:240-266` (`_resolve_add_encoders`, the only path that reaches the model); inference sites `base.py:1650` (`self._time_id.split("_")[0][0]`) vs `model_catalog.py:260` (`level[-1]`); `tests/test_views_dataset.py:269-291`.
- **Narrative:** `to_darts_timeseries` still computes `month_sin`/`month_cos` and appends them — but to a *local* list, and `_split_targets_covariates` then selects only `self.features` for `past_cov`, so the columns never reach `model.fit`. The only encoder path that reaches the model is `ModelCatalog._resolve_add_encoders` via Darts' own `add_encoders`. Net: no doubling. The two resolution-inference sites both remain and still disagree in method, but the dataset one now feeds nothing. `tests/test_views_dataset.py:269-291` asserts the columns *are* appended — a green test documenting behaviour the pipeline throws away.
- **Cross-refs:** C-17 (a test that proves the wrong thing — same family); C-36 (another computed-then-unused path in the same method).

---

### C-24 — `parallel_workers > 1` races on the process-global RNG that `lock_entropy` reseeds

- **Tier:** 2 *(the Entropy Guardian's documented "bit-perfect identity" guarantee silently does not hold under the configuration the code explicitly offers; structural fragility with a clear trigger)*
- **Source:** repo-assimilation (2026-09-09) (Phases 3 & 5, invariant 11)
- **Trigger:** Setting `parallel_workers > 1` in a config for a CPU evaluation run; or removing/altering the GPU sequential-forcing branch at `darts_forecasting_model_manager.py:337`.
- **Location:** `views_r2darts2/engines/darts_forecasting_model_manager.py:335-345` (`parallel_workers` `:336`, sequential-forcing `:337`, `ThreadPoolExecutor` `:339`); `views_r2darts2/engines/darts_forecaster.py:335` (`lock_entropy` inside `predict`); `views_r2darts2/infrastructure/reproducibility_gate.py:736-751`.
- **Narrative:** `_evaluate_model_artifact` submits every rolling-origin sequence to a `ThreadPoolExecutor` sharing one `forecaster` and one model instance. Each `predict()` then calls `ReproducibilityGate.Data.lock_entropy(random_state)`, which reseeds `random`, `numpy.random`, and `torch` — all **process-global**. With more than one worker these reseeds interleave with other threads' sampling, so the bit-perfect reproducibility the `lock_entropy` docstring promises does not hold for MC-dropout or any probabilistic forecast. The GPU path is forced sequential, but only with a comment about Darts device-shifting race conditions; the reproducibility hazard on the CPU branch is documented nowhere.
- **re-derivation (2026-09-10, `development` @ `fe7e681`):** **Still present.** The `self.min_length` sub-claim is resolved (attribute no longer exists); the RNG race is unchanged and still undocumented on the CPU branch.
- **Cross-refs:** C-21 (the CPU branch is currently unreachable for a locally-trained model, which masks but does not remove this); silent-acceptance seam group.

---

### C-25 — Production infrastructure imports the test package

- **Tier:** 3 *(dependency-direction violation that makes installed-library behaviour depend on whether a `tests` module happens to be importable; contrary to ADR-002's first topological invariant)*
- **Source:** repo-assimilation (2026-09-09) (Phase 2 — dependency graph)
- **Trigger:** Installing the published wheel into an environment where an unrelated `tests` package is importable; or adding a `lint-imports` / import-linter contract to enforce ADR-002.
- **Location:** `views_r2darts2/infrastructure/patches.py:45` (`from tests.conftest import CLEAN_TORCH_LOAD`, inside `try/except` at `:44-53`), reached from `views_r2darts2/engines/darts_forecasting_model_manager.py:91` via `apply_all_patches()`. No `lint-imports` contract in `pyproject.toml`.
- **Narrative:** `apply_torch_load_patch` attempts `from tests.conftest import CLEAN_TORCH_LOAD` inside a `try/except (ImportError, ModuleNotFoundError)`. The intent is defensive — avoiding capture of a `Mock` during test runs — but it inverts the dependency direction, with shipped infrastructure reaching upward into the test package. ADR-002 states "Upward Imports are Forbidden" as its first topological invariant. Because the import is opportunistic, the installed library resolves `torch.__original_load__` differently depending on whether *any* importable `tests.conftest` exposing that name exists in the consumer's process.
- **re-derivation (2026-09-10, `development` @ `fe7e681`):** **Still present**, verbatim.
- **Cross-refs:** ADR-002 (revised 2026-09-10; now admits it has no mechanical enforcement); C-12, C-28 (patch-layer concerns).

---

### C-28 — `torch.load` is monkey-patched process-wide to `weights_only=False`

- **Tier:** 3 *(a global default weakened for every consumer in the process, not just this package's own artifact loads; correctness impact is nil today, so not Tier 2, but the blast radius is out of proportion to the need)*
- **Source:** repo-assimilation (2026-09-09) (Phases 2 & 3)
- **Trigger:** Another package in the same pipeline process calls `torch.load` on externally-supplied or untrusted weights after `DartsForecastingModelManager` has been constructed.
- **Location:** `views_r2darts2/infrastructure/patches.py:35-62` (global rebind `:60`, `weights_only=False` `:56-57`), invoked unconditionally from `views_r2darts2/engines/darts_forecasting_model_manager.py:91`.
- **Narrative:** `apply_torch_load_patch` rebinds the global `torch.load` so that **every** deserialization in the process defaults to `weights_only=False` — including calls made by `views-pipeline-core`, other model packages, or third-party libraries sharing the interpreter. The stated need is real and narrow: this package's `.scalers` sidecar holds live sklearn/Darts objects that `weights_only=True` refuses. **The narrower fix is now already applied at the load site** — `views_r2darts2/engines/darts_forecaster.py:669-671` passes `weights_only=False` explicitly for the `.scalers` sidecar — so the stated need for a process-global override is largely obsolete, yet the override remains and is silent to other consumers. The `Mock`-avoidance branch also couples this to C-25.
- **re-derivation (2026-09-10, `development` @ `fe7e681`):** **Still present; justification weaker.** The explicit `weights_only=False` at the load site means removing the global patch is now a deletion, not a migration.
- **Cross-refs:** C-25 (same function, test-package import); C-12 (patch-layer fragility).

---

### C-29 — Artifact timestamp is extracted by a fixed 15-character slice with no validation

- **Tier:** 3 *(a silently-wrong value in an ADR-governed cross-repo contract, but reachable only via a caller-supplied non-conforming artifact name, so not Tier 2)*
- **Source:** repo-assimilation (2026-09-09) (Phases 3 & 5, invariant 17)
- **Trigger:** Passing `--artifact_name` for a file whose stem does not end in the generated 15-character `YYYYMMDD_HHMMSS` timestamp — e.g. a hand-renamed, copied, or externally-produced artifact.
- **Location:** `views_r2darts2/engines/darts_forecasting_model_manager.py:307` and `:364` (`timestamp = path_artifact.stem[-15:]`, each immediately followed by `add_config({"timestamp": timestamp})`).
- **Narrative:** The extracted string is written straight into the config via `_config_manager.add_config({"timestamp": timestamp})` and thereby into every prediction filename. ADR-015 documents this implementation as the *correct* discharge of the cross-repo artifact-prediction timestamp contract (views-pipeline-core ADR-052), and it is correct for names produced by `generate_model_file_name`. But there is no length check, no format check, and no error path on the read side, while `artifact_name` is explicitly caller-supplied (`_evaluate_model_artifact(eval_type, artifact_name=None)`). ADR-015 itself states that a timestamp mismatch causes "silent fallback to subprocess re-execution or failure" in the ensemble manager — so the failure mode is documented while the guard is absent. The contract is enforced by convention at the write site and by nothing at the read site.
- **re-derivation (2026-09-10, `development` @ `fe7e681`):** **Still present**, verbatim. ADR-015 still describes this implementation as correct for generated names; the missing guard is on the read side.
- **Cross-refs:** ADR-015 (the governing contract); silent-acceptance seam group.

---

### C-30 — Untested infrastructure outside the existing coverage entries: scheduler catalog and custom LR scheduler

- **Tier:** 3 *(three modules performing non-trivial config translation and phase arithmetic with zero test references; increases the cost and risk of change for multiple contributors, but no identified silent-corruption path, so not Tier 2)*
- **Source:** repo-assimilation (2026-09-09) (Phase 6 — test coverage)
- **Trigger:** Selecting a non-default `lr_scheduler_cls` (especially `WarmupCAWR`); or adding an entry to `SchedulerCatalog._KWARG_MAP` or `_STATIC_KWARGS`.
- **Location:** `views_r2darts2/catalogs/scheduler_catalog.py` (128 LOC: `_KWARG_MAP:17`, `_STATIC_KWARGS:44`, `_CUSTOM_SCHEDULERS:53,57-59`, last-wins injection `:123`), `views_r2darts2/math/warmup_cawr.py` (106 LOC; reachable via `reproducibility_gate.py:308-313` + `scheduler_catalog.py:58-59`) — zero references to either under `tests/`.
- **Narrative:** C-06 and C-13 cover the untested loss family and callbacks; these two modules fall outside both. `SchedulerCatalog` performs non-trivial work that is entirely unverified: config-key→torch-kwarg remapping via `_KWARG_MAP`, pass-through of the nested `lr_scheduler_kwargs` block with genome-key precedence, lazy loading of `_CUSTOM_SCHEDULERS`, and last-wins injection of `_STATIC_KWARGS` that deliberately overrides config values.
- **graphify (2026-09-09) note:** The harness the archived roadmap specified exists as `tests/losses/harness.py`; it is loss-oriented and does not cover schedulers.
- **re-derivation (2026-09-10, `development` @ `fe7e681`):** **Changed: 1 of 3 closed.** `encoders.py` is now covered by `tests/test_encoders.py:37-149` (shape/range/dtype, period-12 orbit, `CYCLIC_ENCODERS_BY_RESOLUTION`) — the `(idx - 1) % period` convention is exercised. Scheduler catalog and `warmup_cawr.py` remain at zero.
- **Cross-refs:** C-06, C-13 (adjacent coverage gaps); C-21 (the structural reason integration coverage is hard here).

---

### C-32 — `ModelCatalog.self.device` is assigned and never read

- **Tier:** 4 *(one dead assignment; no correctness or reliability impact. Two of the entry's three original halves were resolved by the rewrite — see re-derivation bullet.)*
- **Source:** repo-assimilation (2026-09-09) (Phase 2 — dependency graph)
- **Trigger:** Reading `ModelCatalog.device` and assuming it is used somewhere; or adding a second call to `get_device()` believing the first is load-bearing.
- **Location:** `views_r2darts2/catalogs/model_catalog.py:115` (`self.device = get_device()`; `self.device` has no other reference in the file).
- **Narrative:** The assignment survives from 0.1.x, where it held open a cross-layer import. That import is gone; the value is still computed and still unread. Deleting the line is a one-line change with zero behavioural cost — and it also removes one of the two call sites that can trip C-34's `set_default_dtype` side effect.
- **re-derivation (2026-09-10, `development` @ `fe7e681`):** **Changed: 2 of 3 resolved.** The `catalogs → engines` edge is gone — `model_catalog.py:31` imports `infrastructure.device.get_device`, and `device.py:3-6` documents the extraction as the circular-import fix. `SchedulerCatalog` is now in the root `__all__` (`__init__.py:38`, lazy `:71-81`). Only the dead assignment remains.
- **Cross-refs:** C-34 (this call site triggers its side effect); C-11 (same catalog).

---

### C-33 — `test_rinorm_curvature_patch.py` validates a local replica that has diverged from the production RevIN patch

- **Tier:** 2 *(a 288-line guard over the repo's most mathematically intricate production code asserts against a copy whose formula no longer matches what ships; the suite reports green while the real patch is unverified — structural fragility with a clear trigger)*
- **Source:** graphify (2026-09-09) (semantic extraction, test→production coverage edges)
- **Trigger:** Editing `apply_rinorm_compression_patch()` in `patches.py` — changing the centering space, the σ clamp, the ±50 pre-`sinh` clamp, or the likelihood scale-parameter branch — and relying on `test_rinorm_curvature_patch.py` passing as evidence the change is safe.
- **Location:** `tests/test_rinorm_curvature_patch.py` (288 lines; imports only `pytest`/`torch`/`numpy` at `:18-21`; `class FakeRawSpaceRINorm` at `:24`; docstring `:5-8` states the v2 pure-raw-space formula); `views_r2darts2/infrastructure/patches.py:204-309` (`apply_rinorm_compression_patch`, hybrid v3: `torch.sinh(x - self.mean)` `:236`, `torch.asinh(x_centered_raw / self.stdev)` `:249`, ±50 clamp `:271`, likelihood scale-parameter identity branch `:287-300`).
- **Narrative:** The test file imports nothing from `views_r2darts2`. It defines its own `FakeRawSpaceRINorm` and asserts curvature, Jensen-bias, and z-range properties against that local class. The replica implements **pure raw-space RevIN** — the v2 formula — while production has since moved to **hybrid-space v3** (`z = asinh(sinh(x − μ_asinh) / σ_c)`, centering in asinh space and normalizing variance in raw space). The comment block preceding the patch documents v2's systematic positive mean bias as the explicit reason v3 replaced it. The test therefore passes by validating the exact behaviour the production code was rewritten to abandon: a green run is evidence about a formula that no longer ships. This is the most consequential instance of the C-17 false-coverage family, because RevIN sits directly on the inverse path from model output to raw counts, and because C-18 already establishes that the patch layer is unverified against the pinned Darts version.
- **re-derivation (2026-09-10, `development` @ `fe7e681`):** **Still present, verbatim** — the replica followed the rewrite across; the test file still imports no production code. *(The CIC and the 0.2.x test listing both now say so explicitly.)*
- **Cross-refs:** C-17 (same false-coverage family — this is the "runs but proves nothing" variant); C-12, C-18 (patch-layer fragility and version divergence); C-15 (drift between copies, same root cause in the loss family).

---

### C-34 — `get_device()` mutates global torch state from inside a getter

- **Tier:** 4 *(a process-wide default-dtype mutation triggered as a side effect of a device query; latent today because the codebase is float32 throughout and the MPS branch is unreachable while C-21 stands)*
- **Source:** graphify (2026-09-09) (surfaced via the archived reproducibility-investigation cluster, confirmed against source)
- **Trigger:** Running on an MPS host after C-21 is fixed, in a process where another library depends on `torch.get_default_dtype()` being float64 — or calling `ModelCatalog(config)` purely to inspect the catalog, which invokes `get_device()` as a side effect.
- **Location:** `views_r2darts2/infrastructure/device.py:22` (`torch.set_default_dtype(torch.float32)` inside the MPS branch of `get_device()` `:15-25`); called from `views_r2darts2/catalogs/model_catalog.py:115` and `views_r2darts2/engines/darts_forecaster.py:143`; `DartsForecaster.get_device` (`:157-159`) is now a thin re-export.
- **Narrative:** `get_device()` is a module-level function whose name promises a pure query, but the MPS branch calls `torch.set_default_dtype(torch.float32)`, a process-global mutation, before returning. Any caller that merely asks which device is available silently changes global tensor defaults for every other library in the interpreter. The effect is benign in this repo (ADR-016 mandates float32 everywhere), and the MPS branch cannot currently be reached during training because C-21 hardcodes a GPU accelerator — but the side effect is invisible at the call site and one of those two mitigations is a defect rather than a design. `ModelCatalog.__init__` calls it for a value it never reads (C-32), so a catalog instantiation alone can trip it.
- **re-derivation (2026-09-10, `development` @ `fe7e681`):** **Moved** to `infrastructure/device.py` when the cycle-break extraction happened. The docstring (`:18-20`) now *acknowledges* the side effect — documentation improved, mutation unchanged.
- **Cross-refs:** C-28 (same pattern: process-global torch mutation as a side effect); C-32 (the dead `self.device` call site); C-21 (currently masks the MPS path).

---

---

### C-35 — Entity-length filter removed entirely: short-history entities are no longer excluded from training

- **Tier:** 2 *(a behaviour the 0.1.x code applied deliberately and documented — excluding entities whose training window is shorter than `input_chunk_length + output_chunk_length` — no longer exists; what Darts does with such series is unverified, and if it silently skips or pads them the change is invisible in any metric. Structural fragility with a realistic trigger: any dataset with a new or short-lived entity.)*
- **Source:** repo-assimilation (2026-09-10) (successor to C-20's context)
- **Trigger:** Training on a dataset that contains an entity with fewer than `input_chunk_length + output_chunk_length` observations in the training partition — e.g. a country that enters the panel late — and expecting the 0.1.x exclusion to apply.
- **Location:** `views_r2darts2/dataset/base.py:1131` (`fit_scalers`) and `:1223-1235` (`_split_targets_covariates`) — no length filter; `git grep 'Training filter\|min_length' development -- views_r2darts2/` → zero hits.
- **Narrative:** The 0.1.x `_preprocess_timeseries` filtered entities by training-slice length and logged `Training filter: N/M entities passed`. The rewrite deleted that method (resolving C-20) and did not carry the filter anywhere. Every entity now reaches `model.fit` regardless of length. Whether Darts raises, skips, or pads short series depends on the model and could not be checked on the audit machine. Either the filter was intentionally dropped — in which case the decision is undocumented — or it was lost in the rewrite.
- **Cross-refs:** C-20 (resolved; the method the filter lived in); C-18 (the same missing-`darts` limitation blocks verification).

---

### C-36 — `compute_static_covariates` is dead: 392 tested lines with zero production callers

- **Tier:** 3 *(a complete, tested feature that nothing calls; the fingerprint it replaced was a documented modelling input, so its silent absence changes what models see — but the absence is deterministic and produces no wrong values, so not Tier 2)*
- **Source:** repo-assimilation (2026-09-10) (Phase 4 — transformers)
- **Trigger:** Configuring `static_covariate_stats` or `use_static_covariates` and expecting the µ/σ/max/trend/sparsity fingerprint to reach the model; or deleting the module as unused without realising the README and ADR-001 still describe the feature.
- **Location:** `views_r2darts2/transformers/static_covariates.py:133` (`compute_static_covariates`); callers: `views_r2darts2/transformers/__init__.py:18` (export only) and `tests/test_static_covariates.py`; `views_r2darts2/dataset/base.py:1706-1709` attaches only the entity id as a static covariate.
- **Narrative:** The module reimplements the 0.1.x pandas fingerprint in numpy, claims bit-for-bit parity with it, and is covered by a 420-line test file. Nothing in `views_r2darts2/` calls it. `ViewsDataset.to_darts_timeseries` attaches the entity id and nothing else. So on 0.2.x the "Static Covariate Fingerprints" feature the README advertises does not exist at runtime, while a fully-built implementation of it sits one import away. The parity claim is against an implementation that was deleted.
- **Cross-refs:** C-08 (the latent leakage seam inside this dead module — reconnecting it re-arms that one); C-07 (four architectures would ignore the fingerprint anyway); C-40 (README still advertises the feature).

---

### C-37 — Device-restore failure logs a warning and continues on CPU, contradicting ADR-008 / ADR-011

- **Tier:** 2 *(the ADRs mandate fail-loud; the code degrades silently to a state the ADR-011 context describes as "massive performance degradation" and a race-condition risk; structural contradiction between a stated invariant and its implementation, with a realistic trigger on any GPU host under memory pressure)*
- **Source:** repo-assimilation (2026-09-10) (ADR audit, Phase 3)
- **Trigger:** A prediction run on a CUDA host where `model.to(device)` fails or is refused (OOM, device busy) — the run continues on CPU with only a `WARNING` in the log.
- **Location:** `views_r2darts2/engines/darts_forecaster.py:633-640` (`_ensure_model_on_device`; `logger.warning("Failed to move model from CPU to %s; continuing on CPU.")` at `:640`).
- **Narrative:** ADR-008 §1 and ADR-011 §Validation both require that a failed CPU→GPU restoration raise and abort. The implementation warns and proceeds. The 0.1.x code raised `RuntimeError` here; the rewrite softened it. Both ADRs now carry a compliance note pointing to D-01, and the `DartsForecaster` CIC records the current behaviour. This entry is the code side of that disagreement: whichever way D-01 is ruled, one artifact changes.
- **Cross-refs:** D-01 (the ruling); ADR-008, ADR-011; C-21 (the CPU branch this degrades into is the one that cannot train).

---

### C-38 — `apply_tide_skip_layernorm_patch` is dead-but-callable and would overwrite the live TiDE forward

- **Tier:** 3 *(the C-16 pattern recurring: ~180 LOC of disabled patch code that a caller can still invoke. Not exported this time, so the footgun is one step further away — but invoking it clobbers `_TideModule.forward` set by the live MC-dropout patch, so the consequence is worse than C-16's.)*
- **Source:** repo-assimilation (2026-09-10) (Phase 5 — infrastructure)
- **Trigger:** A contributor calls `apply_tide_skip_layernorm_patch()` directly, or uncomments it in `apply_all_patches` without removing `apply_tide_mc_dropout_patch`.
- **Location:** `views_r2darts2/infrastructure/patches.py:761-800` (function) with helpers at `:631-677` and `:679-759`; commented out of `apply_all_patches` at `:943`; sets `_TideModule.forward` at `:793`, which `apply_tide_mc_dropout_patch` already set at `:603`.
- **Narrative:** The function is complete, importable, and disabled by a comment. Unlike the resolved C-16, it is not in the public `__all__`. But the two TiDE patches both assign `_TideModule.forward`, last-writer-wins, with no guard checking whether the other has run. Re-enabling this one silently removes the MC-dropout behaviour that `apply_tide_mc_dropout_patch` exists to provide. Either it should be deleted, or the two should be made mutually exclusive with a runtime check.
- **Cross-refs:** C-12 (forward-method overwrite hazard); C-16 (resolved predecessor).

---

### C-39 — Ten core classes carry no in-code Intent Contract, contrary to ADR-006

- **Tier:** 3 *(ADR-006 makes the in-code contract mandatory for every non-trivial class; ten of the most important classes have plain docstrings instead. The Markdown CICs now cover them, so no reader is misled — but the ADR's "must live in the class docstring or a linked Markdown file" is met only by the second clause, and the code-side half is missing. Maintainability and governance-compliance cost.)*
- **Source:** repo-assimilation (2026-09-10) (ADR audit)
- **Trigger:** Marking ADR-006 as compliant; or refactoring one of the ten classes with only its docstring open and no CIC to hand.
- **Location:** `views_r2darts2/dataset/base.py:40` (`ViewsDataset`), `views_r2darts2/engines/darts_forecasting_model_manager.py:58`, `views_r2darts2/catalogs/loss_catalog.py:27`, `views_r2darts2/catalogs/optimizer_catalog.py:7`, `views_r2darts2/catalogs/scheduler_catalog.py:8`, `views_r2darts2/infrastructure/reproducibility_gate.py:75`, `views_r2darts2/transformers/feature_scaler_manager.py:37`, `views_r2darts2/transformers/scaler_selector.py:93`, `views_r2darts2/dataset/builder.py:145` (`DatasetBuilder`), `views_r2darts2/dataset/zarr_store.py:33` (`ZarrStore`). Present on: `ModelCatalog`, `DartsForecaster`, and eleven callbacks.
- **Narrative:** The ADR's Form-of-the-Contract section allows either a docstring block or a linked Markdown file. Every one of the ten now has a Markdown CIC (Stage 2, 2026-09-10), so the ADR is satisfied by the letter. The spirit — that a reader of the class sees its intent without leaving the file — is not. ADR-006 is marked "compliance open" until D-02 is ruled.
- **Cross-refs:** D-02 (the ruling: require both, or accept Markdown-only); ADR-006.

---

### C-40 — The shipping `README.md` documents a deleted class and a deleted function

- **Tier:** 3 *(the first document a newcomer reads describes an API that does not exist; misleads every reader of the release, though it produces no wrong runtime behaviour)*
- **Source:** repo-assimilation (2026-09-10) (Phase 8 — docs on `development`)
- **Trigger:** A user follows the README's API Reference and calls `_ViewsDatasetDarts(...)` or `ScalerSelector.get_chained_scaler(...)`.
- **Location:** `README.md` on `development` — §API Reference documents `_ViewsDatasetDarts` (L360); the Chained Scalers section calls `get_chained_scaler` (L335); the "⚡ Loss Functions" heading appears twice (L158, L294).
- **Narrative:** The README survived the `ec34786` documentation purge and was not updated for the `a79f23b` dataset rewrite. It also advertises "Static Covariate Fingerprints" as a feature, which on 0.2.x is not wired (C-36). Fixed by Stage 5 of the `governance-0.2.x` branch; this entry is registered so the resolution is traceable.
- **Cross-refs:** C-36 (the advertised feature that is dead); C-17 / C-27 (false-signal family).

---

### C-41 — Thirteen of twenty-two loss modules have no loss card; two spec cards advertise constructor defaults that do not exist

- **Tier:** 3 *(the loss-card layer is the only place the loss family's parameters and behaviour are documented for researchers; more than half of it is missing, and two cards contradict ADR-003 by listing "Default" values for arguments the constructors make mandatory)*
- **Source:** repo-assimilation (2026-09-10) (docs audit, Phase C)
- **Trigger:** A researcher configures `SpotlightLoss` (the production loss) or any Spotlight/Prism/Sentinel variant from documentation; or reads `tweedie_loss_spec.md` / `shrinkage_loss_spec.md` and omits a "defaulted" gene from the DNA.
- **Location:** `docs/loss_cards/` — 9 cards for 22 modules under `views_r2darts2/math/`; missing for `PrismLoss`, `SpotlightLoss`, `SpotlightLossLogcosh`, `SpotlightLossHuber`, `SpotlightLossAsinh`, `SpotlightLossPowerLaw`, `SpotlightFocalLoss`, `SentinelLoss`, `CharbonnierLoss`, and the passthroughs. `docs/loss_cards/tweedie_loss_spec.md` lists `p` Default 1.5 and `eps` Default 1e-6; `docs/loss_cards/shrinkage_loss_spec.md` lists `a` Default 10.0, `c` Default 0.2 — `TweedieLoss.__init__` and `ShrinkageLoss.__init__` have no defaults and `LOSS_GENOMES` mandates every argument.
- **Narrative:** The "Default" columns were relabelled "Typical value (must be declared in DNA)" in Stage 4 of `governance-0.2.x`, and `loss_cards/README.md` (previously 0 bytes) now indexes the nine cards and names the thirteen gaps. The gaps themselves remain: the production loss has no card. Writing them is blocked on the same knowledge C-06 needs — nobody has verified the advanced family's behaviour.
- **Cross-refs:** C-06 (same modules, untested); C-15 (same modules, duplicated); ADR-003.

---

### C-42 — CI never installs the `manager` extra, so `views-pipeline-core` and `views-r2darts2` have never met a dependency resolver

- **Tier:** 2 *(a published release is uninstallable alongside its own optional dependency, and nothing in the repository can detect that; structural fragility with a trigger on every release)*
- **Source:** repo-assimilation (2026-09-10) (Phase 6); externally corroborated by `views-r2darts2` issue #34 (filed 2026-09-09)
- **Trigger:** Cutting a release; or changing any dependency bound in `pyproject.toml` and relying on a green CI run as evidence it resolves.
- **Location:** `.github/workflows/run_pytest.yml:31-33` (`poetry install`, no `--extras manager`, no committed lockfile); `pyproject.toml` (`views-pipeline-core` optional under `[tool.poetry.extras] manager`; `wandb = ">=0.28.2"`).
- **Narrative:** Because pipeline-core is an optional extra and CI runs a bare `poetry install`, every CI run installs r2darts2 *without* pipeline-core. The two packages' dependency lists are therefore never intersected by any automated process. The consequence surfaced on 2026-09-09: `views-r2darts2 0.2.x` requires `wandb>=0.28.2` while every published `views-pipeline-core` requires `wandb<0.19.0`, so `pip install "views-r2darts2[manager]"` fails with `ResolutionImpossible` against the current platform — verified by pip on Python 3.11. The maintainer's environment works because pipeline-core was installed before the bound changed. One CI step — `pip install --dry-run "views-r2darts2[manager]"` — would have caught it. **The dependency fix itself is out of scope here**; this entry records the harness gap that let it ship.
- **Cross-refs:** C-21 (the other CI seam — training can't run there either); C-18 (no lockfile, fresh resolve per run).

---

### C-43 — A live sweep config names an entrypoint script that exists only under `reports/archived/`

- **Tier:** 4 *(one W&B sweep config points at a script that is not on any runtime path; the sweep would fail at launch, loudly)*
- **Source:** repo-assimilation (2026-09-10) (reports audit, Part 3)
- **Trigger:** Launching `lr_finder_sweep` through W&B.
- **Location:** `sweep_configs/experimental_sweep_configs/lr_finder_sweep.py:7` on `survey_risk` (the `sweep_configs/` tree is absent from `development`; entry scoped to the 0.1.x line) — `"program": "simple_training_run.py"`; the only file of that name is `reports/archived/simple_training_run.py`.
- **Narrative:** The archived directory carries an `__init__.py` and is therefore importable, which may be why this once worked from a particular working directory. Noted in `reports/archived/README.md` (Stage 0). The sweep file is out of scope for the documentation branch.
- **Cross-refs:** none.

---

## Disagreements

> Each D-entry records a contradiction between an accepted ADR and the 0.2.x code, surfaced by the
> 2026-09-10 re-derivation. None is ruled here. The ADR text carries a compliance note pointing at
> the D-entry; the corresponding C-entry (where one exists) is the code side. Whichever way a ruling
> goes, exactly one artifact changes.

### D-01: Device-restore failure — fail-loud (ADR-008, ADR-011) vs warn-and-continue (code)

| Field | Value |
|-------|-------|
| ID | D-01 |
| Source | repo-assimilation (2026-09-10) |
| Perspectives | **ADR-008 §1 / ADR-011 §Validation** — a failed CPU→GPU restoration is a structural lie and must raise; ADR-011's own context names silent CPU fallback as a race-condition and performance hazard. **Code** (`darts_forecaster.py:633-640`) — warns and continues; a long evaluation that would otherwise abort completes on CPU. |
| Resolution | Unresolved — requires maintainer ruling. Ruling for the ADRs → change one line in `_ensure_model_on_device` (C-37). Ruling for the code → soften both ADRs and `docs/standards/logging_and_observability_standard.md` §5.2. |

---

### D-02: Intent Contracts — in-code on every non-trivial class (ADR-006) vs Markdown-only for ten (code)

| Field | Value |
|-------|-------|
| ID | D-02 |
| Source | repo-assimilation (2026-09-10) |
| Perspectives | **ADR-006** — "the contract must live in the class docstring *or* a linked Markdown file"; the ADR's spirit is that intent travels with the class. **Code** — `ModelCatalog`, `DartsForecaster` and eleven callbacks carry an `Intent Contract:` block; ten core classes (C-39) do not, and are covered only by `docs/CICs/`. |
| Resolution | Unresolved. Ruling for in-code → ten docstring additions (C-39). Ruling for Markdown-sufficient → reword ADR-006 §Form and close C-39. |

---

### D-03: Scaling — "custom scaling wrappers are forbidden" (ADR-012) vs `transformers/inverse.py` (code)

| Field | Value |
|-------|-------|
| ID | D-03 |
| Source | repo-assimilation (2026-09-10) |
| Perspectives | **ADR-012 §1** — all transformations use Darts-native `Pipeline`/`Scaler`; custom wrappers were the cause of the 2026-02 calibration collapse. **Code** — `inverse.py` reaches into Darts' private `_fitted_params`/`_fit_called` to preserve the sample dimension for probabilistic inverses, because the native path does not; it consolidated two 0.1.x copies into one and confines the fragility to one module. |
| Resolution | Unresolved. Ruling for the ADR → replace `inverse.py` with a native path (unknown feasibility). Ruling for the code → amend ADR-012 to sanction a single, named exception and require the silent passthrough (C-10) to raise. |

---

### D-04: File organisation — 1-Class-1-File (ADR-013) vs three homogeneous-family files (code)

| Field | Value |
|-------|-------|
| ID | D-04 |
| Source | repo-assimilation (2026-09-10) |
| Perspectives | **ADR-013 §1** — every non-trivial class has exactly one file. **Code** — `dataset/converters.py` (5 converters), `dataset/subclasses.py` (6 LOA datasets, 81 lines total), `transformers/static_covariates.py` (config + result dataclass). Each is a family with one shared contract; splitting `subclasses.py` would produce six ten-line files. |
| Resolution | Unresolved. Ruling for the ADR → split three files into thirteen. Ruling for the code → extend ADR-013 §3's hub concept to "homogeneous families", and record the three as sanctioned. |

---

### D-05: Output semantics — no semantic floors in the data layer (ADR-010/016) vs `clip_negatives=True` (code)

| Field | Value |
|-------|-------|
| ID | D-05 |
| Source | repo-assimilation (2026-09-10) |
| Perspectives | **ADR-010 §3, carried into ADR-016 as an open question** — clipping in the data layer is a hidden heuristic that masks model behaviour; any floor belongs in evaluation or as a declared gene. **Code** — `ViewsDataset.ingest_darts_predictions` / `ingest_numpy_predictions` default `clip_negatives=True`; `DartsForecaster`'s docstring lists non-negativity as a guarantee; the argument is that fatality counts cannot be negative and this is physics, not modelling. |
| Resolution | Unresolved. Ruling for the ADR → flip the default to `False` or add a `prediction_floor` gene; update the forecaster docstring and CIC. Ruling for the code → ADR-016 §3 becomes a decision rather than an open question, distinguishing the physical zero floor from modelling thresholds. |

---

## Resolved Concerns

### C-27 — `ADR_COMPLIANCE_REPORT.md` asserts total compliance while the register records 14 open concerns *(resolved 2026-09-10 (this branch, Stage 0))*

- **Tier:** 3
- **Source:** repo-assimilation (2026-09-09) (Phases 1 & 6)
- **Location:** Previously at `docs/ADR_COMPLIANCE_REPORT.md`
- **Resolution:** `ADR_COMPLIANCE_REPORT.md` moved to `docs/archive/` with a `HISTORICAL` status header stating that nothing in it is asserted of 0.2.x. `docs/INSTANTIATION_CHECKLIST.md` no longer cites it as current evidence (Stage 4). It was already absent from `development` (deleted in `ec34786`); this resolution concerns the governance branch.

---

### C-26 — ADR-002's layer map names four directories that do not exist in the package *(resolved 2026-09-10 (this branch, Stage 1))*

- **Tier:** 3
- **Source:** repo-assimilation (2026-09-09) (Phases 1 & 2)
- **Location:** Previously at `docs/ADRs/002_topology_and_dependancy_rules.md` (Layer 0 `utils/`, Layer 1 `data/`, Layer 2 `model/`, Layer 3 `manager/`)
- **Resolution:** ADR-002's layer map was rewritten from a measured import graph to the six real packages — L0 `infrastructure/`+`math/`, L1 `transformers/`+`catalogs/`, L2 `dataset/`, L3 `engines/` — with the `device.py` cycle-break recorded and the absence of mechanical enforcement stated in the ADR itself (`docs/ADRs/002_topology_and_dependancy_rules.md`, revised 2026-09-10). The residual — no import-linter contract — is carried by C-25's trigger.

---

### C-31 — `math/warmup_cosine.py` is unreachable dead code *(resolved 2026-09-10 (by 0.2.x rewrite))*

- **Tier:** 4
- **Source:** repo-assimilation (2026-09-09) (Phase 2 — dependency graph)
- **Location:** Previously at the deleted `views_r2darts2/math/warmup_cosine.py`
- **Resolution:** `views_r2darts2/math/warmup_cosine.py` is absent from `git ls-tree -r development`. Deleted.

---

### C-23 — Dataset feature-list mutation makes cyclic-encoder scaling depend on which feature-scaler style is configured *(resolved 2026-09-10 (by 0.2.x rewrite))*

- **Tier:** 2
- **Source:** repo-assimilation (2026-09-09) (Phase 4)
- **Location:** Previously at the deleted `transformers/views_dataset_darts.py` feature-list mutation and the 0.1.x `DartsForecaster.__init__` scaler construction
- **Resolution:** `self.features` is never mutated on 0.2.x — its only writes are at construction (`views_r2darts2/dataset/base.py:104`) and on the subset path (`:188`). `FeatureScalerManager` is built inside `fit_scalers` (`:1183-1187`) with the final `all_features`. Cyclic-encoder columns never enter `past_cov` at all (see C-22), so the `feature_scaler_map`-vs-`feature_scaler` asymmetry cannot arise.

---

### C-20 — `_preprocess_timeseries` overwrites its own entity-alignment fix for `past_covariates` *(resolved 2026-09-10 (by 0.2.x rewrite))*

- **Tier:** 2
- **Source:** repo-assimilation (2026-09-09) (Phases 3 & 5 — engines, invariant 14)
- **Location:** Previously at the deleted `_preprocess_timeseries` in `views_r2darts2/engines/darts_forecaster.py`
- **Resolution:** `_preprocess_timeseries` no longer exists (`tests/test_darts_forecaster.py:16` documents its removal). `DartsForecaster.train()` receives `(target_series, past_covariates)` as one lockstep return from `ViewsDataset.fit_scalers(..., return_series=True)`; the split happens once in `ViewsDataset._split_targets_covariates` (`views_r2darts2/dataset/base.py:1223-1235`), so the two lists are index-aligned by construction and there is no overwrite. The open verification question (does Darts validate list lengths) is moot — the code path is gone. **Successor concern:** the entity-length filter the old code applied was removed entirely, not fixed; registered as C-35.

---

### C-16 — `apply_nbeats_patch` is disabled in `apply_all_patches` but still in the public API *(resolved 2026-09-10 (by 0.2.x rewrite))*

- **Tier:** 3
- **Source:** tech-debt-cleanup (survey)
- **Location:** Previously at the deleted `apply_nbeats_patch` in `views_r2darts2/infrastructure/patches.py` and its export in `__init__.py`
- **Resolution:** `apply_nbeats_patch` no longer exists; `views_r2darts2/__init__.py` exports only `apply_all_patches` and `apply_tide_mc_dropout_patch`. **The pattern recurred** in a non-exported form — `apply_tide_skip_layernorm_patch` is defined and commented out of `apply_all_patches` — and is registered separately as C-38.

---

### C-14 — `_ViewsDatasetDarts` correctness guarantees (entity isolation, schema, index) untested *(resolved 2026-09-10 (by 0.2.x rewrite))*

- **Tier:** 2
- **Source:** test-review (Kleppmann lens; CIC contract alignment)
- **Location:** Previously at the deleted `transformers/views_dataset_darts.py` (entity grouping and schema audit)
- **Resolution:** `_ViewsDatasetDarts` was deleted (`de65932`, `a79f23b`); its replacement `ViewsDataset` (`views_r2darts2/dataset/base.py`) is covered by `tests/test_views_dataset.py` (831 lines, 47 tests: `to_darts_timeseries` incl. all-entities, single-row and empty-subset cases; fail-loud schema validation for missing target column, empty targets, missing file, unsupported source; scalers; persistence), plus `tests/test_parquet_loader.py` and `tests/test_zarr_cleanup.py`. **Residual, not re-registered:** no test is named specifically for cross-entity contamination — add one if that guard is wanted explicitly.

---

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

---

## Register Conventions

- **ID format:** `C-xx` for concerns, `D-xx` for disagreements. IDs are permanent — gaps indicate merged or resolved entries.
- **Sources:** `repo-assimilation`, `expert-review`, `test-review`, `falsification-audit`, `clean-architecture-review`, `pr-review`, `review-diff`, `tech-debt-audit`, `graphify`, `incident`.
- **Disagreements:** `D-xx` entries record an ADR-vs-code contradiction awaiting a ruling. They are not concerns; they point at the concern (if any) that is the code side.
- **Re-derivation:** when the code moves under an entry, the entry is re-verified and carries a dated `re-derivation` bullet; Location is updated to current line numbers; Tier is changed only with a stated reason.
- **Resolution:** Move to "Resolved Concerns" with resolution date and one-line summary when addressed. Do not delete.
- **Header counts:** Manually maintained — update whenever a concern is added or resolved.
- **Governed by:** ADR-014.
