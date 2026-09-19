# ADR-016: Numerical Precision, Raw Output, and the Non-Negativity Floor

**Status:** Accepted (supersedes ADR-010)  
**Date:** 2026-09-10  
**Deciders:** Simon Polichinel von der Maase  
**Revised:** 2026-09-17 — decision 3 ruled (maintainer, issue #40, 2026-09-10): the floor stays. D-05 closed.  

---

## Context

ADR-010 (2026-02-11) made four decisions for the 0.1.x codebase: universal `float32`, a raw-output mandate, a prohibition of hardcoded semantic floors in the data and model layers, and separation of "what counts as zero" into the evaluation layer or an explicit DNA gene.

The 0.2.x rewrite (`views_r2darts2/dataset/`, commit `a79f23b`) kept the first, second and fourth. It did not keep the third. `ViewsDataset.ingest_darts_predictions` and `ViewsDataset.ingest_numpy_predictions` in `views_r2darts2/dataset/base.py` take `clip_negatives: bool = True` and, by default, execute `np.maximum(values, 0.0)` on every prediction before it is stored. `DartsForecaster`'s own docstring advertises this as a guarantee: *"Predictions are inverse-transformed and clipped to non-negative."* The 0.1.x forecaster did the same in its prediction-processing method, which ADR-010 named as the enforcement point — so the contradiction predates the rewrite. The rewrite made it a default parameter on the ingest path *and kept a second, unconditional clip* on the streaming path: `views_r2darts2/engines/darts_forecaster.py:515` — `np.maximum(target_values, 0.0, out=target_values)` — with no `clip_negatives` opt-out at all.

ADR-010 also cites a deleted method, a nonexistent epsilon, and a deleted test file. It cannot be patched into truth. It is superseded.

---

## Decision

1. **Universal Precision (carried forward).** All data tensors — targets and covariates — are `float32` before entering any model or loss. On 0.2.x this holds by construction: `ViewsDataset` stores `float32` (`dataset/converters.py` emits nothing else), and `ReproducibilityGate.Data.audit_frame_schema` raises `NumericalSanityError` on any non-`float32` frame as a defensive guard — though that guard has no production caller (C-44).
2. **Raw Output Mandate (carried forward).** Models return their raw expected values. No rounding to integers anywhere in the package.
3. **Non-negativity is a physical floor, applied in the data layer (ruled 2026-09-10).** Fatality counts cannot be negative, so predictions are clipped to `>= 0` before they leave the package — at ingest (`ViewsDataset.ingest_*_predictions`, `clip_negatives=True` by default) and on the streaming path. This is the one semantic floor ADR-010's prohibition does *not* cover: it is domain physics, not a modelling threshold. The maintainer ruled this on issue #40 ("leave as is"); the counter-argument (a negative prediction is a diagnostic signal that clipping erases) is recorded there and in D-05's history, and was heard. Two consequences follow: (a) the opt-out `clip_negatives=False` exists for inspection and should reach every clipping site, including `_predict_streaming`, which today clips unconditionally — a consistency defect, tracked as C-61, not a reopening of this decision; (b) `DartsForecaster`'s Intent Contract may keep "clipped to non-negative" as a guarantee.
4. **Separation of Metrics (carried forward).** Any threshold beyond non-negativity — "fewer than 1.0 means zero" — belongs in the evaluation layer or must be a declared gene. Nothing in this package applies one.

**In scope:** precision, output semantics, the non-negativity floor.
**Out of scope:** the evaluation layer's own thresholds (owned by `views-evaluation`).

---

## Rationale

- ADR-010's three surviving decisions are still correct and still implemented; they should not lose their ADR because one clause failed.
- Silently editing ADR-010 to permit clipping would launder a code behaviour into a decision nobody made. Recording the contradiction and deferring the ruling is the honest form.
- The distinction in decision 3 is real: a floor at zero for counts is defensible as physics; a floor at 1.0 is a modelling choice. The ADR-010 text conflated them.

---

## Considered Alternatives

### Alternative A: patch ADR-010 in place
- **Reason for rejection:** it names two deleted artifacts and one nonexistent constant; a patched ADR-010 would be a new document wearing an old number.

### Alternative B: remove the floor, or make it a declared DNA gene
- **Reason for rejection:** the maintainer ruled (issue #40, 2026-09-10) that the floor is domain physics and stays. A gene would let a config declare that fatalities may be negative, which no consumer wants. The diagnostic cost — a negative prediction is a signal that clipping erases — is real and is why the `clip_negatives=False` opt-out must remain and must reach every site (C-61).

---

## Consequences

### Positive
- The precision and raw-output guarantees are re-anchored to code that exists.
- The clipping contradiction is visible in the register instead of buried in a docstring.

### Negative
- The floor erases one diagnostic signal (a model producing negatives). Mitigation: the opt-out must work everywhere (C-61) so raw outputs can be inspected on demand.

---

## Implementation Notes

- Precision: `views_r2darts2/dataset/converters.py` (all converters emit `float32`); `views_r2darts2/infrastructure/reproducibility_gate.py` (`audit_frame_schema`).
- Clipping sites: `views_r2darts2/dataset/base.py` (`ingest_darts_predictions`, `ingest_numpy_predictions`, parameter `clip_negatives`); `views_r2darts2/engines/darts_forecaster.py:515` (unconditional, streaming path); and two unconditional `np.maximum(arr, 0)` floors on the `log_targets` inverse path, `base.py:1520` and `:1559`, applied before `expm1`. (`base.py:1242`, `:1260` floor *inputs* before `log1p` — a domain guard, not a prediction clip.)
- The only code change this ADR now implies is the consistency fix in C-61: make `clip_negatives=False` reach `_predict_streaming` (and the two `expm1`-path floors, which are domain guards before the inverse transform and may stay unconditional). No change to the default.

---

## Validation & Monitoring

- `tests/test_scaler_selector.py` — transform round-trips.
- `tests/test_parity_e2e.py` — end-to-end precision through fit → predict → inverse.
- `tests/test_views_dataset.py` — ingest paths, including the `clip_negatives` parameter.
- Reconsider this ADR if `views-evaluation` adopts a floor that overlaps with ingest clipping, or if a target with a legitimately signed domain is ever added.

---

## Open Questions

- None. (D-05 was closed by the maintainer's ruling on 2026-09-10; see the register's Resolved Disagreements.)

---

## References

- ADR-010 (superseded), ADR-003, ADR-008, ADR-009.
- `reports/technical_risk_register.md` — D-05 (resolved), C-61.
- `views-r2darts2` issue #40 — the ruling.
- Commit `a79f23b` ("vdsxr") — the `dataset/` rewrite.
