# ADR-016: Numerical Precision, Raw Output, and the Clipping Question

**Status:** Accepted (supersedes ADR-010)  
**Date:** 2026-09-10  
**Deciders:** Simon Polichinel von der Maase  

---

## Context

ADR-010 (2026-02-11) made four decisions for the 0.1.x codebase: universal `float32`, a raw-output mandate, a prohibition of hardcoded semantic floors in the data and model layers, and separation of "what counts as zero" into the evaluation layer or an explicit DNA gene.

The 0.2.x rewrite (`views_r2darts2/dataset/`, commit `a79f23b`) kept the first, second and fourth. It did not keep the third. `ViewsDataset.ingest_darts_predictions` and `ViewsDataset.ingest_numpy_predictions` in `views_r2darts2/dataset/base.py` take `clip_negatives: bool = True` and, by default, execute `np.maximum(values, 0.0)` on every prediction before it is stored. `DartsForecaster`'s own docstring advertises this as a guarantee: *"Predictions are inverse-transformed and clipped to non-negative."* The 0.1.x forecaster did the same in its prediction-processing method, which ADR-010 named as the enforcement point — so the contradiction predates the rewrite. The rewrite made it a default parameter on the ingest path *and kept a second, unconditional clip* on the streaming path: `views_r2darts2/engines/darts_forecaster.py:515` — `np.maximum(target_values, 0.0, out=target_values)` — with no `clip_negatives` opt-out at all.

ADR-010 also cites a deleted method, a nonexistent epsilon, and a deleted test file. It cannot be patched into truth. It is superseded.

---

## Decision

1. **Universal Precision (carried forward).** All data tensors — targets and covariates — are `float32` before entering any model or loss. On 0.2.x this holds by construction: `ViewsDataset` stores `float32` (`dataset/converters.py` emits nothing else), and `ReproducibilityGate.Data.audit_frame_schema` raises `NumericalSanityError` on any non-`float32` frame as a defensive guard — though that guard has no production caller (C-44).
2. **Raw Output Mandate (carried forward).** Models return their raw expected values. No rounding to integers anywhere in the package.
3. **The clipping question is open.** Whether clipping negative predictions to zero at ingest is (a) a semantic floor that ADR-010 forbade and should be removed or made an explicit DNA gene, or (b) a physical-domain constraint (fatality counts cannot be negative) that belongs in the data layer, is **register D-05**. This ADR does not rule. Until D-05 is ruled, the code's behaviour — clip by default, opt out with `clip_negatives=False` — is the documented state, and every document that describes prediction output says so.
4. **Separation of Metrics (carried forward).** Any threshold beyond non-negativity — "fewer than 1.0 means zero" — belongs in the evaluation layer or must be a declared gene. Nothing in this package applies one.

**In scope:** precision, output semantics, the clipping default.
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

### Alternative B: rule on clipping here
- **Reason for rejection:** it is a modelling decision with downstream consequences for `views-evaluation` and for every published forecast, and the maintainer has not been consulted. It belongs in D-05.

---

## Consequences

### Positive
- The precision and raw-output guarantees are re-anchored to code that exists.
- The clipping contradiction is visible in the register instead of buried in a docstring.

### Negative
- One decision is explicitly unresolved. Readers must check D-05 for the current state.

---

## Implementation Notes

- Precision: `views_r2darts2/dataset/converters.py` (all converters emit `float32`); `views_r2darts2/infrastructure/reproducibility_gate.py` (`audit_frame_schema`).
- Clipping sites: `views_r2darts2/dataset/base.py` (`ingest_darts_predictions`, `ingest_numpy_predictions`, parameter `clip_negatives`); `views_r2darts2/engines/darts_forecaster.py:515` (unconditional, streaming path); and two unconditional `np.maximum(arr, 0)` floors on the `log_targets` inverse path, `base.py:1520` and `:1559`, applied before `expm1`. (`base.py:1242`, `:1260` floor *inputs* before `log1p` — a domain guard, not a prediction clip.)
- No code change is required by this ADR. The change required by D-05's eventual ruling touches two sites — the ingest default and the hardcoded streaming clip — plus `DartsForecaster`'s docstring and `docs/CICs/darts_forecaster.md`.

---

## Validation & Monitoring

- `tests/test_scaler_selector.py` — transform round-trips.
- `tests/test_parity_e2e.py` — end-to-end precision through fit → predict → inverse.
- `tests/test_views_dataset.py` — ingest paths, including the `clip_negatives` parameter.
- Reconsider this ADR if D-05 is ruled, or if `views-evaluation` adopts a floor that overlaps with ingest clipping.

---

## Open Questions

- D-05: keep, remove, or gene-ify `clip_negatives` — at both sites.

---

## References

- ADR-010 (superseded), ADR-003, ADR-008, ADR-009.
- `reports/technical_risk_register.md` — D-05.
- Commit `a79f23b` ("vdsxr") — the `dataset/` rewrite.
