# ADR-017: Entity Presence Is Decided at the Last Observed Month

**Status:** Accepted  
**Date:** 2026-09-10 (decided in code by the maintainer, `99f00ec`); recorded 2026-09-19  
**Deciders:** Dylan (maintainer), in code  
**Recorded by:** Simon Polichinel von der Maase  
**Consulted:** views-pipeline-core ADR-064 (prediction-boundary entity check)

---

## Context

`ViewsDataset` stores a dense `(time, entity, sample, feature)` grid. Every entity that has ever
appeared in a source occupies a row for every time step, whether or not it existed at that time.
Before 0.2.2, entities that had ceased to exist — dissolved states, retired grid cells — were
therefore carried through training and forecasting on all-zero inputs, and the forecasts for them
were delivered as if they were real.

Two things made this untenable at once. views-pipeline-core 3.3 (ADR-064 there) refuses a
DataFrame-path prediction that carries an entity absent from the input's last observed month, so
such forecasts are now rejected at the boundary rather than silently accepted. And the 0.1.x
line's `min_length` filter — which had incidentally excluded most such entities by requiring a
minimum history — was lost in the 0.2.x rewrite (register C-35), so nothing in 0.2.0/0.2.1 removed
them at all.

The question the code had to answer is: **what does it mean for an entity to exist in a dataset?**

---

## Decision

An entity exists in an observational dataset if and only if it is **present at the source's final
timestamp**. At ingest, every observational source is reduced to those entities.

1. **Presence** means: for a `DataFrame`, `FeatureFrame` or parquet source, a row exists for the
   entity at the maximum `time_id`; for an `xarray.Dataset` source, at least one data variable is
   non-NaN for the entity at the maximum `time_id`.
2. **Default on.** `ViewsDataset(..., filter_entities_at_end=True)` is the default and is passed
   through to every converter. Passing `False` disables it for that dataset.
3. **Prediction sources are never filtered.** A source with `pred_*` columns, or an `xarray`
   dataset carrying `is_prediction=True`, bypasses the filter unconditionally — predictions are
   about the entities the model was given, not about presence.
4. **Empty sources fail loudly.** An observational source with no rows, or with no entity present
   at its final timestamp, raises `ValueError`.
5. **Every dropped entity is logged** at `INFO` with the final timestamp, the counts before and
   after, and the full list of dropped ids.

---

## Rationale

- The last observed month is the only presence signal the grid carries. There is no separate
  "entity exists" table; a row at the final time is the closest thing to one.
- Forecasting an entity that cannot receive a prediction is wasted compute, and after ADR-064 it is
  also a refusal at the boundary. Removing it at ingest keeps both training and prediction honest
  and makes the refusal unreachable rather than merely handled.
- A history-length criterion (the 0.1.x approach) answers a different question. An entity can have
  a long history and be dissolved; a new entity can have a short history and be real.

---

## Considered Alternatives

### A. Restore a minimum-history-length filter (the 0.1.x `min_length`)
Rejected. Length is not presence: it keeps dissolved states with long histories and drops young,
real entities.

### B. Filter at prediction time only
Rejected. The model would still train on phantom rows of zeros, which shapes what it learns about
zero-inflation; and every consumer of the training set would still see the phantoms.

### C. Rely on the pipeline-core boundary refusal alone
Rejected. A refusal at the end of an hour-long run is a poor place to learn the input was wrong.

---

## Consequences

### Positive
- Forecasts are produced only for entities that exist at the end of the observational window.
- The set of entities in an output is determined by the data, once, at ingest, and is logged.
- The structural-sparsity half of D-07 is settled: cells that are NaN *because the entity is absent*
  no longer reach the Darts path, so any NaN that remains there is a genuinely missing observation.

### Negative
- An entity that legitimately has no row at the final month — late-arriving data, a partially
  loaded source — is dropped with nothing but an `INFO` line to say so. There is no error signal.
  Registered as C-63.
- "Present" is decided by the source's own last timestamp, not by a declared window. A source cut
  short by a fetch problem silently redefines who exists.

---

## Implementation Notes

- Flag: `views_r2darts2/dataset/base.py:48` (`filter_entities_at_end: bool = True`), forwarded to
  every converter.
- Frame sources: `views_r2darts2/dataset/converters.py:47` (`filter_frame_entities_at_end`), called
  from `DataFrameConverter` (`:218`), `FeatureFrameConverter` (`:367`) and `ParquetConverter`
  (`:617`); `PredictionFrameConverter` does not call it.
- xarray sources: `views_r2darts2/dataset/converters.py:69` (`filter_dataset_entities_at_end`),
  which returns the dataset untouched when `is_prediction` is set.
- Logging: `views_r2darts2/dataset/converters.py:30` (`_log_entity_filter`).
- No change to the Darts path, the scalers, or the prediction ingest.

---

## Validation & Monitoring

- `tests/test_views_dataset.py` — nine tests: default filtering for DataFrame, parquet, FeatureFrame
  and xarray sources; the opt-out; presence-by-row-not-value; the log line; the empty-source
  `ValueError`; and the prediction-frame bypass.
- Monitoring: the `Entity-at-end filter:` log line reports `dropped=N` on every ingest. A
  `dropped` count that differs between two runs of the same queryset is the signal that the
  source's final month changed.

---

## Open Questions

- Whether a dropped-entity count above some threshold should be an error rather than a log line
  (the C-63 residual). Not decided here.

---

## References

- views-pipeline-core ADR-064 — prediction-boundary entity check.
- `reports/technical_risk_register.md` — C-35 (resolved by this decision), C-63 (residual), D-07.
- views-r2darts2 issue #39 — the NaN-policy thread this decision partly settles.
- Commit `99f00ec` ("fix country exclusion logic") — the decision as made.
