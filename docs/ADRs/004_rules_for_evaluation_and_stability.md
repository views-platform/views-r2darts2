# ADR-004: Rules for Evolution and Stability

**Status:** Accepted  
**Date:** 2026-02-11  
**Deciders:** Simon Polichinel von der Maase  

---

## Context

A research repository in conflict forecasting must balance two opposing forces:
1. **Stability:** The need for consistent, reproducible results over years.
2. **Evolution:** The need to integrate new deep learning architectures and data sources rapidly.

If everything is stable, research stagnates. If everything is experimental, results become untrustworthy. We need a tiered approach to stability.

---

## Decision

We define three **Stability Tiers** for the components of this repository. The stability of a component determines the "cost" of changing it.

### Tier 1: The Fortress (High Stability)
- **Scope:** `utils/gates.py`, `utils/nbeats_patch.py`, and the mathematical definitions of standard losses in `utils/loss.py`.
- **Guarantee:** These define the physical laws of the system (e.g., "thou shalt not peek into the future").
- **Change Rule:** Changes require a new ADR or a superseding of an existing one. Breaking a "Gate" is considered a critical regression.

### Tier 2: The DNA Schema (Medium Stability)
- **Scope:** `REPRODUCIBILITY_MANIFEST.md` and the `MANDATORY_MANIFEST` list in `gates.py`.
- **Guarantee:** Defines what an experiment *must* declare.
- **Change Rule:** New parameters can be added (evolving the schema), but removing or renaming existing mandatory parameters requires updating all active `sweep_configs` and existing artifacts.

### Tier 3: Models & Managers (Research Stability)
- **Scope:** `ModelCatalog`, `DartsForecaster`, `DartsForecastingModelManager`.
- **Guarantee:** These are expected to evolve as we add new model types (e.g., TiDE, TSMixer) or optimization strategies.
- **Change Rule:** Evolution is encouraged. Breaking changes are permitted during research sprints, provided that `Tier 1` and `Tier 2` invariants are maintained.

---

## Backward Compatibility of Artifacts

- **The Golden Rule:** A trained model artifact must always be loadable and evaluatable by the version of the code that created it.
- **Divergence:** If a code change makes older artifacts unreadable, the change must include a migration script or the old code must be preserved in a versioned submodule.

---

## Consequences

### Positive
- **Trust:** Researchers can trust that the "Fortress" won't let them accidentally lie with data.
- **Agility:** Engineers can refactor the `manager` layer without fearing they are violating "Core Laws."
- **Clarity:** It's clear which PRs require high-level architectural review (Tier 1) vs. standard code review (Tier 3).

### Negative
- Tier 1 changes become slower and more bureaucratic.
- Requires maintaining a "Fortress" mindset even during fast research.

---

## Notes

Stability is a design constraint, not a preference. This ADR ensures that we don't accidentally "innovate" away our reproducibility guarantees.
