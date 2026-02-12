# ADR-003: Authority of Declarations Over Inference

**Status:** Accepted  
**Date:** 2026-02-11  
**Deciders:** Simon Polichinel von der Maase  

---

## Context

Reproducibility in conflict forecasting is often compromised by "magic" behavior — logic that depends on filenames, folder structures, or implicit defaults buried in library code. In such systems, two identical-looking runs can produce different results because a hidden state was inferred from the environment.

We need a system where the behavior of any component is explicitly declared, not guessed.

---

## Decision

In this repository:

> **All meaningful semantics must be explicitly declared in the Configuration Manifest (DNA). Inference of semantics across component boundaries is forbidden.**

The DNA manifest—delivered by `views_pipeline_core` from upstream `views_models` definitions—is the absolute authority for system behavior. If a required declaration is missing, ambiguous, or contradictory, the system **must fail loudly and immediately.**

---

## Global Invariant: Fail Loud on Ambiguity

Silent failure is considered a bug. Warning-only behavior, implicit fallbacks, or "best-effort" inference are **forbidden** for any decision-relevant semantics.

This includes:
- Raising explicit runtime errors during initialization if DNA is incomplete.
- Refusing to load a model if its manifest is missing or altered.
- Failing a run if temporal boundaries in the data don't match the declared partition.

---

## Rules of Semantic Authority

- **Explicit over Magic:** If a parameter matters, it must be declared. No logic may be triggered by naming patterns or directory nesting.
- **Authority of the DNA:** If a manifest says a model uses `AsymmetricQuantileLoss`, it *uses* that loss, regardless of where the script is located or what the model name suggests.
- **No Parameter Spillover:** Authority is limited to relevance. A declaration in the DNA only has authority if it corresponds to a recognized gene for the chosen algorithm. Forcing a model to declare a parameter it cannot consume (e.g., `use_static_covariates` for N-BEATS) is a violation of the principle of Intentionality.
- **No Implicit Fallbacks:** "Sensible defaults" are forbidden for parameters affecting model identity (stochastic seeds, loss hyperparameters, feature sets).

---

## Examples of Forbidden Behavior

- **Naming-based logic:** "If the model name contains 'log', apply log scaling." -> **Forbidden.** The manifest must explicitly include `"log_targets": True`.
- **Structural inference:** "If this feature is in the 'wdi' folder, use StandardScaler." -> **Forbidden.** The manifest must map the feature group to the scaler.
- **Shape-based guessing:** Inferring whether a forecast is probabilistic based on the tensor shape. -> **Forbidden.** The manifest must declare `num_samples`.

---

## Consequences

### Positive
- **Guaranteed Reproducibility:** If the manifests match, the logic matches.
- **Observability:** You can understand a model's full behavior just by looking at its configuration.
- **Auditability:** Errors are caught at initialization time (via `ReproducibilityGate`).

### Negative
- Manifests become large and verbose.
- More upfront work is required to define feature mappings and scaling strategies.

---

## Notes

This ADR establishes *how* we know what to do. The specific contents of the manifest are defined in the `REPRODUCIBILITY_MANIFEST.md`. The validation logic is implemented in `views_r2darts2/utils/gates.py`.

