# ADR-005: Testing as Mandatory Critical Infrastructure

**Status:** Accepted  
**Date:** 2026-02-11  
**Deciders:** Simon Polichinel von der Maase  

---

## Context

In conflict forecasting, failure is not limited to crashes. The most dangerous failures are **silent lies**:
- A model peeking into the future because a partition was sliced incorrectly.
- A run that appears successful but used an implicit default for a critical hyperparameter.
- A stochastic drift that makes results irreproducible across different machines.

Testing is not a quality signal; it is **critical infrastructure** that protects the scientific integrity of the project.

---

## Decision

This repository treats **testing as mandatory critical infrastructure**. All non-trivial functionality must be covered by tests. We use a three-tiered taxonomy to ensure multi-perspective robustness.

---

## Test Taxonomy

### 🟥 Red Team Tests (Adversarial)
- **Goal:** Expose failure modes by deliberately trying to break the "Fortress" invariants.
- **Mindset:** *“How can I make this model lie?”*
- **Views-r2darts2 Examples:**
  - **Temporal Injection:** Passing a dataset with a 1-month hole to see if `ReproducibilityGate` catches it.
  - **Future Peeking:** Attempting to train on data that includes the test partition boundary.
  - **Numerical Poisoning:** Injecting NaNs or Infs into the data stream to ensure the model fails loudly.

### 🟫 Beige Team Tests (Human Error / Realistic Usage)
- **Goal:** Catch failures caused by common researcher mistakes or ambiguous configurations.
- **Mindset:** *“What happens if a researcher forgets a parameter?”*
- **Views-r2darts2 Examples:**
  - **DNA Manifest Audit:** Verifying that a run is blocked if `random_state` is missing.
  - **OCL/Step Mismatch:** Ensuring an error is raised if the forecast horizon isn't a multiple of the output chunk length.
  - **Ghost Imports:** Verifying that the code fails if it tries to import from a temporary/stale folder.

### 🟩 Green Team Tests (Resilience & Correctness)
- **Goal:** Ensure the system works as intended and remains stable over time.
- **Mindset:** *“Is the system solid and reproducible?”*
- **Views-r2darts2 Examples:**
  - **Stochastic Parity:** Verifying that saving and reloading a model produces bit-identical predictions.
  - **Loss Landscape:** Ensuring that custom loss functions produce valid gradients across different data scales.
  - **Scaling Integrity:** Verifying that `FeatureScalerManager` applies correct transforms to group-specific features.

---

## Enforcement Rules

- **Happy Paths are Insufficient:** A PR that only tests that a model "runs" will be rejected. It must include at least one "Red" or "Beige" test if it modifies the core pipeline.
- **Fail-Loud in Tests:** Tests must verify that the system fails loudly when it should. Capturing an exception and doing nothing is forbidden.
- **CI Obligations:** Green team tests form the backbone of our CI. Red and Beige tests ensure the boundaries of that backbone are respected.

---

## Consequences

### Positive
- **High Trust:** Results can be defended as mathematically and temporally sound.
- **Safer Refactoring:** The "Fortress" gates are themselves guarded by tests.
- **Scientific Rigor:** Reproducibility is verified continuously, not just at publication time.

### Negative
- Higher upfront development cost for new model integrations.
- Requires researchers to think like "adversaries" against their own code.

---

## Notes

Testing is the primary mechanism for enforcing ADR-001 (Ontology) and ADR-003 (Authority). If a rule isn't tested, it doesn't exist.
