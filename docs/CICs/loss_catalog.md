# Class Intent Contract: LossCatalog

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-02-16  
**Related ADRs:** ADR-001, ADR-003, ADR-006, ADR-009, ADR-013  

---

## 1. Purpose

The `LossCatalog` is a specialized factory responsible for translating abstract DNA manifests into concrete mathematical objective functions (Losses).

> **Its primary goal is to provide a "Genomic Firewall" for mathematical objectives, ensuring that only verified Fortress losses are used and that all mandatory hyperparameters are explicitly declared and numerically valid.**

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** implement the mathematical logic of the loss functions (delegated to `torch.nn.Module` subclasses).
- This class does **not** calculate gradients or perform optimization steps.
- This class does **not** manage the training loop.
- This class does **not** handle data scaling.

---

## 3. Responsibilities and Guarantees

- **Guarantees Genomic Compliance:** Enforces that every loss request includes all mandatory genes (e.g., `p` for Tweedie, `tau` for Quantile).
- **Ensures Fail-Loud Initialization:** Refuses to instantiate any loss with "magic defaults." Every parameter must come from the DNA.
- **Enforces Identifier Mapping:** Maps high-level string identifiers (e.g., "WeightedPenaltyHuberLoss") to concrete, scientifically verified implementation classes.
- **Provides Instructional Errors:** On failure, guarantees an error message that lists all currently authorized loss functions to improve researcher UX.

---

## 4. Inputs and Assumptions

- **Loss Name:** Assumes a string matching an entry in the `LOSS_GENOMES` registry.
- **Keyword Arguments:** Assumes a dictionary of potential loss parameters delivered via the DNA manifest.

---

## 5. Outputs and Side Effects

- **Loss Module:** Produces an instantiated `torch.nn.Module` object ready for backpropagation.
- **Side Effects:** None. This is a stateless factory.

---

## 6. Failure Modes and Loudness

- **Unknown Loss:** Raises `ValueError` if the `loss_name` is not in the whitelist.
- **Missing Genes:** Raises `ValueError` if mandatory hyperparameters for the specific loss are missing or `None`.
- **Numerical Insanity:** Produced loss objects raise `NumericalSanityError` if NaNs are detected during the forward pass (Handshake Principle).

---

## 7. Boundaries and Interactions

- **Upstream:** Orchestrated by `ModelCatalog`.
- **Physical Zen:** Lives in `views_r2darts2/utils/loss/loss_catalog.py`.
- **Execution:** Produced objects are consumed by the Darts model's internal trainer.

---

## 8. Examples of Correct Usage

```python
# Create translator
catalog = LossCatalog(config=dna)

# Get instantiated objective
loss_fn = catalog.get_loss()
```

---

## 9. Examples of Incorrect Usage

- **Direct Loss Import:** Importing `TweedieLoss` directly in the Manager bypassing the catalog (violates ADR-001).
- **Silent Defaults:** Expecting the catalog to "guess" a default `delta` for Huber loss if omitted from DNA (violates ADR-003).

---

## 10. Test Alignment

- **Mathematical Integrity:** `tests/losses/test_mathematical_integrity.py` (Verification of gradients and stability).
- **Green Team:** `tests/test_loss.py` (Validation of registry mapping).

---

## End of Contract

This document defines the **intended meaning** of `LossCatalog`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
