# Class Intent Contract: LossSelector

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-02-11  
**Related ADRs:** ADR-001, ADR-003, ADR-006, ADR-009  

---

## 1. Purpose

The `LossSelector` is a specialized factory responsible for instantiating objective functions (Losses) for deep learning models. 

> **Its primary goal is to provide a unified interface for retrieving both standard PyTorch losses and custom "Fortress" losses while ensuring that all architecture-specific hyperparameters are correctly injected.**

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** implement the mathematical logic of the loss functions (delegated to `torch.nn.Module` subclasses).
- This class does **not** calculate gradients or perform optimization steps.
- This class does **not** manage the DNA manifest directly (it consumes filtered arguments from the `ModelCatalog`).
- This class does **not** handle data scaling.

---

## 3. Responsibilities and Guarantees

- **Guarantees Dynamic Injection:** Automatically inspects the constructor signatures of loss modules and injects only the matching hyperparameters from the provided configuration.
- **Enforces Identifier Mapping:** Maps high-level string identifiers (e.g., "TweedieLoss") to concrete, verified implementation classes.
- **Ensures Fail-Loud Selection:** Guarantees that requests for unknown loss functions raise immediate and descriptive errors.
- **Provides Unified Standard/Custom Access:** Acts as a bridge, allowing the rest of the pipeline to treat standard PyTorch losses (`MSELoss`) and complex domain losses (`SpikeFocalLoss`) identically.

---

## 4. Inputs and Assumptions

- **Loss Name:** Assumes a string matching one of the supported identifiers in the internal registry.
- **Keyword Arguments:** Assumes a flat dictionary of potential loss parameters (e.g., `delta`, `tau`, `alpha`).
- **Signature Trust:** Assumes that custom loss modules follow standard PyTorch `__init__` patterns.

---

## 5. Outputs and Side Effects

- **Loss Module:** Produces an instantiated `torch.nn.Module` object ready for forward-pass computation.
- **Side Effects:** None. This is a stateless factory.

---

## 6. Failure Modes and Loudness

- **Unknown Loss:** Raises `ValueError` if the `loss_name` is not in the registry.
- **Instantiation Failure:** Raises `TypeError` if mandatory constructor arguments (not filtered out) are missing.
- **Parameter Conflict:** Logs no warnings if extra parameters are passed (they are silently filtered), as the `ModelCatalog` provides a broad set of candidate keys.

---

## 7. Boundaries and Interactions

- **Upstream:** Orchestrated by `ModelCatalog`.
- **Registry:** Maintains a closed list of approved loss implementations.
- **Execution:** Produced objects are consumed by the Darts model's internal trainer.

---

## 8. Examples of Correct Usage

```python
# Instantiating a standard loss
mse_fn = LossSelector.get_loss_function("MSELoss")

# Instantiating a complex custom loss with filtered arguments
custom_fn = LossSelector.get_loss_function(
    "TweedieLoss", 
    p=1.5, 
    irrelevant_param=42  # Will be ignored
)
```

---

## 9. Examples of Incorrect Usage

- **Direct Import:** Importing `SpikeFocalLoss` directly in the Manager bypassing the selector (violates ADR-001).
- **Manual Signature Filtering:** Manually checking `inspect.signature` in the `ModelCatalog` instead of trusting the selector.

---

## 10. Test Alignment

- **Beige Team:** `tests/test_loss.py` (Validation of registry mapping).
- **Green Team:** `tests/test_loss.py` (Verification of hyperparameter injection accuracy).

---

## 11. Evolution Notes

### Known Deviations / Technical Debt
- **Missing Genome Enforcement:** While the selector filters parameters, it doesn't currently *validate* that the required parameters for a specific loss (e.g., `p` for Tweedie) are present. This validation is currently deferred to the `ReproducibilityGate`. The selector should ideally have its own mini-handshake to guarantee the loss is mathematically complete.

---

## End of Contract

This document defines the **intended meaning** of `LossSelector`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
