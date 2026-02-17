# Class Intent Contract: OptimizerCatalog

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-02-16  
**Related ADRs:** ADR-001, ADR-003, ADR-006, ADR-009, ADR-013  

---

## 1. Purpose

The `OptimizerCatalog` is a specialized factory responsible for translating abstract DNA optimization manifests into concrete PyTorch optimizer classes and their associated hyperparameters.

> **Its primary goal is to provide a "Genomic Firewall" for optimization, ensuring that only whitelisted optimizers are used and that all mandatory hyperparameters are explicitly declared.**

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** perform optimization steps or manage gradients.
- This class does **not** manage the training loop (delegated to PyTorch Lightning).
- This class does **not** implement new optimization algorithms (uses `torch.optim`).
- This class does **not** manage Learning Rate Schedulers (delegated to `ModelCatalog`).

---

## 3. Responsibilities and Guarantees

- **Guarantees Genomic Compliance:** Enforces that every optimizer request includes all mandatory genes (e.g., `lr` and `weight_decay` for Adam).
- **Ensures Whitelisted Access:** Refuses to instantiate any optimizer not explicitly registered in the Fortress whitelist.
- **Provides Instructional Errors:** On failure, guarantees an error message that lists all currently authorized optimizers to improve researcher UX.
- **Translates Hyperparameters:** Correctly maps DNA keys to the specific keyword arguments expected by PyTorch optimizer constructors.

---

## 4. Inputs and Assumptions

- **Config Snapshot:** Assumes a raw dictionary containing an `optimizer_cls` key and potential hyperparameter keys.
- **PyTorch Registry:** Assumes that whitelisted names correspond to attributes in the `torch.optim` module.

---

## 5. Outputs and Side Effects

- **Optimizer Class:** Returns the uninstantiated class type from `torch.optim`.
- **Keyword Arguments:** Returns a validated dictionary of hyperparameters ready for injection.
- **Side Effects:** None. This is a stateless translator.

---

## 6. Failure Modes and Loudness

- **Unregistered Optimizer:** Raises `ValueError` if the requested optimizer is not in the `OPTIMIZER_GENOMES` whitelist.
- **Missing Genes:** Raises `ValueError` if mandatory hyperparameters for the specific optimizer are missing or `None`.
- **Invalid PyTorch Name:** Raises `ValueError` if a whitelisted name is missing from `torch.optim` (safety check for registry drift).

---

## 7. Boundaries and Interactions

- **Upstream:** Orchestrated by `ModelCatalog`.
- **Physical Zen:** Lives in `views_r2darts2/utils/optimizer_catalog.py`.
- **Downstream:** Produced configurations are consumed by Darts models during their `configure_optimizers` phase.

---

## 8. Examples of Correct Usage

```python
# Create translator
catalog = OptimizerCatalog(config=dna)

# Get class and validated kwargs
opt_cls = catalog.get_optimizer_cls()
opt_kwargs = catalog.get_optimizer_kwargs()

# Instantiate (usually handled by model framework)
optimizer = opt_cls(params, **opt_kwargs)
```

---

## 9. Examples of Incorrect Usage

- **Direct torch.optim Import:** Importing `Adam` directly in the Manager bypassing the catalog (violates ADR-001).
- **Silent Defaults:** Passing a config with missing `lr` and expecting the catalog to provide a default (violates ADR-003).

---

## 10. Test Alignment

- **Red Team:** `tests/test_model_catalog.py` (Validation of invalid optimizer failure).
- **Green Team:** `tests/test_genomic_handshake.py` (Verification of parameter mapping).

---

## End of Contract

This document defines the **intended meaning** of `OptimizerCatalog`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
