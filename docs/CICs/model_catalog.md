# Class Intent Contract: ModelCatalog

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-02-11  
**Related ADRs:** ADR-001, ADR-002, ADR-003, ADR-006, ADR-009  

---

## 1. Purpose

The `ModelCatalog` acts as the central factory for translating abstract DNA manifests (configurations) into concrete, runnable Darts model instances. 

> **It is the "Translator" that converts researcher intent into mathematical objects, ensuring that every hyperparameter is correctly mapped and validated before training begins.**

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** manage the data pipeline or scaling logic (delegated to the Forecaster).
- This class does **not** execute training or inference (delegated to the Forecaster).
- This class does **not** persist or load model artifacts (delegated to the Manager/Forecaster).
- This class does **not** handle Weights & Biases synchronization.

---

## 3. Responsibilities and Guarantees

- **Guarantees Architectural Alignment:** Validates that the model's `output_chunk_length` is mathematically compatible with the forecast `steps` (ADR-009).
- **Ensures Fail-Loud Initialization:** Guarantees that models are never initialized with "magic defaults." Every architecture-defining parameter must come from the DNA.
- **Delegates Configuration Auditing:** Orchestrates `LossCatalog` and `OptimizerCatalog` to ensure total genomic compliance for optimization and objective functions.
- **Configures Fortress Callbacks:** Automatically attaches mandatory monitoring callbacks (e.g., `GradientHealth`) to every PyTorch Lightning trainer.

---

## 4. Inputs and Assumptions

- **Polymorphic Config:** Assumes a configuration dictionary that has already passed the `ReproducibilityGate.Config.audit_manifest` check (The Genomic Firewall).

---

## 5. Outputs and Side Effects

- **Instantiated Models:** Produces a concrete subclass of `TorchForecastingModel` (e.g., `NBEATSModel`).
- **Side Effects:** None. This is a pure factory.

---

## 6. Failure Modes and Loudness

- **Unknown Algorithm:** Raises `ValueError` if the requested model name is not in the whitelist.
- **Mathematical Mismatch:** Raises `ArchitectureMismatchError` if `steps % output_chunk_length != 0`.

---

## 7. Boundaries and Interactions

- **Upstream:** Managed by `DartsForecastingModelManager`.
- **Physical Zen:** Lives in `views_r2darts2/model/model_catalog.py`.
- **Specialized Factories:** Depends on `LossCatalog` and `OptimizerCatalog`.

---

## 8. Examples of Correct Usage

```python
# Create factory
catalog = ModelCatalog(config=validated_dna)

# Get specific model
model = catalog.get_model("TiDEModel")
```

---

## 9. Examples of Incorrect Usage

- **Direct Instantiation:** Importing `NBEATSModel` from Darts and bypasssing the catalog (loses ADR-009 validation and Fortress callbacks).
- **Manual Callback Management:** Adding custom callbacks to the model *after* retrieval without updating the catalog factory.

---

## 10. Test Alignment

- **Beige Team:** `tests/test_catalog.py` (Exhaustive verification of parameter mapping for all 10 models).
- **Green Team:** `tests/test_loss.py` (Integration with LossSelector).

---

## 11. Evolution Notes

### Known Deviations / Technical Debt
- **Shared Weights Logic:** Currently, `input_chunk_length` and `output_chunk_length` logic is repeated across multiple `_get_X` methods. This should be refactored into a shared base builder method.
- **Norm Type Defaults:** Some models still have hardcoded string defaults for `norm_type` (e.g., "RMSNorm" in TFT). These should be moved to the DNA genome.

---

## End of Contract

This document defines the **intended meaning** of `ModelCatalog`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
