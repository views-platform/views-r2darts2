# The Fortress Protocol: Contributor Governance

This document defines the mandatory engineering and mathematical standards for the `views-r2darts2` repository. Adherence to this protocol is required for all contributions to guarantee absolute scientific integrity and reproducibility in conflict forecasting.

---

## 1. Core Principles

### A. The Authority of Declarations (ADR-003)
**"Never infer; only trust declarations."**
All meaningful semantics (architectures, loss functions, scaling strategies, seeds) must be explicitly declared in the Configuration Manifest (DNA). 
- **Prohibited:** Filename-based logic, directory-structure inference, or shape-based guessing.
- **Requirement:** If a parameter affects model identity, it must be a mandatory gene in the `ReproducibilityGate`.

### B. The Fail-Loud Mandate (ADR-008)
**"A crash is a successful defense of scientific integrity."**
Silent failures, implicit fallbacks, and "best-effort" corrections are forbidden. 
- **Requirement:** Violations of physical, temporal, or configuration invariants must raise an explicit `ReproducibilityError` or `NumericalSanityError` immediately.
- **Prohibited:** Using `nan_to_num`, silent clipping, or "sensible defaults" for critical parameters.

### C. The Numerical Airlock (ADR-010)
All data entering the system must pass through a numerical airlock.
- **Requirement:** Downcast all input to `float32` immediately.
- **Requirement:** Detect and raise errors on NaNs or Infs at every boundary (Data entry, Loss calculation, Prediction output).

---

## 2. Contributor Requirements

### Adding a New Model
1.  **Define the Genome:** Register the mandatory hyperparameters in `ReproducibilityGate.Config.ALGORITHM_GENOMES`.
2.  **Register in Catalog:** Add the instantiation logic to `ModelCatalog`.
3.  **Audit Alignment:** Ensure all DNA parameters are passed correctly to the underlying framework (Darts/Torch).

### Adding a New Loss Function
1.  **Modularize:** Create a dedicated file in `views_r2darts2/utils/loss/`.
2.  **Enforce Sanity:** Implement explicit NaN/Inf checks in the `forward()` method.
3.  **Register Genome:** Add mandatory hyperparameters to `ReproducibilityGate.Config.LOSS_GENOMES`.
4.  **Update Factory:** Add the class to `LossCatalog`.

---

## 3. Mandatory Testing Taxonomy (ADR-005)

Every Pull Request must include tests covering the following three perspectives:

### 🟩 Green Team (Stability & Correctness)
*   **Goal:** Ensure the system works as intended and remains stable.
*   **Examples:** Gradient verification, stochastic parity (bit-identical reloads), scaling integrity.

### 🟫 Beige Team (DNA & Human Error)
*   **Goal:** Catch failures caused by common configuration mistakes or missing parameters.
*   **Examples:** Manifest audits (blocking runs if a gene is missing), OCL/Step alignment verification.

### 🟥 Red Team (Adversarial)
*   **Goal:** Expose failure modes by deliberately trying to make the model lie.
*   **Examples:** Injecting temporal holes (missing months), future-peeking injections, numerical poisoning (NaN/Inf injection).

---

## 4. Operational Invariants

- **Hardware Self-Healing:** Models must implement Verify-and-Restore to prevent CPU-drift (ADR-011).
- **GPU Sequentialism:** Parallel prediction is forbidden on GPUs to prevent race conditions.
- **Entropy Locking:** All probabilistic sampling must be preceded by a seed reset via `ReproducibilityGate.Data.lock_entropy`.

---

🖖 **"In this repository, we value bit-perfect reproducibility over convenient execution."**
