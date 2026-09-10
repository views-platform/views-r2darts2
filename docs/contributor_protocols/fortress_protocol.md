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
  *Note (2026-09-10):* the data layer currently clips predictions to non-negative (`ViewsDataset.ingest_*_predictions` default `clip_negatives=True`; `DartsForecaster._predict_streaming` unconditionally) — register **D-05** / ADR-016 — and applies `np.nan_to_num(nan=0.0)` on the Darts path for structural sparsity (`dataset/base.py:1645`) — register **D-07**. Both await a ruling.

### C. The Numerical Airlock (ADR-016, superseding ADR-010)
All data entering the system must pass through a numerical airlock.
- **Requirement:** Downcast all input to `float32` immediately.
- **Requirement:** Detect and raise errors on NaNs or Infs at every boundary (Data entry, Loss calculation, Prediction output).

### D. Physical Symmetrical Architecture (ADR-013)
**"1 Class, 1 File, 1 Name."**
Organizational Zen is a requirement for maintainability.
- **Requirement:** Every non-trivial class must live in its own file named after the class in `snake_case`.
- **Requirement:** Heterogeneous logic (callbacks, patches, exceptions) must be consolidated into pre-defined symmetrical hubs (`views_r2darts2/infrastructure/callbacks.py`, `views_r2darts2/infrastructure/patches.py`, `views_r2darts2/infrastructure/exceptions.py`). Three homogeneous-family files are contested under this rule — register **D-04**.

---

## 2. Contributor Requirements

### Adding a New Model
Models are Darts classes, not local files. There is nothing to create under `views_r2darts2/`; there are two registries to keep in sync (register C-11):
1.  **Define the Genome:** Register mandatory hyperparameters in `ReproducibilityGate.Config.ALGORITHM_GENOMES` (`views_r2darts2/infrastructure/reproducibility_gate.py`).
2.  **Register in Catalog:** Add a `_get_<model>` factory and its registry entry in `ModelCatalog` (`views_r2darts2/catalogs/model_catalog.py`). A genome without a factory passes the manifest audit and then crashes opaquely.

### Adding a New Loss Function
1.  **Symmetrical Entry:** Create `views_r2darts2/math/<my_new_loss>.py` and export it from `views_r2darts2/math/__init__.py`. *(Three existing passthrough modules — `huber_loss.py`, `logcosh_loss.py`, `mse_loss.py` — are not exported and are shadowed or unregistered; register C-46.)*
2.  **Enforce Sanity:** Implement explicit NaN/Inf checks in the `forward()` method.
3.  **Register Genome:** Add mandatory hyperparameters to `ReproducibilityGate.Config.LOSS_GENOMES`.
4.  **Update Catalog:** Add the class to `LossCatalog` (`views_r2darts2/catalogs/loss_catalog.py`).
5.  **Write the Loss Card:** Add `docs/loss_cards/<name>_spec.md` and index it in `docs/loss_cards/README.md`. Do not list constructor "defaults" — every gene is mandatory (ADR-003).

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
