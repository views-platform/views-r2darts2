# Fortress: End-of-the-Road Roadmap

**Status:** Final Hardening Phase  
**Objective:** Finalize the "Fortress" architecture to guarantee absolute scientific integrity, numerical precision, and configuration authority before merging.

---

## Phase 1: Mathematical Hardening & Numerical Integrity
*Focus: Ensuring core operations are robust, explicit, and noise-free.*

### 1.1 Entropy Locking (The Reloading Gap)
*   **The Problem:** MC Dropout distributions drift across save/load cycles because the global RNG state (Torch, Numpy) evolves.
*   **The Solution:** Implement `ReproducibilityGate.Data.lock_entropy(seed)` called inside `forecaster.predict()`.
*   **Action:** Force-reset `torch.manual_seed`, `np.random.seed`, and `cuda.manual_seed` immediately before sampling.

### 1.2 The Great Numerical Purge (Anti-Silent-Fix)
*   **The Problem:** `nan_to_num` or silent clipping masks model instability and data corruption.
*   **The Solution:** Remove all silent correction logic in `utils/loss.py` and `DartsForecaster`.
*   **Action:** Replace "fixes" with explicit `NumericalSanityError` assertions. Fixes must only happen upstream, outside this package.

### 1.3 Loss Function Refactor (Ontological Separation)
*   **The Problem:** A monolithic `loss.py` is difficult to test and maintain.
*   **The Solution:** Move to `views_r2darts2/utils/loss/` directory.
*   **Action:** Split each loss into its own file and standardize the interface via a common base class.

---

## Phase 2: Genomic Handshake (Validation & Alignment)
*Focus: Hardening the ReproducibilityGate to prevent configuration drift.*

### 2.1 Optimizer & Loss Genomes
*   **The Problem:** Missing or unknown hyperparameters for losses and optimizers cause undefined behavior.
*   **The Solution:** Define mandatory genomes for all supported optimizers and custom losses.
*   **Action:** Update `ReproducibilityGate` to raise hard errors on missing OR unknown keys.

### 2.2 Darts & Model Alignment Audit
*   **The Problem:** Misaligned hyperparameters (ghost parameters) create "Silent Lies" about model identity.
*   **The Solution:** Cross-reference `ModelCatalog` with Darts source/docs.
*   **Action:** Verify all exposed HPs exist in Darts, are passed correctly, and that no declared HP remains unused.

### 2.3 Architectural Decomposition (The Triple Catalog)
*   **The Problem:** `ModelCatalog` is currently a "God Object" violating the Single Responsibility Principle. It manages model architecture, loss instantiation (with 20+ arguments), and optimizer genomes. This coupling makes the codebase brittle and violates ADR-003.
*   **The Solution:** Decompose the logic into three specialized **Genome Translators**:
    *   **LossCatalog (Math Factory):** A dedicated class inside `utils/loss/` that owns the `LOSS_GENOMES`. It consumes the DNA, filters for loss-specific parameters, and returns a fully initialized `torch.nn.Module`.
    *   **OptimizerCatalog (Training Engine Factory):** A new utility in `utils/optimizer.py` that owns the `OPTIMIZER_GENOMES`. It handles the handshake between the DNA and `torch.optim` requirements (e.g., momentum, alpha).
    *   **ModelCatalog (Architecture Orchestrator):** A simplified orchestrator that delegates loss and optimizer creation to the specialized catalogs. It focuses purely on architecture-specific parameters and PyTorch Lightning trainer callbacks.
*   **Action:** Refactor `ModelCatalog` and create the new `LossCatalog` and `OptimizerCatalog` entities.

---

## Phase 3: The Red Team Audit (Verification)
*Focus: Verification of scientific integrity.*

### 3.1 Loss Function Test Suite (Red Team)
*   **Action:** Create comprehensive tests for each loss function:
    *   Verify gradients on controlled inputs.
    *   Test edge cases (all-zeros, extreme spikes).
    *   Confirm behavior matches mathematical Intent Contracts.

### 3.2 Contributor Protocol (The Final Law)
*   **Action:** Formalize `docs/contributor_protocols/fortress_protocol.md`:
    *   Define the "No Magic Defaults" standard.
    *   Document the "Crash as Success" philosophy.
    *   Incorporate Phase 2 & 3 of the original roadmap (Automated Scribe postponed).

---

## Phase 4: Deep Fortress Hardening (Post-Falsification)
*Focus: Eliminating all remaining silent inferences identified by the Red Team Falsification Suite.*

### 4.1 Strict Catalog Locking & Whitelisting
*   **The Breach:** Catalogs currently allow unregistered optimizers (like Adagrad) and infer defaults for standard losses (HuberLoss delta=1.0).
*   **The Solution:** 
    *   `OptimizerCatalog` must reject any optimizer not in `OPTIMIZER_GENOMES`.
    *   `LossCatalog` must remove all `.get(key, default)` logic.
    *   Standard losses (MSE, L1) must have explicit empty genomes; `HuberLoss` must have a mandatory `delta`.

### 4.2 ModelCatalog Genomic Firewall
*   **The Breach:** Direct usage of `ModelCatalog` bypasses the Manager's audit, allowing models to be instantiated with missing genes until they crash deep in the framework.
*   **The Solution:** Move `ReproducibilityGate.Config.audit_manifest` inside `ModelCatalog.__init__`.
*   **Action:** Ensure the catalog itself is a gatekeeper, not just a factory.

---

🖖 **"In this repository, a crash is a successful defense of scientific integrity."**
