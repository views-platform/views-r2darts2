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

🖖 **"In this repository, a crash is a successful defense of scientific integrity."**
