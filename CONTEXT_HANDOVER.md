# Fortress Handover: System Context & Architectural Invariants

**Date:** 2026-02-12  
**Status:** PEAK GOOD STATE  
**Security Level:** CRITICAL (Fortress Protocol Active)

---

## 1. The Core Mandate
This repository is no longer a standard ML pipeline; it is a **Fortress**. Every decision is governed by **ADRs 000-012** and **9 Class Intent Contracts (CICs)**. 

> **Authority Principle (ADR-003):** The documentation (ADRs/CICs) is the absolute authority. If the code deviates from the contract, the code is a bug. **Never infer semantics; only trust declarations.**

---

## 2. The DNA Genome (Configuration)
The "Mandatory Manifest" is now **Polymorphic**. 
- **CORE_GENOME:** Parameters required by ALL experiments (e.g., `algorithm`, `loss_function`, `random_state`).
- **ALGORITHM_GENOMES:** Parameters specific to an architecture (e.g., `num_stacks` for N-BEATS).
- **Rule:** If a model cannot mathematically consume a parameter, it **must not** exist in that model's manifest. This is enforced by `ReproducibilityGate.Config.audit_manifest`.

---

## 3. Physical Laws (Non-Negotiable)

### A. Numerical Precision (ADR-010)
- **Law:** Everything is `float32`. 
- **The Airlock:** `_ViewsDatasetDarts` physically downcasts all data at entry.
- **The Silent Lie:** We eradicated the silent conversion of NaNs to 0.0. If a model outputs NaNs, the system **must fail loud** via `NumericalSanityError`.

### B. Hardware Integrity (ADR-011)
- **The Darts Bug:** Darts shifts models to CPU post-training. 
- **The Fix:** `DartsForecaster` implements **Verify-and-Restore** before every prediction.
- **The Parallelism Rule:** GPU prediction **must** be sequential (`max_workers=1`). CPU prediction can be parallel.

### C. Scaling Calibration (ADR-012)
- **Law:** Standardized on Darts native `Pipeline`. 
- **The Guarantee:** All target scalers **must** use `global_fit=True` to preserve the 3D sample dimension for probabilistic calibration.

---

## 4. Key Infrastructure Pointers

- **`ModelCatalog`**: The only way to instantiate models. It is DRY and handles all boilerplate (Fortress callbacks, optimizer mapping).
- **`ReproducibilityGate`**: The central firewall. It contains the logic for Temporal ($t+1$), Data (Leakage), and Config (Genome) audits.
- **`DartsForecastingModelManager`**: Pure orchestrator. Does no math, does no direct I/O (delegates to `_ViewsDatasetDarts.from_views_path`).

---

## 5. Contributor Protocol (ADR-007)
- **Anti-Truncation:** Never use `write_file` on existing files. Use scoped `replace` only.
- **Fail-Loud:** If you encounter a "best-effort" fallback in the code, replace it with an explicit exception.

---

## 6. Pending Roadmap (Priority)
1.  **Entropy Locking:** We still need to force-seed RNGs inside `predict()` to ensure bit-perfect MC Dropout results across reloads.
2.  **Automated Scribe:** A utility to sync `config_hyperparameters.py` directly from a validated WandB run ID.

---

**Final Instruction:** Run `pytest` immediately. All 377 tests must pass. If they don't, the Fortress is breached. 🖖
