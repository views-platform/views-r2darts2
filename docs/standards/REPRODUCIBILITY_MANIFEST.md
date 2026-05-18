# Reproducibility Manifest & Infrastructure Specification

This document defines the mandatory configuration standards and runtime safety gates for the `views-r2darts2` repository. Its purpose is to ensure that every experiment is 100% reproducible, temporally sound, and free from data leakage.

## 1. The Mandatory Reproducibility Manifest (DNA)

No model may be initialized, trained, or evaluated unless its configuration explicitly defines the following parameters. The DNA is **polymorphic**: the requirements are determined dynamically based on the chosen algorithm. The system will **refuse to run** if any manifest key is missing or set to `None`.

### 1.1 The Core Genome (Universal)
Required by ALL experiments regardless of model.

| Key | Purpose |
| :--- | :--- |
| `random_state` | Forces a fixed seed for weight initialization and stochastic operations. |
| `steps` | Defines the explicit forecast horizon (list of month offsets). |
| `run_type` | Defines the partition context (`calibration`, `validation`, `forecasting`). |
| `algorithm` | The specific model architecture to instantiate. |
| `loss_function` | The mathematical objective being minimized. |
| `optimizer_cls` | Explicitly names the optimizer class (e.g., "Adam"). |
| `lr`, `weight_decay` | Standard optimization hyperparameters. |
| `batch_size`, `n_epochs` | Global training control. |
| `num_samples`, `mc_dropout` | Inference behavior (probabilistic vs deterministic). |

### 1.2 The Algorithm-Specific Genome
Each architecture (N-BEATS, TFT, TiDE, etc.) defines its own mandatory "genes" (e.g., `num_stacks`, `use_static_covariates`). Parameters irrelevant to an architecture are forbidden in its manifest to prevent semantic bloat.

---

## 2. Reproducibility Gates (The Fortress)

The following gates are implemented in `views_r2darts2/utils/reproducibility_gate.py` and invoked at critical lifecycle points.

### 2.1 The Config Gate (`ConfigAudit`)
*   **Audit Manifest**: Performs a dynamic, model-aware audit of the DNA.
*   **Audit Architecture**: Verifies that `len(steps) % output_chunk_length == 0` (ADR-009).
*   **Triple Catalog Firewall**: 
    - `ModelCatalog`, `LossCatalog`, and `OptimizerCatalog` perform secondary audits during instantiation.
    - **Refuse-to-Guess Invariant**: Catalogs will raise `MissingHyperparameterError` if any mandatory DNA key is missing or `None`.
*   **Failure Mode**: `MissingHyperparameterError` or `ArchitectureMismatchError`.

### 2.2 The Temporal Gate (`TemporalAudit`)
*   **The Continuity Guardian ($t+1$)**: Verifies that the test set starts exactly one month after the training set ends.
*   **The Horizon Siren**: Logs a high-visibility warning if `len(steps) != 36`.
*   **The Sequence Auditor**: Scans training IDs to ensure a continuous range with **zero holes**.

### 2.3 The Hardware Gate (`HardwareAudit`)
*   **Device Self-Healing**: Audits the model device before every prediction. If a Darts-induced CPU-drift is detected, the model is restored to its target device (ADR-011).
*   **Parallelism Lockdown**: Forces `max_workers=1` for GPU prediction to prevent race conditions.

### 2.4 The Data Gate (`DataAudit`)
*   **Numerical Integrity**: Enforces `float32` standardization and detects `NaN`/`Inf` at the system boundaries (ADR-010).
*   **The Leakage Firewall**: Set-intersection check between train and test partitions.

---

## 3. Implementation Patterns

### 3.1 Immutable Snapshots
All manager methods must capture a local snapshot of the configuration: `active_config = self.configs`. Internal logic must **never** mutate this dictionary.

### 3.2 Global Calibration
All target scalers must use `global_fit=True` to preserve cross-sectional signals and probabilistic calibration (ADR-012).

---

## 4. Team-Based Stress Testing (The Audit Suite)

| Team | Focus | Implementation |
| :--- | :--- | :--- |
| 🟩 **Green** | Resilience | Stochastic Parity and Scaling Integrity tests. |
| 🟫 **Beige** | Human Error | Polymorphic DNA audits and Mismatch detection. |
| 🟥 **Red** | Adversarial | Temporal Injection and Numerical Poisoning. |
