# Reproducibility Manifest & Infrastructure Specification

This document defines the mandatory configuration standards and runtime safety gates for the `views-r2darts2` repository. Its purpose is to ensure that every experiment is 100% reproducible, temporally sound, and free from data leakage.

## 1. The Mandatory Reproducibility Manifest

No model may be initialized, trained, or evaluated unless its configuration explicitly defines the following parameters. The system will **refuse to run** if any manifest key is missing or set to `None`.

### 1.1 Global Execution Context
| Key | Purpose |
| :--- | :--- |
| `random_state` | Forces a fixed seed for weight initialization and stochastic operations. |
| `steps` | Defines the explicit forecast horizon (list of month offsets). |
| `run_type` | Defines the partition context (`calibration`, `validation`, `forecasting`). |

### 1.2 Structural DNA (Architecture)
| Key | Purpose |
| :--- | :--- |
| `input_chunk_length` | Defines the number of past time steps the model sees. |
| `output_chunk_length` | Defines the physical size of the model's output layer. |
| `use_reversible_instance_norm` | Controls normalization behavior (critical for time-series scaling). |

### 1.3 Optimization DNA (Training)
| Key | Purpose |
| :--- | :--- |
| `optimizer_cls` | Explicitly names the optimizer (e.g., "Adam"). |
| `lr` | The learning rate used for gradient descent. |
| `batch_size` | Ensures consistent gradient calculation across runs. |
| `n_epochs` | Fixes the training duration. |
| `loss_function` | Defines the mathematical objective being minimized. |

### 1.4 Inference DNA (Prediction)
| Key | Purpose |
| :--- | :--- |
| `num_samples` | Explicitly sets the number of probabilistic samples (1 for deterministic). |
| `mc_dropout` | Controls whether stochastic dropout is active during inference. |
| `n_jobs` | Sets parallelization for prediction (reproducibility-critical for specific backends). |

---

## 2. Reproducibility Gates (Runtime Validators)

The following gates are implemented in `views_r2darts2/utils/gates.py` and invoked at critical lifecycle points.

### 2.1 The Config Gate (`ConfigAudit`)
*   **Audit Manifest**: Verifies the presence of all keys in Section 1.
*   **Audit Architecture**: Verifies that `len(steps) % output_chunk_length == 0`.
*   **Failure Mode**: `MissingHyperparameterError` or `ArchitectureMismatchError`.

### 2.2 The Temporal Gate (`TemporalAudit`)
*   **The Continuity Guardian ($t+1$)**: Verifies that the test set starts exactly one month after the training set ends.
*   **The Horizon Siren**: Logs a high-visibility warning if `len(steps) != 36`.
*   **The Sequence Auditor**: Scans raw training IDs to ensure a continuous range with **zero holes** (missing months).
*   **Failure Mode**: `TemporalDiscontinuityError` or `ValueError`.

### 2.3 The Data Gate (`DataAudit`)
*   **The Leakage Firewall**: Performs a set-intersection check to guarantee that **zero** time IDs from the test set appear in the training tensors.
*   **Numerical Sanity**: Detects `NaN`, `Inf`, and extreme adversarial outliers ($> 10^9$).
*   **Failure Mode**: `DataLeakageError` or `NumericalSanityError`.

---

## 3. Implementation Patterns

### 3.1 Immutable Snapshots
All manager methods (`_train`, `_evaluate`, `_sweep`) must capture a local snapshot of the configuration:
```python
active_config = self.configs  # Unified, merged dictionary
```
Internal logic must **never** mutate this dictionary.

### 3.2 Workspace Integrity
The test suite and model entry points perform a path-check to ensure `views_r2darts2` is imported from the local project directory, preventing "Ghost Imports" from stale temporary folders.

---

## 4. Team-Based Stress Testing (The Audit Suite)

| Team | Focus | Implementation |
| :--- | :--- | :--- |
| 🟩 **Green** | Resilience | Stochastic Parity tests verify bit-level identity between in-memory and reloaded models. |
| 🟫 **Beige** | Human Error | Manifest audit prevents mundane transcription errors or missing keys in `config_sweep.py`. |
| 🟥 **Red** | Adversarial | Injection tests verify that "Temporal Holes" and "Data Poisoning" (NaNs) are caught. |
