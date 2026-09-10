# Reproducibility Manifest & Infrastructure Specification

This document defines the mandatory configuration standards and runtime safety gates for the `views-r2darts2` repository. Its purpose is to ensure that every experiment is 100% reproducible, temporally sound, and free from data leakage.

## 1. The Mandatory Reproducibility Manifest (DNA)

No model may be initialized, trained, or evaluated unless its configuration explicitly defines the following parameters. The DNA is **polymorphic**: the requirements are determined dynamically based on the chosen algorithm. The system will **refuse to run** if any manifest key is missing or set to `None` — except the six architecture keys in `ReproducibilityGate.Config.NULLABLE_PARAMS` (`hidden_fc_sizes`, `pooling_kernel_sizes`, `n_freq_downsample`, `categorical_embedding_sizes`, `temporal_hidden_size_past`, `temporal_hidden_size_future`), for which `None` is a legal declared value.

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
| `name` | Experiment name; used for artifact and run identification. |
| `lr_scheduler_cls` | Explicitly names the learning-rate scheduler class (e.g., "ReduceLROnPlateau", "WarmupCAWR"). |
| `early_stopping_patience` | Epochs without `val_loss` improvement before training halts. |
| `early_stopping_min_delta` | Minimum `val_loss` change that counts as improvement. |
| `gradient_clip_val` | Gradient-norm clip value passed to the PyTorch Lightning trainer. |

*(Authoritative list: `ReproducibilityGate.Config.CORE_GENOME` in `views_r2darts2/infrastructure/reproducibility_gate.py` — 17 keys as of `development` @ `fe7e681`. This table was missing 5 of them until 2026-09-10.)*

### 1.2 The Algorithm-Specific Genome
Each architecture (N-BEATS, TFT, TiDE, etc.) defines its own mandatory "genes" (e.g., `num_stacks`, `use_static_covariates`) in `ReproducibilityGate.Config.ALGORITHM_GENOMES`. Optimizer, scheduler and loss genomes live alongside it (`OPTIMIZER_GENOMES`, `OPTIMIZER_OPTIONAL_GENES`, `SCHEDULER_GENOMES`, `LOSS_GENOMES`). Parameters irrelevant to an architecture are discouraged in its manifest to prevent semantic bloat — **not enforced**: `audit_manifest` checks presence and non-`None` of required keys only, and does not reject extra keys.

---

## 2. Reproducibility Gates (The Fortress)

The following gates are implemented as three nested classes on `ReproducibilityGate` in `views_r2darts2/infrastructure/reproducibility_gate.py` — `Config`, `Temporal`, `Data`. There is no fourth `Hardware` class; the hardware invariants in §2.3 are enforced in the engines layer.

> **Wired vs unwired (verified 2026-09-10, register C-44):** on the production path only `audit_manifest`, `audit_architecture`, `audit_continuity`, `audit_prediction_horizon` and `lock_entropy` are invoked. `audit_boundary_integrity`, `audit_sequence_contiguity`, `audit_leakage`, `audit_frame_schema` and `audit_numerical_sanity` are implemented and tested but **called by nothing** in `views_r2darts2/`. Entries below marked ⚠ are unwired.

### 2.1 The Config Gate (`ReproducibilityGate.Config`)
*   **Audit Manifest**: Performs a dynamic, model-aware audit of the DNA.
*   **Audit Architecture**: Verifies that `len(steps) % output_chunk_length == 0` (ADR-009).
*   **Quadruple Catalog Firewall**: 
    - `ModelCatalog` audits at instantiation; `LossCatalog`, `OptimizerCatalog`, and `SchedulerCatalog` audit at first resolution (`get_loss` / `get_*_kwargs`).
    - **Refuse-to-Guess Invariant**: `ModelCatalog` (via `audit_manifest`) raises `MissingHyperparameterError`; the other three catalogs raise plain `ValueError` for a missing or `None` gene.
*   **Failure Mode**: `MissingHyperparameterError` or `ArchitectureMismatchError`.

### 2.2 The Temporal Gate (`ReproducibilityGate.Temporal`)
*   **The Continuity Guardian ($t+1$)**: `audit_continuity` verifies that the test set starts exactly one month after the training set ends.
*   ⚠ **The Boundary Firewall**: `audit_boundary_integrity` verifies that training series end *exactly* at the partition boundary — no leak past it, no starvation short of it. *Unwired on 0.2.x.*
*   ⚠ **The Sequence Auditor**: `audit_sequence_contiguity` scans training IDs to ensure a continuous range with **zero holes**. *Unwired on 0.2.x.*
*   **The Horizon Lockdown**: `audit_prediction_horizon` forbids forecasting past known ground truth for `calibration`/`validation` runs.
*   **The Horizon Siren** (in `Config.audit_architecture`): logs a high-visibility warning if `len(steps) != 36` or `steps[0] != 1`.

### 2.3 Hardware Invariants (enforced outside the gate)
There is no `HardwareAudit` class. These invariants live in the engines layer:
*   **Device Self-Healing**: `DartsForecaster._ensure_model_on_device()` (`views_r2darts2/engines/darts_forecaster.py`) audits the model device before every prediction and restores it if Darts drifted it to CPU (ADR-011). *On restoration failure the current code warns and continues — see register D-01.*
*   **Parallelism Lockdown**: `DartsForecastingModelManager._evaluate_model_artifact` forces `max_workers=1` unless `forecaster.device == "cpu"`.
*   **Device Resolution**: `get_device()` in `views_r2darts2/infrastructure/device.py`.

### 2.4 The Data Gate (`ReproducibilityGate.Data`)
*   ⚠ **Numerical Integrity**: `audit_numerical_sanity` detects `NaN`/`Inf`; *unwired* — the live NaN scan is in `DartsForecaster.predict`. `float32` is guaranteed by construction in `views_r2darts2/dataset/converters.py` (ADR-016).
*   ⚠ **Frame Schema**: `audit_frame_schema` validates a `views_frames.FeatureFrame` and raises on any non-`float32` dtype; *unwired*.
*   ⚠ **The Leakage Firewall**: `audit_leakage` — set-intersection check between train and test partitions; *unwired*.
*   **The Entropy Lock**: `lock_entropy(seed)` reseeds `random`, `numpy` and `torch` before every prediction (see register C-24 for the multi-worker caveat).

---

## 3. Implementation Patterns

### 3.1 Immutable Snapshots
All manager methods must capture a local snapshot of the configuration: `active_config = self.configs`. Internal logic must **never** mutate this dictionary. *One sanctioned mutation:* the artifact timestamp is injected via `_config_manager.add_config` and the snapshot re-taken (ADR-015).

### 3.2 Global Calibration
All target scalers must use `global_fit=True` to preserve cross-sectional signals and probabilistic calibration (ADR-012).

---

## 4. Team-Based Stress Testing (The Audit Suite)

| Team | Focus | Implementation |
| :--- | :--- | :--- |
| 🟩 **Green** | Resilience | Stochastic Parity and Scaling Integrity tests. |
| 🟫 **Beige** | Human Error | Polymorphic DNA audits and Mismatch detection. |
| 🟥 **Red** | Adversarial | Temporal Injection and Numerical Poisoning. |
