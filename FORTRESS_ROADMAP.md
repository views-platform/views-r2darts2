# Fortress Roadmap: Strengthening the Reproducibility Infrastructure

This document outlines the high-priority tasks remaining to finalize the **Fortress Architecture** and achieve absolute scientific integrity in the forecasting pipeline.

## 1. Entropy Locking (The Reloading Gap)
**Goal:** Guarantee bit-perfect identity in probabilistic distributions across save/load cycles.

*   **The Problem:** While we verify that model weights are identical after reloading, the global RNG state (Torch, Numpy, Python) continues to evolve. Two identical models will produce different MC Dropout distributions if prediction starts at different points in the entropy stream.
*   **The Solution:** Implement `ReproducibilityGate.Data.lock_entropy(seed)` to be called inside `forecaster.predict()`.
*   **Action:** Force-reset `torch.manual_seed`, `np.random.seed`, and `cuda.manual_seed` immediately before the first sample is generated.

## 2. The Automated Scribe (Beige Team Defense)
**Goal:** Eliminate mundane human transcription errors during hyperparameter transfer.

*   **The Problem:** Users must manually copy "best" hyperparameters from WandB into `config_hyperparameters.py`. A single typo (e.g., `lr: 0.003` vs `0.0003`) causes a silent, deadly deviation.
*   **The Solution:** Create a synchronization utility: `python -m views_r2darts2.utils.sync_run <wandb_run_id>`.
*   **Action:**
    1.  Download the run configuration via the WandB API.
    2.  Validate it against the **Mandatory DNA Manifest**.
    3.  Automatically overwrite the local `config_hyperparameters.py` with the audited values.

## 3. The Executioner's Handbook (Standardization)
**Goal:** Formalize the new "Loud Failure" standard for all contributors.

*   **The Problem:** New or external developers may see the strict "Kill Gates" as bugs rather than features.
*   **The Solution:** Update `README.md` or create `docs/FORTRESS_PROTOCOL.md`.
*   **Action:**
    1.  Define the **"No Magic Defaults"** rule.
    2.  Explain the **Firewall Gates** (Starvation, Peeking, Continuity).
    3.  Document the **Reproduction Certificate** produced at the start of every run.
    4.  Clearly state: *“In this repository, a crash is a successful defense of scientific integrity.”*
