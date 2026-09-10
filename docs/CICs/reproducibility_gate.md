# Class Intent Contract: ReproducibilityGate

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-09-10  
**Related ADRs:** ADR-001, ADR-003, ADR-005, ADR-008, ADR-009, ADR-016  

---

## 1. Purpose

The `ReproducibilityGate` is "The Law" of the repository. It is a stateless utility class that centralizes all validation logic required to ensure that an experiment is scientifically sound, temporally contiguous, and 100% reproducible.

> **Its primary goal is to prevent "Silent Lies" by halting execution immediately upon the detection of any architectural or temporal violation.**

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** perform model training or inference.
- This class does **not** modify data or configurations (it is strictly read-only).
- This class does **not** manage file paths or artifacts.
- This class does **not** implement metric calculations.

---

## 3. Responsibilities and Guarantees

- **Enforces Genomic Integrity:** Guarantees that every experiment context contains all mandatory hyperparameters for its chosen algorithm (ADR-009).
- **Guarantees Temporal Continuity:** Verifies the $t+1$ invariant, ensuring that the test set starts exactly one month after the training set ends.
- **Prevents Data Leakage:** Ensures that no month IDs from the test partition are physically present in the training tensors.
- **Enforces Sequential Integrity:** Scans for "Temporal Holes" (missing months) in historical data to prevent distorted time-series dynamics.
- **Numerical Sanity Firewall:** `audit_numerical_sanity` detects NaNs, Infs, and extreme adversarial outliers — **but is not invoked on any production path on 0.2.x** (C-44).
- **Architecture Alignment:** `Config.audit_architecture` requires `len(steps) % output_chunk_length == 0` and warns loudly on a non-36 horizon or a non-1 start offset.
- **Boundary Integrity:** `Temporal.audit_boundary_integrity` requires training series to end exactly at the partition boundary — no leak, no starvation.
- **Horizon Lockdown:** `Temporal.audit_prediction_horizon` forbids forecasting past known ground truth for `calibration`/`validation` runs.
- **Frame Schema:** `Data.audit_frame_schema` validates a `views_frames.FeatureFrame` and raises on any non-`float32` dtype — **but has no production caller** (C-44); the pandas `audit_dataframe_schema` of 0.1.x is gone.
- **Wired vs unwired (2026-09-10):** on the production path only `Config.audit_manifest`, `Config.audit_architecture`, `Temporal.audit_continuity`, `Temporal.audit_prediction_horizon` and `Data.lock_entropy` are called. `audit_boundary_integrity`, `audit_sequence_contiguity`, `audit_leakage`, `audit_frame_schema` and `audit_numerical_sanity` exist, are tested, and are called by nothing in `views_r2darts2/`.
- **Entropy Lock:** `Data.lock_entropy(seed)` reseeds `random`, `numpy` and `torch` so probabilistic draws are bit-identical across reloads.

---

## 4. Inputs and Assumptions

- **Config Snapshot:** Assumes a raw dictionary of hyperparameters.
- **Temporal Lists:** Assumes lists of integer month IDs representing time series indices.
- **Darts Series:** Assumes a list of `darts.TimeSeries` objects for boundary auditing.

---

## 5. Outputs and Side Effects

- **Assertions:** Produces no data; its only output is either "Pass" (continued execution) or "Fail" (exception raised).
- **Logging:** Emits `ERROR` or `CRITICAL` logs describing the exact nature of an invariant violation before raising.
- **Siren Alerts:** Emits high-visibility warnings for non-standard but valid states (e.g., non-36 month horizons).

---

## 6. Failure Modes and Loudness

- **Fail-Loud Mandate:** This class must **never** swallow an error. All violations must raise a subclass of `ReproducibilityError`. *Exception:* `audit_frame_schema` raises bare `KeyError` for missing columns (`reproducibility_gate.py:666`, `:672`).
- **Immediate Termination:** Fails at the earliest possible moment (usually during the Handshake phase).
- **Explicit Rationale:** Every raised exception must include a descriptive message explaining *why* the contract was violated.

---

## 7. Boundaries and Interactions

- **Universal Utility:** Accessible and consumed by all layers (Data, Model, and Manager).
- **Independent:** Stateless. Imports only `infrastructure/exceptions.py` from this package (ADR-002 Layer 0) — but it is *not* dependency-free: it imports `darts.TimeSeries`, `torch`, and `views_frames.FeatureFrame`.
- **Physical Zen:** Lives in `views_r2darts2/infrastructure/reproducibility_gate.py`.
- **Registries:** Owns `CORE_GENOME`, `ALGORITHM_GENOMES`, `OPTIMIZER_GENOMES`, `SCHEDULER_GENOMES`, `LOSS_GENOMES` on `Config`. The four catalogs hold parallel registries; the two must be edited together (C-11).
- **Master Auditor:** Validates the handshake between `views_pipeline_core` and the local codebase.

---

## 8. Examples of Correct Usage

```python
# Validating a configuration
ReproducibilityGate.Config.audit_manifest(config_dict)

# Validating temporal continuity
ReproducibilityGate.Temporal.audit_continuity(partition_dict)
```

---

## 9. Examples of Incorrect Usage

- **Conditional Auditing:** Only calling the gate if a certain flag is set (violates the "Fortress" mandate).
- **Result Catching:** Using `try...except` to catch a gate failure and continue execution.
- **Data Modification:** Attempting to use a gate method to "fill" missing data holes.

---

## 10. Test Alignment

- **Red + Beige Team:** `tests/test_reproducibility_gate.py` (33 tests: hole and leak injection, DNA omission, `None`-value rejection, horizon overflow, entropy lock).

---

## 11. Evolution Notes

### Known Deviations / Technical Debt
- **`MultiQueryTransformerModel` is registered in `ALGORITHM_GENOMES` with no catalog factory** — `audit_manifest` accepts it and `ModelCatalog.get_model` then crashes opaquely (C-11).
- **`float64` check is fail-loud** in `audit_frame_schema` (`:657-661` raises `NumericalSanityError`), unreachable in practice because the converters emit only `float32`, and never invoked in production regardless (C-44).

---

## End of Contract

This document defines the **intended meaning** of `ReproducibilityGate`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
