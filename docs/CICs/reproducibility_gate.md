# Class Intent Contract: ReproducibilityGate

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-02-11  
**Related ADRs:** ADR-001, ADR-003, ADR-005, ADR-008, ADR-009, ADR-010  

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
- **Numerical Sanity Firewall:** Detects NaNs, Infs, and extreme adversarial outliers at the data entry points.

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

- **Fail-Loud Mandate:** This class must **never** swallow an error. All violations must raise a subclass of `ReproducibilityError`.
- **Immediate Termination:** Fails at the earliest possible moment (usually during the Handshake phase).
- **Explicit Rationale:** Every raised exception must include a descriptive message explaining *why* the contract was violated.

---

## 7. Boundaries and Interactions

- **Universal Utility:** Accessible and consumed by all layers (Data, Model, and Manager).
- **Independent:** Must remain stateless and dependency-free (ADR-002 Layer 0).
- **Physical Zen:** Lives in `views_r2darts2/utils/reproducibility_gate.py`.
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

- **Red Team:** `tests/test_reproducibility_infra.py` (Adversarial injection of holes and leaks).
- **Beige Team:** `tests/test_reproducibility_infra.py` (DNA manifest omission tests).
- **Infrastructure:** `tests/repro_phase1_gate.py` (Verification of polymorphic genome auditing).

---

## 11. Evolution Notes

### Known Deviations / Technical Debt
- **Precision Auditing:** Currently, the gate does not strictly check for `float32` vs `float64` at the tensor level. This should be added to the Data gate to enforce ADR-010.

---

## End of Contract

This document defines the **intended meaning** of `ReproducibilityGate`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
