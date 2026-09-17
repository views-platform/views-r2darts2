# Loss Function Integrity Review Plan

**Date:** 2026-02-13  
**Status:** DRAFT  
**Owner:** Simon Polichinel von der Maase  
**Target:** `views_r2darts2/utils/loss.py`

---

## 1. Executive Summary
The current loss function implementations in `views-r2darts2` were developed under experimental conditions ("fail fast"). While functional, they contain hardcoded magic numbers, inconsistent type safety, and "best-effort" error handling that violates the repository's "Fortress" architecture (specifically ADR-008 and ADR-010).

This plan outlines the strategy to elevate these components to **Critical Infrastructure Standards**. The goal is to ensure that every loss function is mathematically proven, numerically stable, type-safe, and fully compliant with the "Fail Loud" philosophy.

---

## 2. Objectives
1.  **Mathematical Correctness:** Verify that code implements the exact mathematical formulas defined in literature/docs.
2.  **Numerical Stability:** Eliminate "silent" NaN masking. Ensure robust gradient flow even in extreme regimes.
3.  **ADR Compliance:** Enforce `float32` precision (ADR-010) and explicit failure on invalid states (ADR-008).
4.  **Configuration Integrity:** Remove all magic numbers (e.g., `1e-4`, `10`) and expose them as configurable parameters.
5.  **Type Safety:** Achieve 100% type hint coverage for all forward passes.

---

## 3. Scope
The review covers the following 7 classes in `views_r2darts2/utils/loss.py`:
1.  `WeightedHuberLoss`
2.  `TimeAwareWeightedHuberLoss`
3.  `SpikeFocalLoss`
4.  `WeightedPenaltyHuberLoss` (Critical: Primary conflict loss)
5.  `TweedieLoss` (Critical: Primary continuous loss)
6.  `AsymmetricQuantileLoss`
7.  `ZeroInflatedLoss`

---

## 4. Methodology & Audit Rubric

For each loss function, we will perform the following checks:

### A. Static Analysis
- [ ] **Type Hints:** Are `preds` and `targets` typed as `torch.Tensor`?
- [ ] **Docstrings:** Do they contain LaTeX formulas? Do they explain *why* this loss is used?
- [ ] **Magic Numbers:** Are there any hardcoded literals (e.g., `0.5`, `10`, `1e-4`) buried in the logic?

### B. Mathematical Audit
- [ ] **Formula Verification:** Compare code against the `docs/specs/loss_function_tuning_guide.md` and standard literature.
- [ ] **Reduction:** Is `reduction='mean'` explicitly handled and correct?
- [ ] **Broadcasting:** Does the loss handle `(Batch, Time, 1)` vs `(Batch, Time)` correctly?

### C. Fortress Hardening (ADR Compliance)
- [ ] **Precision:** Does the code explicitly check `inputs.dtype == torch.float32`?
- [ ] **Fail-Loud:** Does it raise an exception on NaN/Inf instead of silently masking them (unless masking is mathematically justified and logged)?
- [ ] **Device Safety:** Are new tensors created on the correct device (`preds.device`)?

---

## 5. Identified Issues & Action Items

### 5.1 Global Issues
- **File Structure:** All losses are in a single 700-line file (`utils/loss.py`).
    - *Action:* Split into `views_r2darts2/model/loss/` package with one file per class.
- **Type Hints:** Missing standard Python type hints.
    - *Action:* Add full typing support.

### 5.2 Specific Issues

#### `WeightedPenaltyHuberLoss`
- **Contradiction:** Comments say "Replace with large penalty" to fix instability, but code does `torch.zeros_like`.
- **Action:** Decide on a strategy (Crash or Penalty) and implement it consistently. If Crashing, remove the masking.

#### `TweedieLoss`
- **Softplus Offset:** Uses `self.eps` to avoid log(0). Verify if `1e-8` is sufficient for `float32`.
- **Power Derivation:** Verify the `1-p` and `2-p` terms against the Tweedie deviance formula.

#### `TimeAwareWeightedHuberLoss`
- **Magic Number:** Hardcoded `1e-4` for zero/non-zero classification.
- **Action:** Parameterize this as `zero_threshold` in `__init__`.

#### `ZeroInflatedLoss`
- **Magic Number:** Hardcoded scaling factor `10` in `torch.sigmoid(-preds_flat * 10)`.
- **Action:** Parameterize as `temperature` or `scale_factor`.
- **Numerical Risk:** `exp` in Pseudo-Huber can overflow. Verify stability range.

---

## 6. Execution Phases

### Phase 1: Refactoring & Isolation (Day 1)
- Move `utils/loss.py` to `model/loss/`.
- Split into individual files.
- Update `LossSelector` to import from new locations.

### Phase 2: Code Hardening (Day 1-2)
- Apply `float32` guards.
- Remove magic numbers.
- Fix the `WeightedPenaltyHuberLoss` logic.

### Phase 3: Verification (Day 3)
- Write "Gold Standard" tests for each loss using `torch.autograd.gradcheck`.
- Verify gradients are non-zero and finite.

---

## 7. Deliverables
1.  Refactored codebase in `views_r2darts2/model/loss/`.
2.  Updated `ADR-008` compliance report.
3.  New test suite `tests/model/loss/test_gradients.py`.

---

**Approval:**
Pending User Review.
