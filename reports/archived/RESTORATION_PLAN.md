# Restoration Plan: Getting Back to Peak State

**Date:** 2026-02-11  
**Objective:** Manually re-implement and verify the lost architectural improvements to achieve ADR compliance and "Fortress" integrity.

---

## Phase 1: Re-Implement The Polymorphic Gate (CRITICAL)
1.  **Rebuild `ReproducibilityGate`:** Re-define `CORE_GENOME` and the `ALGORITHM_GENOMES` mapping in `views_r2darts2/utils/gates.py`.
2.  **Restore Dynamic `audit_manifest`:** Update the validation logic to switch based on the `algorithm` key.
3.  **Verification:** Run `tests/test_reproducibility_infra.py` (it will fail until Phase 2 is complete).

## Phase 2: Re-Implement Parameter Propagation (MODEL CATALOG)
1.  **Resolve `optimizer_cls`:** Restore the `_get_optimizer_cls()` helper and pass it to all model constructors.
2.  **Safe Config Access:** Standardize all 10 model instantiation methods to use `.get()` for every parameter.
3.  **Correct `use_static_covariates`:** Remove the flag from models that do not support it (N-BEATS, NHiTS, TCN).
4.  **Verification:** Run `tests/test_catalog.py`.

## Phase 3: Hardware & Boundary Hardening
1.  **Confirm Device Restoration:** Verify `DartsForecaster.predict` still contains the self-healing device restoration logic (it appears this part *was* preserved in `HEAD`).
2.  **Verify Data Handshake:** Ensure `_ViewsDatasetDarts` still calls `ReproducibilityGate.Data.audit_dataframe_schema`.

## Phase 4: Final Validation
1.  **Run Full Test Suite:** Execute `conda run -n views_pipeline pytest`.
2.  **Check ADR Consistency:** Verify that the code matches the "Constitutional" refinements made to ADRs 001, 003, and 009.

---

## Execution Guardrails
- **No Force Pushing:** Do not force-push until the user has explicitly verified the state.
- **Incremental Verification:** Test every change immediately after implementation.
- **Respect Commit Logic:** Each phase will be its own commit to avoid future shadowing.
