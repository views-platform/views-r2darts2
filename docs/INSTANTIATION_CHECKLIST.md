# Instantiation Checklist

Adapted from `base_docs/INSTANTIATION_CHECKLIST.md` for **views-r2darts2**, a
brownfield project that adopted the governance framework via `adopt-base-docs`.
Items are checked off to reflect the documentation set as it actually exists.
Run `bash docs/validate_docs.sh` (from anywhere) to re-verify internal consistency — since
2026-09-10 it also checks that every code/test path the live docs name exists in the repo.

**Re-derived 2026-09-10 against the 0.2.x codebase** (`development` @ `fe7e681`).

---

## Before You Start

- [x] Adoption phase: full constitutional set (ADR-000 … ADR-009) plus
      project-specific ADRs (ADR-010 … ADR-016; ADR-010 superseded by ADR-016) — see `ADRs/README.md`
- [x] Ontological categories identified — see `ADRs/001_ontology_of_the_repository.md`
      (datasets, forecasters/model managers, catalogs, transformers/scalers,
      callbacks, configs, artifacts)

---

## ADR Adaptation

### All adopted ADRs
- [x] Status set to `Accepted` (constitutional 000–009 and project 011–016); ADR-010 `Superseded`; ADR-006 `Accepted — compliance open` (D-02)
- [x] Date / Deciders / Consulted / Informed fields filled

### Per-ADR adaptation notes
- [x] **ADR-000:** `ADRs/` path reference matches `docs/ADRs/`
- [x] **ADR-001:** Ontological categories and stability levels defined for this repo
- [x] **ADR-002:** Layering and forbidden dependency patterns defined
- [x] **ADR-003:** Forbidden-behavior examples adapted to the forecasting domain
- [x] **ADR-005:** Testing taxonomy adopted (universal)
- [x] **ADR-006:** Intent-contract criteria adopted (universal)
- [x] **ADR-007:** Contributor-protocol paths verified against `contributor_protocols/`
- [x] **ADR-009:** Boundary/configuration-validation examples adapted to this repo

### Project-specific ADRs (beyond the constitutional set)
- [x] **ADR-010:** Numerical precision and semantic thresholds — *superseded by ADR-016*
- [x] **ADR-011:** Hardware integrity and parallelism
- [x] **ADR-012:** Scaling pipeline and calibration integrity
- [x] **ADR-013:** Physical symmetrical architecture
- [x] **ADR-014:** Technical risk register (governs `reports/technical_risk_register.md`)
- [x] **ADR-015:** Artifact-prediction timestamp contract
      (satellite of views-pipeline-core ADR-052; cross-repo refs are intentional)
- [x] **ADR-016:** Numerical precision, raw output, and the clipping question (D-05 open)

---

## CICs

- [x] `CICs/README.md` active-contract list reflects this project's classes
- [x] Intent contracts written for the non-trivial classes (21 active):
      `darts_forecaster`, `darts_forecasting_model_manager`, `model_catalog`, `loss_catalog`,
      `optimizer_catalog`, `scheduler_catalog`, `views_dataset`, `dataset_builder`, `zarr_store`,
      `dataset_converters`, `dataset_subclasses`, `feature_scaler_manager`, `scaler_selector`,
      `inverse`, `darts_bridge`, `frame_builder`, `static_covariates`, `reproducibility_gate`,
      `fortress_monitoring_callbacks`; `views_dataset_darts` retired to `archive/`
- [ ] In-code `Intent Contract:` docstrings on the ten classes named in register C-39 (D-02)

---

## Contributor Protocols

- [x] `contributor_protocols/silicon_based_agents.md` adapted for this tooling
- [x] `contributor_protocols/carbon_based_agents.md` adapted for the team
- [x] Hardened protocol present and domain-adapted as
      `contributor_protocols/fortress_protocol.md`

---

## Standards

- [x] `standards/logging_and_observability_standard.md` scoped to this domain
- [x] Reproducibility expectations captured in `standards/REPRODUCIBILITY_MANIFEST.md`
- [N/A] The optional physical-architecture standard is not used as a separate
      file — its concerns are covered by `ADRs/013_physical_symmetrical_architecture.md`

---

## Loss Cards

- [x] `loss_cards/README.md` indexes every card and names the modules without one
- [ ] A card for every registered loss — 8 of 20 loss modules have one (register C-41);
      the production `SpotlightLoss` family has none
- [x] No card advertises a constructor default the code does not have (ADR-003)

---

## Final Verification

- [x] No files carry Status `--template--` except where intentionally deferred
- [ ] No phantom references to non-existent local files — *re-verified at the end of the
      `governance-0.2.x` branch by `validate_docs.sh` pass 7; ticked in Stage 6*
      (cross-repo ADR refs — views-baseline ADR-016, views-hydranet ADR-026,
      views-pipeline-core ADR-052 — are external satellites, not local gaps)
- [x] All local cross-ADR references resolve correctly
- [x] `reports/technical_risk_register.md` present and governed by ADR-014
- [x] `docs/validate_docs.sh` present and passing
- [x] Compliance report archived — `archive/ADR_COMPLIANCE_REPORT.md` is marked HISTORICAL
      (it certified the 0.1.x codebase; nothing in it is asserted of 0.2.x). Current
      compliance state is the register: `reports/technical_risk_register.md` and D-01..D-05.
