# Class Intent Contract: SchedulerCatalog

**Status:** Active  
**Owner:** Core Engineering  
**Last reviewed:** 2026-09-10  
**Related ADRs:** ADR-001, ADR-003, ADR-006, ADR-009, ADR-013  

---

## 1. Purpose

`SchedulerCatalog` is the fourth Genome Translator: it maps the DNA's `lr_scheduler_cls` and `lr_scheduler_*` keys to a concrete `torch.optim.lr_scheduler` class (or the package's own `WarmupCAWR`) and its validated kwargs.

> **Its primary goal is the same Genomic Firewall as its three siblings: no scheduler is instantiated with a guessed parameter.**

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** step the scheduler or manage the training loop.
- Does **not** implement schedulers — `WarmupCAWR` lives in `views_r2darts2/math/warmup_cawr.py`.

---

## 3. Responsibilities and Guarantees

- `get_scheduler_cls`: resolves `_CUSTOM_SCHEDULERS` (lazily loaded; currently `WarmupCAWR`) first, then `torch.optim.lr_scheduler`.
- `get_scheduler_kwargs`: for every key in `ReproducibilityGate.Config.SCHEDULER_GENOMES[name]`, reads the config value and renames it through `_KWARG_MAP` (e.g. `lr_scheduler_factor` → `factor`).
- Passes through extra keys from a nested `lr_scheduler_kwargs` block, **genome keys taking precedence**.
- Injects `_STATIC_KWARGS` last — `ReduceLROnPlateau` always gets `mode="min", monitor="val_loss"` — and these override config. This is intentional: they are Darts/Lightning integration requirements, not hyperparameters.

---

## 4. Inputs and Assumptions

- A config with `lr_scheduler_cls` and the genome keys for that scheduler.

---

## 5. Outputs and Side Effects

- A class and a kwargs dict. Stateless; no side effects.

---

## 6. Failure Modes and Loudness

- `ValueError` — scheduler name not a `torch.optim.lr_scheduler` attribute; not in `SCHEDULER_GENOMES`; any genome key missing or `None`.

---

## 7. Boundaries and Interactions

- **Physical Zen:** `views_r2darts2/catalogs/scheduler_catalog.py`.
- Registry duplication with the gate: `_KWARG_MAP` keys must match `SCHEDULER_GENOMES` (C-11).
- Exported from the package root `__all__` (lazy).

---

## 8. Examples of Correct Usage

```python
sc = SchedulerCatalog(config=dna)
cls, kwargs = sc.get_scheduler_cls(), sc.get_scheduler_kwargs()
```

---

## 9. Examples of Incorrect Usage

- Putting `monitor` in the DNA and expecting it to override the static `val_loss`.
- Adding a scheduler to `_KWARG_MAP` without adding its genome to the gate.

---

## 10. Test Alignment

- **None.** No test under `tests/` references `scheduler_catalog` or `WarmupCAWR` (register C-30). The kwarg remapping, pass-through precedence, and static-kwargs override are unverified.

---

## End of Contract

This document defines the **intended meaning** of `SchedulerCatalog`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
