# Archived reports (Jan–Feb 2026)

Thirty files — 25 Markdown, 5 Python — from the investigations that preceded and accompanied
the Feb-2026 "Fortress" refactor of the 0.1.x codebase: Tweedie loss memos and diagnostics,
evaluation-library implementation reports, reproducibility audits, restoration and hardening
plans, context handovers. They describe code, paths, and test files that in most cases no
longer exist on the 0.2.x line.

**Status:** historical record. Nothing outside `reports/` references any file here; the nine
inbound links that exist are between siblings in this directory and `reports/post_mortems/`.
The register does not cite them. They are kept so the reasoning behind decisions made then
remains recoverable, not because they describe the present.

**Not carried onto this branch:** `survey_risk` @ `d847a34` also holds a `TEMP/docs/` directory — five
February-2026 draft templates (ADR/CIC compliance-audit templates, a hardened-protocol template,
a specification-card template, and `physical_architecture_standard.md`) that exist nowhere else.
They were never adopted and are not restored here; `survey_risk` remains their home.

**One live pointer to be aware of:** `sweep_configs/experimental_sweep_configs/lr_finder_sweep.py:7`
declares `"program": "simple_training_run.py"` as a W&B sweep entrypoint. The only file of that
name in the repository is `reports/archived/simple_training_run.py`, which is not on any import
path at runtime. That sweep config is out of scope for the documentation pass and is recorded
in the register (C-43).

The `__init__.py` here makes this directory importable as a package. It should not be; that is
a leftover and is noted, not fixed, in this pass.
