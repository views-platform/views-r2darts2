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

**The 0.1.x archive:** the pre-rewrite governance layer — the original `docs/` and `reports/`, and a
`TEMP/docs/` directory of 28 February-2026 draft templates — is preserved at tag
`archive/survey_risk-0.1.x` (commit `d847a34`). The `survey_risk` branch that carried it was retired
on 2026-09-10 after PR #38 re-derived the layer against 0.2.x. The three drafts that existed nowhere
else (the ADR and CIC compliance-audit templates and the specification-card template) were moved
out of the repository into the platform's shared `base_docs` template set; the rest were superseded.

**One live pointer to be aware of:** `sweep_configs/experimental_sweep_configs/lr_finder_sweep.py:7`
declares `"program": "simple_training_run.py"` as a W&B sweep entrypoint. The only file of that
name in the repository is `reports/archived/simple_training_run.py`, which is not on any import
path at runtime. That sweep config is out of scope for the documentation pass and is recorded
in the register (C-43).

The `__init__.py` here makes this directory importable as a package. It should not be; that is
a leftover and is noted, not fixed, in this pass.
