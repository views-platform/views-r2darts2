# docs/archive — superseded governance artifacts

**Status:** historical. Every file here describes the 0.1.x codebase (pre-July-2026) and is kept
so the reasoning of that period stays recoverable. Paths, class names and test names inside them
may no longer exist; `docs/validate_docs.sh` deliberately does not check this directory.

| File | What it was | Superseded by |
|---|---|---|
| `ADR_COMPLIANCE_REPORT.md` | Feb-2026 compliance certification of the 0.1.x code against ADRs 000–013 | the register's D-entries and compliance notes in each ADR |
| `CIC_views_dataset_darts.md` | contract for `_ViewsDatasetDarts`, deleted in the 0.2.x rewrite | `docs/CICs/views_dataset.md` |
| `FORTRESS_ROADMAP.md` | the Feb-2026 hardening plan (status COMPLETED / SUPERSEDED) | `reports/post_mortems/2026-02-12_fortress_hardening.md` (the same initiative, concluded) |

Placement rule, stated once: governance documents (ADRs, CICs, standards) that are superseded go
here; reports, memos and plans go to `reports/archived/`. The full pre-rewrite tree is at tag
`archive/survey_risk-0.1.x`.
