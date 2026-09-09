# ADR-014: Technical Risk Register as First-Class Governance Artifact

**Status:** Accepted
**Date:** 2026-04-11
**Deciders:** Simon Polichinel von der Maase

---

## Context

The Fortress governance stack (ADRs 000–013, CICs, contributor protocols) codifies *how* code should be written, but provides no durable ledger of *known risks that have not yet been fixed*. When an audit (expert-code-review, review-diff, test-review, falsify, tech-debt-cleanup) identifies a concern that is real but not blocking, that concern currently has nowhere to live except the conversation transcript and commit messages — both of which are write-only from the perspective of future sessions.

Without a register, the same risks get rediscovered every few weeks, tier assignments drift, and deduplication depends on the reviewer's memory. This directly contradicts ADR-007 (silicon-based agents as untrusted contributors): a stateless agent cannot remember prior findings, so the codebase must remember for it.

---

## Decision

This repository adopts a **Technical Risk Register** as a first-class governance artifact, stored at `reports/technical_risk_register.md`.

### 1. The Register is the Single Source of Truth
Every known but unaddressed concern with structural, maintainability, or correctness implications MUST be recorded in the register. Audits that identify new risks MUST funnel through `register-risk` — no ad-hoc concern lists in scratch files, PR bodies, or post-mortems.

### 2. Tiered by Causal Severity
Concerns are assigned one of four tiers:

| Tier | Criterion |
|------|-----------|
| 1 | Silent data or model-output corruption with no error signal |
| 2 | Structural fragility with a concrete, realistic trigger |
| 3 | Coupling / maintainability cost affecting multiple contributors |
| 4 | Code quality observation with no correctness or reliability impact |

Tier 1 and Tier 2 require explicit evidence (prior incident, reproducible trigger). Tier 3/4 are discretionary.

### 3. Triggers Must Be Actionable
Every concern MUST specify a concrete future developer action that would make the concern acute. "Any change to this module" and "performance degradation" are not acceptable triggers. The trigger answers: *"what should a future reviewer check before doing X?"*

### 4. Deduplication is Mandatory
No concern may be registered without first checking every existing entry for overlap. Overlapping findings merge into existing entries rather than creating new ones. IDs are permanent — gaps indicate merged or resolved entries.

### 5. Resolution is Explicit
Concerns move from `Open Concerns` to `Resolved Concerns` only with an explicit resolution note and date. A concern is never silently deleted.

---

## Consequences

**Positive:**
- Audit findings become durable state rather than conversation ephemera.
- Deduplication gate prevents the register from becoming a grab-bag.
- Tier discipline forces reviewers to distinguish structural risk from code-quality nitpicks.
- Future stateless agents can load the register at session start and avoid re-flagging known concerns.

**Negative:**
- Requires discipline to keep tier assignments calibrated — drift is a real failure mode. The `review-rr` skill exists to periodically recalibrate.
- Adds a mandatory step to every audit workflow.

**Neutral:**
- The register file lives under `reports/` rather than `docs/` because it is operational state, not architectural intent. ADRs codify *what should be true*; the register records *what is currently wrong*.

---

## Alternatives Considered

1. **GitHub Issues.** Rejected: issues are ephemeral across repo migrations and are invisible to local audit agents without API calls. The register lives in the repo so it is present in every clone and every Claude session.
2. **Inline `# TODO(risk):` comments.** Rejected: tags decay, cannot be tiered, cannot be deduplicated, and cannot track resolution. Good for one-off reminders; wrong tool for a risk ledger.
3. **A separate `docs/risks/` directory with one file per concern.** Rejected: adds filesystem overhead, fragments the deduplication check, and hides the total count.

---

## Related Artifacts

- `reports/technical_risk_register.md` — the register itself
- `~/.claude/skills/register-risk/` — the intake skill
- `~/.claude/skills/review-rr/` — the recalibration skill
- ADR-007 (Silicon-based agents as untrusted contributors) — the "stateless agents need durable memory" premise
- ADR-003 (Authority of declarations over inference) — the register is a declaration, not an inference
