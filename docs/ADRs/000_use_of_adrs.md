# ADR-000: Use of Architecture Decision Records (ADRs)

**Status:** Accepted  
**Date:** 2026-02-11  
**Deciders:** Simon Polichinel von der Maase  
**Informed:** All contributors  

---

## Context

The `views-r2darts2` repository supports long-lived research development in conflict forecasting under high uncertainty. It involves multiple contributors, evolving deep learning architectures, and strict operational constraints regarding temporal integrity and reproducibility.

Many important decisions in this system (e.g., how we handle zero-inflation in loss functions, how we enforce temporal boundaries, or how we manage scalers for probabilistic forecasts) are:
- Made under research uncertainty
- Revised as new models or data become available
- Obvious to those present during a "sprint," but opaque months later
- Revisited implicitly, often leading to regressions in reproducibility or duplicated debate

Without a shared record of *why* decisions were made, the project risks re-litigating settled questions, accidentally reversing critical design choices (like the "Fortress" firewall gates), and losing institutional memory as research focus shifts.

We therefore need a lightweight but rigorous mechanism to document **significant decisions**, their **rationale**, and their **consequences**.

---

## Decision

We will use **Architecture Decision Records (ADRs)** to document significant technical, architectural, and conceptual decisions in this project.

ADRs are:
- Written in Markdown
- Stored in the repository under `docs/ADRs/`
- Numbered sequentially
- Treated as first-class project artifacts, subject to code review

An ADR records **a decision**, not a discussion or a design proposal.

---

## What Is an ADR?

An ADR is a short, structured document that captures:
- The context in which a decision was made
- The decision itself (the "new law")
- The rationale behind it
- The alternatives considered
- The consequences (positive and negative)

An ADR answers the question: *“Why is the system the way it is?”* (e.g., "Why do we force sequential prediction on GPUs?").

---

## When to Write an ADR

Write an ADR when making a decision that:
- Affects system architecture or data layout (e.g., introducing a new data handler layer)
- Constrains future design choices (e.g., standardizing on `layer_widths`)
- Changes assumptions or invariants (e.g., temporal continuity requirements)
- Introduces or accepts technical debt for research speed
- Has non-obvious trade-offs

Examples include:
- Data representations or schemas
- Model interfaces or abstractions (e.g., `ModelCatalog` vs manual instantiation)
- Evaluation or validation strategies (e.g., standardizing on 36-month horizons)
- Handling of uncertainty, scaling, or transformations

Do **not** write ADRs for:
- Routine refactors
- Purely local implementation details
- Temporary experiments (unless they become part of the "production" pipeline)

---

## Lifecycle of an ADR

- **Proposed** — decision under consideration
- **Accepted** — decision is active and authoritative
- **Superseded** — replaced by a newer ADR (e.g., a better loss function strategy)
- **Deprecated** — decision remains but should no longer be used

Decisions are never deleted. If a decision changes, it is **superseded**, not erased.

---

## Relationship to Code

ADRs and code must agree.
- Code should implement the decision described in the ADR
- Significant deviations require a new ADR or an update
- ADRs must be referenced from code comments when implementing non-trivial logic

If code and ADRs disagree, the ADR is the source of truth until the code is fixed or the ADR is superseded.

---

## Consequences

### Positive
- Clearer decision-making under research pressure
- Fewer repeated debates on architecture
- Easier onboarding for new researchers/engineers
- Better long-term coherence of the "Fortress" infrastructure

### Negative
- Small upfront cost in writing
- Requires discipline to maintain during fast-paced research cycles

These costs are accepted intentionally.

---

## References

- `docs/ADRs/adr_template.md`
- `docs/standards/REPRODUCIBILITY_MANIFEST.md`
