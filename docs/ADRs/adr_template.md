
# ADR-XXXX: <Concise decision title>

**Status:** Proposed | Accepted | Superseded | Deprecated  
**Date:** YYYY-MM-DD  
**Deciders:** <Names / roles>  
**Consulted:** <Optional>  
**Informed:** <Optional>  

---

## Context

Describe the problem that motivated this decision.

Include:
- What is *not working* or *no longer tenable*
- Relevant technical, organizational, or scientific constraints
- Prior assumptions that turned out to be wrong
- Why this decision matters *now* (and not later)

This section should make it obvious to a future reader **why a decision was needed at all**.

---

## Decision

State the decision **clearly and unambiguously**.

- What is being decided?
- What is explicitly *in scope*?
- What is explicitly *out of scope*?

Use assertive language.  
This is the **source of truth**.

---

## Rationale

Explain *why this option was chosen* over alternatives.

Include:
- Key design principles or values (e.g. correctness > convenience)
- Trade-offs consciously accepted
- Alignment with long-term architecture or research goals
- Why this decision reduces risk, ambiguity, or technical debt

This is where future disagreements get defused.

---

## Considered Alternatives

List the main alternatives that were seriously considered.

For each alternative:
- Brief description
- Why it was *not* chosen
- Any conditions under which it might be revisited

Example format:

### Alternative A: <name>
- **Pros:**  
- **Cons:**  
- **Reason for rejection:**  

---

## Consequences

Describe the consequences of this decision.

### Positive
- Benefits unlocked
- Simplifications introduced
- Risks reduced

### Negative
- New constraints imposed
- Short-term pain
- Technical debt explicitly accepted

Be honest. This section builds trust.

---

## Implementation Notes

Concrete guidance for implementation.

Include:
- Where the decision should be enforced (code, config, docs, tests)
- Migration strategy (if applicable)
- Required follow-up tasks or refactors
- Guardrails to prevent regression

If nothing is required yet, say so explicitly.

---

## Validation & Monitoring

How will we know this decision was correct?

Examples:
- Tests or invariants that should hold
- Metrics or signals to watch
- Failure modes that would trigger reconsideration

This turns the ADR into a *living* artifact.

---

## Open Questions

List unresolved questions or known unknowns.

- What do we still not know?
- What depends on future work or data?
- What should be revisited later?

---

## References

Links to:
- PRs
- Issues
- Design docs
- Papers
- Slack threads / meeting notes

Future readers should be able to reconstruct the full story.
