
# ADR-002: Topology and Dependency Rules

**Status:** --template--
**Date:** YYYY-MM-DD  
**Deciders:** <roles / team>  

---

## Context

Once the ontology of the repository is defined, uncontrolled dependencies between concepts can still lead to architectural decay.

In particular, ML and data-heavy systems tend to accumulate:
- circular dependencies
- “helpful” cross-layer access
- post-hoc reconstruction of intent

To prevent this, explicit rules are required governing **allowed and forbidden dependencies**.

---

## Decision

This repository enforces **explicit dependency rules** between ontological categories.

These rules define:
- which categories may depend on which others
- which dependency directions are forbidden
- where semantic boundaries must not be crossed

The goal is not to prescribe every allowed interaction, but to **forbid known failure modes**.

---

## Dependency Principles

The following principles apply globally:

- Dependencies must flow in a single, intentional direction.
- No component may infer semantics from another component’s internal implementation.
- Higher-level concerns may depend on lower-level concerns, but not vice versa.
- Infrastructure may depend on everything; nothing depends on infrastructure.

---

## Forbidden Dependencies

> This section must be adapted per project.

Examples:
- Evaluation may not infer model behavior.
- Models may not reach into evaluation internals.
- Configuration may not be reconstructed from runtime behavior.
- Experimental code may not be depended upon by stable components.

Forbidden dependencies are enforced socially and through code review.

---

## Allowed Flexibility

This topology:
- does **not** prescribe directory structure
- does **not** forbid all cross-cutting concerns
- does **not** constrain internal implementation details

It constrains **responsibility and authority**, not creativity.

---

## Consequences

### Positive
- Reduced architectural entropy
- Easier reasoning about changes
- Clearer boundaries for contributors

### Negative
- Some shortcuts are disallowed
- Requires reviewers to actively enforce rules

These constraints are intentional.

---

## Notes

This ADR governs *relationships between concepts*, not their internal semantics.  
Semantic authority is defined separately.
