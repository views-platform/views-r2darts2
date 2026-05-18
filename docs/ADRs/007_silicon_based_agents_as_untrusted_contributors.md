# ADR-007: Silicon-Based Agents as Untrusted Contributors

**Status:** Accepted  
**Date:** 2026-02-11  
**Deciders:** Simon Polichinel von der Maase  

---

## Context

This repository is developed with the assistance of **silicon-based agents** (LLMs, coding assistants). While these agents are powerful accelerators, they differ from carbon-based agents in critical ways:
- They prioritize local plausibility over global architectural correctness.
- They lack a genuine understanding of "Temporal Integrity" and the "Fortress" mindset.
- They are prone to **silent failures**, such as file truncation or "logical drift," where they replace complex logic with structurally valid but incorrect code.

Without explicit guardrails, silicon agents introduce risks that are difficult to detect during standard code review.

---

## Decision

Silicon-based agents are treated as **untrusted contributors**. They are permitted to assist in code modification only under explicit constraints and never as autonomous authorities.

All silicon agent activity must comply with:
1. **The Silicon-Based Agent Protocol:** Defined in `docs/contributor_protocols/silicon_based_agents.md`.
2. **The Anti-Truncation Rule:** Full-file rewrites of existing files are forbidden to prevent "silent lobotomy" failures.
3. **Mandatory Edit-In-Place:** Existing files must be modified using targeted, scoped replacements (like the `replace` tool) rather than `write_file`.

---

## Authority and Responsibility

- **Intent Ownership:** Silicon agents do not own intent. They cannot declare new ontology (ADR-001) or change topology (ADR-002) without a carbon-based agent's explicit, reviewed command.
- **Responsibility:** Carbon-based agents remain 100% responsible for all code committed to the repository. "The AI did it" is not a valid defense for architectural violations.
- **Review Posture:** Silicon-assisted changes require **heightened scrutiny**. Reviewers must check for "plausible hallucinations" in math and logic.

---

## Scope

This ADR applies to any non-carbon-based system that proposes or applies code changes. It does not regulate read-only analysis tools or framework-internal generators.

---

## Consequences

### Positive
- **Architectural Preservation:** Prevents the silent erosion of the "Fortress" gates.
- **Reliability:** The Anti-Truncation Rule eliminates a major class of automation-induced bugs.
- **Traceability:** Forces agents to explain *why* they are making a change relative to existing ADRs.

### Negative
- Adds friction to the development workflow.
- Requires carbon-based agents to maintain better "agent-steering" discipline.

---

## Notes

This ADR establishes **that** agents are constrained. The specific operational rules (how to read files, how to apply diffs) are governed by the **Silicon-Based Agent Protocol**.

