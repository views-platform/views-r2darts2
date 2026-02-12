
# ADR README and Governance Map

This repository uses Architectural Decision Records (ADRs) to govern
structural, semantic, and operational behavior.

ADRs are divided into two categories:

1. **Constitutional ADRs (001–009)**  
   Foundational architectural rules that apply across the system.

2. **Project-Specific ADRs (010+)**  
   Domain, implementation, or feature-level decisions.

---

## Constitutional ADRs

These ADRs define system philosophy and governance:

- **ADR-001** — Ontology of the Repository  
  Defines what concepts exist.

- **ADR-002** — Topology and Dependency Rules  
  Defines structural dependency direction.

- **ADR-003** — Authority of Declarations Over Inference  
  Defines where semantic authority lives.

- **ADR-004** — Rules for Evolution and Stability  
  Defines Stability Tiers for the Fortress architecture.

- **ADR-005** — Testing as Mandatory Critical Infrastructure  
  Defines red / beige / green test doctrine.

- **ADR-006** — Intent Contracts for Non-Trivial Classes  
  Requires declared class-level purpose.

- **ADR-007** — Silicon-Based Agents as Untrusted Contributors  
  Governs automated modification.

- **ADR-008** — Observability and Explicit Failure  
  Defines fail-loud + log requirements.

- **ADR-009** — Boundary Contracts and Configuration Validation  
  Defines explicit interface contracts and configuration validation.

These ADRs form the architectural constitution of the repository.

---

## Operational & Domain ADRs

ADRs numbered 010 and above define specific infrastructure, mathematical, and domain-level decisions:

- **ADR-010** — Numerical Precision and Semantic Thresholds  
  Standardizes `float32` and prohibits hardcoded model floors.

- **ADR-011** — Hardware Integrity and Parallelism  
  Defines device self-healing and GPU prediction constraints.

- **ADR-012** — Scaling Pipeline and Calibration Integrity  
  Standardizes on Darts `Pipeline` and mandatory `global_fit`.

These must comply with the constitutional ADRs above.

---

## Governance Structure (Conceptual Map)

- **Ontology (001)** defines what exists.
- **Topology (002)** defines structural direction.
- **Authority (003)** defines who owns meaning.
- **Evolution (004)** defines stability tiers.
- **Boundary Contracts (009)** define interaction rules.
- **Numerical Laws (010)** ensure precision and raw intentionality.
- **Hardware Laws (011)** prevent race conditions and device drift.
- **Mathematical Laws (012)** preserve probabilistic calibration.
- **Observability (008)** enforces failure semantics.
- **Testing (005)** verifies system integrity.
- **Intent Contracts (006)** bind class-level behavior.
- **Automation Governance (007)** constrains silicon-based agents.

Together, these define the invariant layer of the Fortress architecture.
