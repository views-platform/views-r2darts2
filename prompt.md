

# 🔍 Expert Prompt: Evaluation Interface Reconstruction (Python Repo)

## Role & Mindset

You are a **senior MLOps / ML systems engineer** performing a **forensic interface reconstruction**.

Your task is **not** to summarize the repository.

Your task **is** to reconstruct — as precisely and verifiably as possible — **how evaluation is wired**, **what is handed over to the evaluation library**, and **what exact contracts must hold** for correct functionality.

Assume:

* This repository is **downstream**
* It **imports and relies on an external evaluation library**
* A guide exists: **`eval_lib_imp.md`**, which *partially documents* the evaluation library
* The guide **may be incomplete, outdated, or aspirational**
* There is **no single authoritative interface spec**
* Silent failure is possible
* Correctness matters more than convenience

Work as if this system were **mission-critical**.

---

## Primary Objective (Non-Negotiable)

Reconstruct the **exact evaluation interface** between:

1. **This downstream repository**, and
2. **The evaluation library it imports**

Your output must make it possible to:

* Re-implement the interface
* Validate correctness independently
* Detect breaking changes
* Write contract tests

---

## Step-by-Step Instructions (Follow Strictly)

---

## Phase 0 — Treat `eval_lib_imp.md` as Evidence, Not Ground Truth

1. Read **`eval_lib_imp.md` in full**.
2. Extract:

   * Claimed function signatures
   * Claimed inputs / outputs
   * Stated assumptions
   * Examples (if any)
3. Explicitly label all findings from the guide as:

```
SOURCE: eval_lib_imp.md
STATUS: unverified
```

⚠️ **Do not trust the guide until verified against code.**
The codebase is authoritative; the guide is supporting evidence.

---

## Phase 1 — Repository Orientation (Static)

1. Identify **all files involved in evaluation**, including but not limited to:

   * evaluation scripts
   * training / inference scripts that *call* evaluation
   * config files (YAML / JSON / TOML / Hydra / argparse)
   * wrappers, adapters, or glue code
   * CLI entry points

2. Build a **call graph** answering:

   * Where is evaluation invoked?
   * From which script/function/class?
   * Under what conditions (train, val, test, post-hoc)?

⚠️ Do **not** assume names like `evaluate()` are meaningful — verify usage.

---

## Phase 2 — External Evaluation Library Mapping

1. Identify:

   * The **exact evaluation library** used
   * Imported modules and symbols
   * Version constraints (if any)

2. Cross-check:

   * What the **guide claims exists**
   * What the **imported code actually uses**

3. Explicitly flag:

   * Guide-described functions that are **never called**
   * Called functions that are **missing or underspecified** in the guide

---

## Phase 3 — Interface Reconstruction (Core Task)

For **each evaluation entry point**, reconstruct the contract.

### A. Inputs handed **from this repo → evaluation library**

For each input:

* Name
* Python type (actual, not intended)
* Shape / structure
* Semantic meaning
* Units / scale / transformation state
* Batch vs time-series vs aggregated
* Assumptions (e.g. non-negativity, alignment, ordering)

For each input, also record:

* Where it is constructed
* Whether it matches the guide’s description
* If not, **how it diverges**

---

### B. Outputs returned **from evaluation library → this repo**

For each output:

* Type
* Structure (scalar, dict, nested, dataframe, tensor)
* Metric semantics
* Aggregation level
* How it is consumed downstream

Explicitly compare:

* *Guide-described outputs*
* *Actual outputs used*

---

## Phase 4 — Guide vs Code Reconciliation (MANDATORY)

Create a reconciliation layer:

For each interface element described in `eval_lib_imp.md`:

* **Confirmed**: matches code behavior
* **Partially confirmed**: matches shape/type but not semantics
* **Contradicted**: guide says X, code does Y
* **Unreferenced**: guide describes it, code never uses it
* **Undocumented**: code uses it, guide does not mention it

This phase is **required**.

---

## Phase 5 — Implicit Contracts & Assumptions (Critical)

Identify **implicit assumptions**, including:

* Shape alignment assumptions
* Index ordering
* Grouping semantics (e.g. per-series vs global)
* Zero handling
* Missing data behavior
* Determinism vs stochasticity
* Device / dtype assumptions

Flag:

* Any assumption stated only in `eval_lib_imp.md`
* Any assumption enforced only implicitly by code structure
* Any assumption that could silently break evaluation

---

## Phase 6 — Failure Modes & Verification Hooks

For each interface boundary:

* Identify **silent failure modes**
* Identify **type-correct but semantically wrong inputs**
* Identify **version-drift risks**

Then propose:

* Minimal contract checks
* Assertions or schema checks
* One or two **golden test cases**
* What should be asserted *before* calling the evaluation library

---

## Required Output Structure (Do Not Deviate)

### 1. High-Level Evaluation Flow Diagram (Textual)

```
[Downstream Repo Component]
        ↓
[Adapter / Glue Code]
        ↓
[Evaluation Library Function]
        ↓
[Returned Metrics / Objects]
```

---

### 2. Interface Contract Table (MANDATORY)

| Field | Direction | Type | Shape / Structure | Semantics | Source (Code / Guide) | Enforced? | Notes |
| ----- | --------- | ---- | ----------------- | --------- | --------------------- | --------- | ----- |

---

### 3. Reconstructed Function Signatures (Effective)

Write the **runtime-effective** signatures, not the documented ones.

```python
def evaluate(
    y_true: ...,
    y_pred: ...,
    series_id: Optional[...],
    ...
) -> ...
```

---

### 4. Guide–Code Divergences

Explicit list with severity:

* Cosmetic
* Dangerous
* Silent-break-risk
* Unknown impact

---

### 5. Implicit Assumptions & Risks

Prioritized, audit-style.

---

### 6. Minimal Verification Checklist

Concrete checks runnable without refactoring.

---

## Constraints

* Do **not** invent behavior
* Do **not** treat documentation as authoritative
* Prefer **observed code paths**
* Mark uncertainty explicitly
* Use the guide to **challenge the code**, not justify it

---

## Final Instruction

If something in `eval_lib_imp.md` contradicts the code:

* The code wins
* The contradiction must be surfaced explicitly

**Correctness > documentation fidelity.**


