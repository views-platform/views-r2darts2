
# Memo: Ensuring Correctness of Mission-Critical Loss-Function Implementations

**Scope:** Poisson NLL, Tweedie (compound Poisson–Gamma / EDM form), Huber / SmoothL1, and custom weighted/penalty Huber variants.
**Goal:** Make it *auditable* that the implemented loss matches (i) an authoritative mathematical definition and (ii) trusted reference implementations, and that it behaves correctly under edge cases and gradients—so failures are detected *before* deployment.

---

## 1) What “correct” means (mission-critical definition)

A loss implementation is **correct** only if all of the following hold:

1. **Spec correctness (math):** The implemented expression matches an authoritative definition (up to explicitly documented constants), under explicit domain assumptions.
2. **Numerical correctness:** The loss returns expected values on a suite of **golden test vectors**, including adversarial edge cases.
3. **Gradient correctness:** Autograd gradients match finite-difference gradients within pre-defined tolerance on representative inputs and near kinks/boundaries.
4. **Behavioral correctness:** The loss exhibits required invariants (monotonicity, convexity where expected, minimizer at correct point, continuity at piecewise joins, etc.).
5. **Cross-implementation agreement:** Results match at least **two independent reference implementations** (library or alternative codebase), to within tolerance, on shared inputs.
6. **Operational safety:** The loss has explicit handling for invalid domains (negative means, NaNs, infinities) and fails fast rather than silently producing misleading gradients.

Anything less is “works on my machine,” not mission-critical.

---

## 2) Deliverables you need for auditability

You want the following artifacts **checked into the repo**:

### A. Loss Specification Sheet (LSS) — one per loss

A short, fixed-format document containing:

* **Name & version** (e.g., `TweedieNLL v1.0`)
* **Intended statistical model** (e.g., Poisson likelihood; Tweedie EDM with (1<p<2))
* **Exact formula** (with and without constants if relevant)
* **Domain constraints**:

  * Poisson: (y \in \mathbb{N}_0), (\lambda>0)
  * Tweedie: (y \ge 0), (\mu>0), (\phi>0), (1<p<2) (compound Poisson–Gamma regime)
  * Huber: residual (r \in \mathbb{R}), (\delta>0)
* **Parameterization choices**:

  * Are you predicting (\lambda) or (\log \lambda)?
  * Are you predicting (\mu) via softplus / exp?
  * Are you including the (\log(y!)) term (Poisson) or dropping constants?
* **Numerical stabilizers** (eps clamps, softplus, log1p tricks)
* **Edge-case policy** (what happens at (y=0), (\mu \rightarrow 0), huge (y), etc.)
* **Known equivalences** (SmoothL1 ≈ Huber with a specific scaling/threshold)

### B. Reference Map

For each loss, list:

* **Primary mathematical source** (authoritative text/paper)
* **At least two reference implementations** (e.g., PyTorch, statsmodels, scikit-learn, R packages, LightGBM)
* Notes on differences (constants, scaling, approximations)

### C. Test suite

* **Golden value tests**
* **Invariant tests**
* **Gradient checks**
* **Simulation tests** (data generated from assumed DGP)

### D. CI integration + kill-switch

* CI must run all tests on every PR.
* If any loss test fails → **block merge** and/or **disable loss** in production configs.

---

## 3) Verification workflow (step-by-step)

This is the practical procedure to follow for each loss.

### Step 1 — Freeze the exact variant

Loss “names” are ambiguous. Freeze the knobs that change math.

**Poisson NLL**:

* `log_input`? (model outputs (\eta=\log\lambda) vs (\lambda))
* include factorial term `full`? (constant for training, but important for likelihood reporting)
* epsilon clamp?

**Huber / SmoothL1**:

* threshold (\delta) (or `beta`)
* scaling factor (many implementations differ by a constant factor)

**Tweedie**:

* whether you’re using **EDM NLL up to constants** vs **deviance**
* power (p) regime: confirm (1<p<2) if you intend compound Poisson–Gamma
* link function for (\mu) (identity vs log)
* dispersion (\phi): fixed? learned? omitted?
* treatment of (y=0)

**Weighted/Penalty Huber**:

* define weight function (w(\cdot)) precisely (depends on (y)? on residual sign? on predicted magnitude?)
* define what “penalty” means (rare-event upweighting? asymmetric costs? non-zero emphasis?)

> Output: LSS draft with all knobs fixed.

---

### Step 2 — Lock the canonical formula + mapping to your code

For each loss, write (in the LSS) a mapping from spec symbols to code variables, e.g.:

* `pred` -> (\eta) or (\lambda)
* `target` -> (y)
* `beta` -> (\delta)
* `p` -> (p) (Tweedie power)

**Crucial:** explicitly document constants that are dropped (e.g., Poisson (\log(y!))). In mission-critical settings, you often want **both forms**:

* training objective (constants dropped OK)
* reporting likelihood (constants included)

---

### Step 3 — Cross-check against reference implementations

Pick **two independent references** and compare on identical inputs.

Recommended pairings:

* **Poisson NLL:** PyTorch `PoissonNLLLoss` + statsmodels Poisson log-likelihood
* **Huber:** statsmodels robust Huber (\rho) + PyTorch `SmoothL1Loss` (with explicit parameter matching)
* **Tweedie:** (i) R `tweedie`/`statmod` density/logLik OR LightGBM’s Tweedie objective, plus (ii) a Python reference (TorchMetrics deviance or your own derived EDM form)

**Acceptance criterion**
Define numerical tolerances explicitly, e.g.:

* float64: max abs error < 1e-10 on golden vectors
* float32: max abs error < 1e-5
* gradients: relative error < 1e-4 away from kinks; handle kink points separately

---

### Step 4 — Golden test vectors (deterministic)

Create a small set of hand-curated test cases (always included in CI), with expected outputs:

#### Poisson NLL golden cases

* (y=0), varying (\lambda)
* (y=1,2,10) with (\lambda=y) (should be near-optimal)
* huge (y) and huge (\lambda) (stress stability)
* log-input vs linear-input mode equivalence test

#### Huber golden cases

* residuals: (-2\delta, -\delta, -0.5\delta, 0, 0.5\delta, \delta, 2\delta)
* check continuity at (|r|=\delta)

#### Tweedie golden cases (most important)

* (y=0), (\mu) small/medium/large
* very small positive (y) near 0 (if your data can have this)
* large (y)
* (p) in representative values (e.g. 1.1, 1.5, 1.9) if you sweep it, or fixed p if not

> Golden tests catch “implementation drift” and accidental refactors.

---

### Step 5 — Invariant tests (property-based)

These tests don’t require a reference implementation—only math facts.

#### Poisson invariants

* For (y=0), NLL should increase with (\lambda).
* For fixed (y), NLL minimized at (\lambda=y).
* Convex in (\lambda) (for (\lambda>0)).

#### Huber invariants

* Quadratic regime: for (|r|\ll \delta), behaves like (0.5r^2).
* Linear regime: for (|r|\gg \delta), grows ~(\delta|r|).
* Continuous and continuously differentiable at (|r|=\delta).

#### Tweedie invariants (in practice)

* Domain: invalid (\mu \le 0) must never silently pass.
* For (1<p<2): should support zero outcomes; loss must be finite at (y=0) under your parameterization.
* Reduction checks (if you implement them): behavior should approach Poisson-like as (p\to 1), Gamma-like as (p\to 2). (Even if not exact due to constants, trends should match.)

#### Weighted/Penalty Huber invariants

* If weights are constant (w=c), loss scales by (c) and gradients scale by (c).
* If weights depend on (y) only, then for identical residuals, relative gradient magnitudes reflect weights.
* No discontinuity introduced at Huber join beyond the original join.

---

### Step 6 — Gradient verification (non-negotiable)

Use finite differences to verify gradients:

* random tensors (typical range)
* edge cases:

  * Poisson/Tweedie: (\mu) near 0 (but positive)
  * Huber: residuals near (\pm \delta)
  * weighted variants: maximal weights

**Rule:**

* Evaluate gradient checks **away from kinks** (Huber join) for strict error bounds.
* At kinks, verify subgradient behavior / one-sided derivatives (document policy).

---

### Step 7 — Simulation-based falsification tests

This is the “does it optimize what we think?” layer.

* Generate synthetic data from the assumed DGP:

  * Poisson((\lambda)) → Poisson NLL should recover (\lambda) better than MSE/Huber.
  * Compound Poisson–Gamma → Tweedie loss should dominate Poisson NLL and MSE on likelihood / calibration metrics.
* Fit a trivial model (even a single-parameter mean model) and verify:

  * optimization converges
  * recovered parameters near truth
  * mis-specified loss is detectably worse

This is the most persuasive evidence in mission-critical reviews because it’s empirical falsification.

---

## 4) Operational safety requirements (what to enforce in code)

### A. Domain enforcement (fail fast)

* If (\mu) must be positive: enforce via `softplus` (and document) or assert (>0) with a clean error.
* If (y) must be non-negative integer counts (Poisson): validate upstream; if floats appear, decide policy (round? treat as rate? error?).
* If NaNs occur: stop training, dump batch, and log full context.

### B. Numerical stability policy

* Define a single shared `EPS` constant (documented), with rationale.
* Use stable transforms:

  * `log_input=True` style for Poisson is typically more stable.
  * For Tweedie deviance/NLL, avoid direct `log(0)` and fragile powers without clamping.

### C. Consistency across precisions

* Test float32 and float64.
* Optional: test AMP/mixed precision if used.

### D. Monitoring hooks

Even with perfect tests, you want runtime guards:

* track loss distribution
* track gradient norms
* detect divergence
* detect domain violations (counts negative, etc.)

---

## 5) Special notes per loss (what typically goes wrong)

### Poisson NLL

Common failure modes:

* Mixing up whether network outputs (\lambda) vs (\log\lambda).
* Forgetting that (\log(y!)) is constant w.r.t. (\lambda) but matters for **reported likelihood**.
* Zero or negative (\lambda) due to missing positivity transform.

### Huber / SmoothL1

Common failure modes:

* Wrong scaling: some implementations use (0.5r^2), some use (r^2).
* Misinterpreting `beta`/`delta`.
* Discontinuous gradient due to incorrect piecewise join.

### Tweedie

Common failure modes:

* Using (p) outside the intended regime.
* Silent instability near zero mean.
* Implementing “deviance-like” objective but calling it “NLL” (fine if documented, disastrous if not).
* Not documenting whether dispersion (\phi) is included or omitted.
* Edge-case mishandling at (y=0).

### Weighted/Penalty Huber

Common failure modes:

* Weighting introduces gradient explosion.
* Asymmetric penalties implemented incorrectly (sign errors).
* Weight depends on prediction and creates unintended feedback loops (document if intended).

---

## 6) Minimal acceptance checklist (copy/paste)

A loss is **approved for production** only if:

* [ ] LSS exists and is versioned.
* [ ] Primary-source formula and symbol mapping are documented.
* [ ] Two independent reference implementations agree within tolerance on golden vectors.
* [ ] Invariant tests pass.
* [ ] Gradient checks pass (finite difference).
* [ ] Simulation-based falsification tests pass.
* [ ] Domain enforcement & NaN policy implemented (fail-fast).
* [ ] CI blocks merges on failures.
* [ ] Runtime monitoring hooks exist for deployment.

---

This document may contain errors or omissions.

