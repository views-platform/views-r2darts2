

---

# Memo

**Title:** Verification and Validation of Trainer, Optimizer, and Control Hyperparameters
**Context:** Mission-critical forecasting models with noisy, heavy-tailed, and zero-inflated targets
**Audience:** Model developers, reviewers, and technical auditors
**Purpose:** Ensure that training hyperparameters are (i) real, (ii) correctly implemented, and (iii) justified by evidence rather than convention.

---

## 1. Motivation and Scope

The training configuration under review includes a set of hyperparameters governing optimization, learning-rate control, gradient clipping, and stopping criteria. While such parameters are often treated as routine or “standard,” in high-stakes forecasting systems this assumption is unsafe.

In particular:

* Some hyperparameters **silently fail** if mis-wired (e.g. schedulers, early stopping).
* Others **behave differently depending on implementation details** (e.g. Adam vs AdamW).
* Several interact in ways that can **mask real learning**, **bias magnitude estimates**, or **prematurely terminate training**.
* Visual inspection of loss curves alone is insufficient to establish correctness.

This memo lays out a systematic approach to **verifying, testing, and warranting** each class of hyperparameter.

---

## 2. Principles for Hyperparameter Verification

We distinguish between three questions that must be answered separately:

1. **Existence:**
   Does this hyperparameter actually affect the training process in the current codebase?

2. **Correctness:**
   Is it implemented in the way the developer intends (semantics, timing, scale)?

3. **Warrant:**
   Is its use justified for this model class and data-generating process?

Each hyperparameter (or tightly coupled group) should be validated against all three.

---

## 3. Trainer-Level Controls

### 3.1 Number of Epochs (`n_epochs`)

**What:**
Defines a hard upper bound on the number of full passes over the training data.

**Why verification matters:**
In the presence of early stopping, `n_epochs` often becomes a nominal parameter. If early stopping is broken or misconfigured, `n_epochs` becomes the *de facto* stopping rule.

**How to verify:**

* Log the actual stopping epoch.
* Explicitly assert that training halts before `n_epochs` in normal runs.
* Run a controlled experiment with early stopping disabled to confirm that `n_epochs` is respected.

**Warrant test:**

* Confirm that validation performance does not systematically improve after the typical stopping epoch.
* Ensure `n_epochs` is large enough to allow scheduler-driven LR reductions to take effect.

---

## 4. Optimizer Configuration

### 4.1 Optimizer Choice (`optimizer_cls = Adam`)

**What:**
Adam provides adaptive per-parameter learning rates using first and second moments.

**Why verification matters:**
The semantics of several other hyperparameters (especially weight decay) depend on the optimizer class.

**How to verify:**

* Log the optimizer class at runtime.
* Explicitly check parameter groups and optimizer state dict contents.

**Warrant test:**

* Confirm that gradient noise and sparsity justify an adaptive optimizer.
* Compare against a fixed-LR baseline (e.g. SGD) to confirm Adam is not masking deeper issues.

---

### 4.2 Weight Decay (`weight_decay = 0.0003`)

**What:**
Applies parameter norm penalization.

**Why verification matters:**
In PyTorch:

* `Adam` implements **L2 regularization coupled to the gradient**
* `AdamW` implements **decoupled weight decay**

These are *not equivalent*, and the difference is consequential.

**How to verify:**

* Inspect optimizer class (`Adam` vs `AdamW`).
* Check whether decay is applied inside the gradient or as a separate step.
* Run a diagnostic training where weight decay is set to zero and confirm a measurable change in parameter norms.

**Warrant test:**

* Track parameter norm trajectories with and without weight decay.
* Assess whether decay disproportionately suppresses magnitude predictions.
* Confirm that decay improves generalization rather than merely stabilizing training.

---

## 5. Gradient Control

### 5.1 Gradient Clipping (`gradient_clip_val = 0.64`)

**What:**
Caps the global norm of the gradient vector.

**Why verification matters:**
Clipping can become the *dominant* control on learning if it activates frequently, effectively overriding the learning rate.

**How to verify:**

* Log gradient norms before and after clipping.
* Log the fraction of steps where clipping is active.
* Assert that clipping is applied *after* backpropagation and *before* optimizer stepping.

**Warrant test:**

* If >30–40% of steps are clipped, reassess learning rate and loss scaling.
* Verify that removing clipping does not lead to divergence but does reveal instability that clipping was masking.

---

## 6. Learning Rate Scheduling

### 6.1 Scheduler Type (`ReduceLROnPlateau`)

**What:**
Reduces the learning rate when a monitored metric stops improving.

**Why verification matters:**
This scheduler is frequently mis-used:

* Stepped on training loss instead of validation loss
* Stepped every batch instead of every epoch
* Stepped without passing the monitored metric

In all cases, it silently degrades into a no-op or behaves erratically.

**How to verify:**

* Log the metric passed to `scheduler.step()`.
* Log learning rate values each epoch.
* Force a synthetic plateau (e.g. freeze model weights) and confirm LR reduction.

**Warrant test:**

* Confirm LR reductions correspond to genuine stagnation, not noise.
* Ensure LR changes precede renewed improvement, not immediate stopping.

---

### 6.2 Scheduler Patience, Factor, and Minimum LR

**Parameters:**

* `lr_scheduler_patience = 7`
* `lr_scheduler_factor = 0.46`
* `lr_scheduler_min_lr = 1e-5`

**Why verification matters:**
These parameters define the *tempo* of learning. Poor coordination with early stopping can result in either wasted epochs or premature termination.

**How to verify:**

* Log LR trajectory over time.
* Log scheduler trigger events.
* Ensure LR never falls below the specified minimum.

**Warrant test:**

* Confirm at least one LR reduction occurs in typical runs.
* Validate that performance improves or stabilizes after reductions.

---

## 7. Early Stopping

### 7.1 Patience and Minimum Delta

**Parameters:**

* `early_stopping_patience = 20`
* `early_stopping_min_delta = 0.001`

**What:**
Stops training if no improvement exceeding `min_delta` occurs for `patience` epochs.

**Why verification matters:**
`min_delta` is scale-dependent. If set too large relative to the metric’s natural variability, real learning will be ignored.

**How to verify:**

* Log raw validation metrics each epoch.
* Log whether each epoch is counted as an “improvement.”
* Run a controlled experiment with `min_delta = 0` and compare stopping points.

**Warrant test:**

* Compare `min_delta` to the empirical distribution of epoch-to-epoch improvements.
* Ensure stopping decisions are not dominated by noise thresholds rather than signal.

---

## 8. Integrated System Tests

Beyond individual checks, the following **system-level tests** are recommended:

1. **Ablation Runs**

   * Disable one control at a time (scheduler, clipping, early stopping).
   * Confirm qualitative changes in training dynamics.

2. **Synthetic Stress Tests**

   * Train on trivial or frozen data to force plateaus.
   * Confirm scheduler and early stopping respond as expected.

3. **Logging Consistency Checks**

   * Ensure logged losses correspond to the loss actually backpropagated.
   * Verify that monitoring metrics match decision metrics.

---

## 9. Conclusion

In complex forecasting systems, hyperparameters governing training dynamics are **part of the model**, not incidental configuration. They must therefore be:

* **Verified** (they exist and are active),
* **Validated** (they behave as intended),
* **Warranted** (they are justified for the data and objective).

Failure to do so risks deploying models that appear stable, performant, and well-behaved, while in reality being shaped by unintended or opaque control logic.

This memo provides a framework for ensuring that such risks are systematically identified and mitigated.
