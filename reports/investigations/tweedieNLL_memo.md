# Formal Sign-Off Memo

**Subject:** Adoption of Tweedie Negative Log-Likelihood for Decision-Grade Magnitude Estimation

---

## 1. Objective and decision requirement

The objective of this change is to ensure **decision-grade magnitude estimation**, defined as:

[
\frac{\bar{\hat y}}{\bar y} \approx 1
]

where (\bar{\hat y}) and (\bar y) denote predicted and observed magnitudes aggregated at decision-relevant levels (e.g. country–month, regional risk mass).

The loss function must therefore:

1. Penalize systematic mean bias
2. Handle zero-inflation and heavy tails
3. Avoid ad-hoc regime switching or threshold effects
4. Remain auditable and statistically interpretable

---

## 2. Proposed statistical model

### 2.1 Observation model

We assume observations follow a **Tweedie exponential dispersion model**:

[
Y \sim \text{Tweedie}(\mu, \phi, p), \quad 1 < p < 2
]

where:

* (\mu = \mathbb{E}[Y]) is the conditional mean
* (\phi > 0) is the dispersion parameter
* (p) is the Tweedie power parameter controlling variance structure

This family admits:

* a point mass at zero
* continuous positive outcomes
* heavy-tailed variance

with variance:
[
\text{Var}(Y) = \phi \mu^p
]

---

### 2.2 Link function (mean parameterization)

We use a **log link** to ensure positivity:

[
\eta = f_\theta(x), \quad \mu = \exp(\eta)
]

where:

* (f_\theta(\cdot)) is the model (e.g. neural network)
* (\eta) is the linear predictor

This guarantees (\mu > 0) without thresholds or clipping.

---

### 2.3 Parameter Selection Strategy

The Tweedie distribution is defined by the power parameter `p` and the dispersion `φ`. For a decision-grade implementation, these must be handled in a principled manner.

*   **Tweedie Power `p`:** This parameter is a critical hyperparameter that defines the model\'s assumed variance structure. It will be selected by profiling the negative log-likelihood on a validation set across a range of candidate values (e.g., in `[1.1, 1.9]`). The value of `p` that yields the best validation performance will be used for the final model.

*   **Dispersion `φ`:** This parameter scales the overall loss and can be treated as a nuisance parameter during model training. For the purpose of optimizing the model weights `θ` that determine `μ`, `φ` can be fixed to `1` without affecting the location of the loss minimum. It only becomes necessary to estimate `φ` (typically from model residuals) if performing full statistical inference or generating prediction intervals.

---

## 3. Loss function: Tweedie negative log-likelihood

The training objective is the **negative log-likelihood**:

[
\mathcal{L}_{\text{Tweedie}}(\mu; y)
= - \log p(y \mid \mu, \phi, p)
]

For (y \ge 0) and (1 < p < 2), this can be expressed (up to constants related to `y` and `φ`) as:

[
\mathcal{L}_{\text{Tweedie}}
\propto
\frac{\mu^{2-p}}{2-p}
-
\frac{y \mu^{1-p}}{1-p}
]

**Key property (critical):**

[
\arg\min_\mu ; \mathbb{E}[\mathcal{L}_{\text{Tweedie}}]
;\Rightarrow;
\mu = \mathbb{E}[Y]
]

Thus, **mean bias is directly penalized by construction**, ensuring (\bar{\hat y}/\bar y \to 1) when the model is well specified.

---

## 4. Optional calibration safeguard (conditional)

If empirical diagnostics show persistent mean bias after correct specification, a **light global calibration regularizer** may be added:

[
\mathcal{L}
=
\mathcal{L}_{\text{Tweedie}}
+
\lambda
\left(
\frac{\mathbb{E}[\hat y]}{\mathbb{E}[y]} - 1
\right)^2
]

This term:

* operates at batch or validation level
* does **not** alter local gradients strongly
* is only introduced after diagnostics justify it

This is considered a **safeguard**, not part of the primary model.

---

## 5. Pseudocode (loss computation)

### 5.1 Core Tweedie loss (log-link)

```python
function tweedie_nll(pred_eta, y, p, eps):
    # pred_eta: model output (real-valued)
    # y: observed targets (y >= 0)
    # p: Tweedie power parameter (1 < p < 2)
    # eps: numerical guard (small positive constant)

    mu = clamp(exp(pred_eta), min=eps)

    term1 = y * mu**(1 - p) / (1 - p)
    term2 = mu**(2 - p) / (2 - p)

    nll = term2 - term1
    return mean(nll)
```

---

### 5.2 Optional mean-ratio regularizer

```python
function mean_ratio_penalty(mu, y, eps):
    mu_bar = mean(mu)
    y_bar = mean(y) + eps
    ratio = mu_bar / y_bar
    return (ratio - 1.0)**2
```

```python
total_loss = tweedie_nll(...) + lambda * mean_ratio_penalty(mu, y)
```

---

## 6. Acceptance criteria (mathematically defined)

A model trained under this plan is accepted if:

1. **Mean calibration**
   [
   0.95 \le \frac{\bar{\hat y}}{\bar y} \le 1.05
   ]
   across:

* validation period
* rolling temporal slices
* decision-relevant spatial aggregation

2. **Error control**
   At least one standard regression metric improves or remains stable relative to baseline:

* MAE
* MSLE

3. **Stability**
   No systematic drift in (\bar{\hat y}/\bar y) across time or regime changes.

---

## 7. Risk management and auditability

* No hard thresholds or FP/FN masks are used
* All assumptions (non-negativity, variance structure) are explicit
* Loss is a **proper scoring rule**
* Failure modes (mis-specification of (p), dispersion instability) are observable via diagnostics

---

## 8. Final decision

Based on:

* the mathematical correctness of the objective,
* its alignment with decision-grade requirements,
* and its auditability,

this plan is **approved for implementation and evaluation**.

---