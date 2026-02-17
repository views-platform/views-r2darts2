
What follows is a **clean, minimal, PyTorch-native specification** of the loss you want, written so it can go **directly into a memo or ADR**.

I’ll give you, in order:

1. **Conceptual description (plain language, but precise)**
2. **Mathematical definition of the loss**
3. **Design choices and why they matter**
4. **Minimal PyTorch-style pseudocode**

No spatial assumptions, no fancy batching tricks beyond short temporal panels.

---

# 1. Loss description (conceptual)

We model observed country-level conflict magnitude (y_t \ge 0) as the realization of a **latent, slowly evolving risk process**.

The model outputs a **latent risk field** in log-space:
[
r_t := \log \mu_t
]
where (\mu_t = \mathbb{E}[y_t \mid x_t]).

The loss has two components:

1. **Data fidelity (Tweedie NLL)**
   Ensures correct magnitude estimation, heavy-tail robustness, and mean calibration.

2. **Temporal coherence regularization**
   Encourages the latent risk field to evolve smoothly over time, reflecting the belief that conflict risk does not jump arbitrarily month-to-month.

The regularizer operates **only on the latent risk field**, not directly on observations.

---

# 2. Mathematical formulation

## 2.1 Observation model

We assume a Tweedie exponential dispersion model:
[
Y_t \sim \text{Tweedie}(\mu_t, \phi, p),
\qquad 1 < p < 2
]

with:
[
\text{Var}(Y_t) = \phi \mu_t^p
]

The model predicts:
[
r_t = f_\theta(x_t), \qquad \mu_t = \exp(r_t)
]

---

## 2.2 Data term: Tweedie negative log-likelihood

The primary loss is:
[
\mathcal{L}_{\text{Tweedie}}
============================

-\log p(y_t \mid \mu_t, \phi, p)
]

Up to constants independent of (\mu_t), this can be written as:
[
\mathcal{L}_{\text{Tweedie}}
\propto
\frac{\mu_t^{2-p}}{2-p}
-----------------------

\frac{y_t \mu_t^{1-p}}{1-p}
]

This loss is a **proper scoring rule** and is minimized (in expectation) when:
[
\mu_t = \mathbb{E}[Y_t \mid x_t]
]

---

## 2.3 Temporal risk-field regularizer

We impose **first-order temporal smoothness** on the latent risk field:
[
\mathcal{R}_{\text{time}}
=========================

\frac{1}{T-1}
\sum_{t=2}^{T}
\left(r_t - r_{t-1}\right)^2
]

Interpretation:

* Penalizes abrupt month-to-month changes in latent risk
* Encourages the model to represent conflict as an evolving process, not iid noise
* Operates in **log-space**, so smoothness is multiplicative in (\mu_t)

---

## 2.4 Total loss

For a temporal panel of length (T):

[
\boxed{
\mathcal{L}
===========

\mathcal{L}*{\text{Tweedie}}
+
\beta , \mathcal{R}*{\text{time}}
}
]

where:

* (\beta \ge 0) controls how strongly we enforce risk-field coherence
* (\beta) is chosen so that (\beta \mathcal{R}_{\text{time}}) is a modest fraction (≈5–20%) of the Tweedie loss during training

---

# 3. Design choices (why this is safe and minimal)

### Why regularize (r_t = \log \mu_t) instead of (\mu_t)?

* Additive smoothness in (r_t) ⇒ multiplicative smoothness in (\mu_t)
* Prevents artificial suppression of magnitude
* Stable gradients even when (\mu_t) varies over orders of magnitude

### Why first-difference (not second)?

* First-difference already encodes “risk doesn’t jump arbitrarily”
* Fewer assumptions than curvature penalties
* Less risk of oversmoothing structural breaks

### Why not regularize observations?

* We do **not** assume observations are smooth
* We assume the **latent process** is smooth
* This is the key conceptual distinction

---

# 4. Minimal PyTorch-style pseudocode

Assume:

* batch contains **short temporal panels per country**
* shape:
  `x: [B, T, F]`
  `y: [B, T]`

---

## 4.1 Tweedie NLL (core)

```python
def tweedie_nll(y, mu, p, phi=1.0, eps=1e-8):
    """
    y  : [B, T] observed targets (>= 0)
    mu : [B, T] predicted mean (> 0)
    """
    mu = mu.clamp_min(eps)

    term1 = y * mu.pow(1.0 - p) / (1.0 - p)
    term2 = mu.pow(2.0 - p) / (2.0 - p)

    nll = (term2 - term1) / phi
    return nll.mean()
```

---

## 4.2 Temporal risk-field regularizer

```python
def temporal_risk_regularizer(r):
    """
    r : [B, T] latent risk field (log-mean)
    """
    return (r[:, 1:] - r[:, :-1]).pow(2).mean()
```

---

## 4.3 Full loss in training step

```python
# forward
r  = model(x)                 # [B, T]
mu = r.exp()                  # ensure positivity

# losses
loss_data = tweedie_nll(y, mu, p=p, phi=phi)
loss_time = temporal_risk_regularizer(r)

# total
loss = loss_data + beta * loss_time

loss.backward()
optimizer.step()
```

---

## 5. What this loss *rewards* (explicitly)

* Correct **mean magnitude** (via Tweedie)
* Coherent **latent risk dynamics** (via regularizer)
* Separation between:

  * noisy manifestation (data term)
  * smooth underlying risk (regularizer)

And importantly:

> It does **not** require
> – event labels
> – thresholds
> – spatial graphs
> – custom batching hacks

Just short temporal panels.

---

## 6. Acceptance checks (non-negotiable)

After training with this loss, you should verify:

1. **Mean calibration**
   [
   \bar{\hat y} / \bar y \approx 1
   ]

2. **No magnitude collapse**
   [
   \mathbb{E}[\hat y - y] \approx 0
   ]

3. **Reduced temporal noise**

* Month-to-month volatility of (\log \hat \mu_t) decreases
* Without flattening real trend shifts

---

### Bottom line

This is the **minimum viable loss** that:

* respects your double/triple stochastic intuition,
* remains PyTorch-native,
* avoids brittleness,
* and directly supports decision-grade magnitude estimation.

If you want, next we can:

* choose a principled initial value for (\beta),
* define a tiny synthetic test to validate behavior,
* or write this up as an **ADR or Methods subsection** verbatim.
