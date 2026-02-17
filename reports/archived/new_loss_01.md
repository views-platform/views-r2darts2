

# Conditional Mean Calibration (CMC) Loss

**Implementation Guide for DARTS + PyTorch Lightning**

---

## 1. Problem Statement and Motivation

Forecasting models trained with MSLE or asymmetric log-scale losses on **zero-inflated, right-skewed, heavy-tailed targets** often exhibit the following pattern:

* Good MSLE and ranking performance
* Accurate identification of event timing and relative variation
* **Systematic underestimation of unconditional mean magnitude**

This manifests empirically as:

> Low `TIME_SERIES_WISE_MSLE_MEAN` but low `TIME_SERIES_WISE_Y_HAT`

where `TIME_SERIES_WISE_Y_HAT` is defined as the **unconditional mean prediction per time series**, averaged across series.

This behaviour is not a modeling bug but a **structural consequence of MSLE**, which:

* Compresses large values
* Strongly penalizes over-prediction
* Places most gradient mass near zero

As a result, models rationally shrink predicted scale to protect MSLE.

The **Conditional Mean Calibration (CMC)** loss addresses this *single, diagnosed failure mode* by adding a **minimal, series-level scale constraint** that aligns training with the evaluation metric.

---

## 2. Design Principles and Non-Goals

### 2.1 Design Principles

CMC is designed to be:

* **Minimal**: introduces exactly one additional constraint
* **Metric-aligned**: operates on the same quantity used for evaluation
* **Aggregation-consistent**: series-wise, not pointwise
* **Non-intrusive**: preserves dominance of the base loss
* **Auditable**: behavior is easy to reason about and test

### 2.2 Explicit Non-Goals

CMC is **not**:

* A hurdle or zero-inflated likelihood
* A tail modeling technique
* A probability calibration method
* A replacement for MSLE
* A structural change to the model architecture

CMC does **not** improve:

* Event timing
* Classification accuracy
* Pointwise prediction error

It corrects **only unconditional mean shrinkage at the series level**.

---

## 3. Mathematical Specification

### 3.1 Notation

Let:

* ( s \in {1, \dots, S} ) index time series
* ( t \in {1, \dots, T_s} ) index time steps within series ( s )
* ( y_{s,t} \ge 0 ) be the observed target
* ( \hat{y}_{s,t} \ge 0 ) be the model prediction

Predictions are assumed to be constrained to non-negativity (e.g. via `softplus`).

---

### 3.2 Series-Wise Unconditional Means

For each series ( s ):

[
\mu_y^{(s)} = \frac{1}{T_s} \sum_{t=1}^{T_s} y_{s,t},
\qquad
\mu_{\hat{y}}^{(s)} = \frac{1}{T_s} \sum_{t=1}^{T_s} \hat{y}_{s,t}
]

These are **exactly the quantities used in `TIME_SERIES_WISE_Y_HAT`**.

---

### 3.3 Log Mean-Ratio

Define the log mean-ratio per series:

[
r_s
===

\log!\left(
\frac{\mu_{\hat{y}}^{(s)} + \varepsilon}
{\mu_y^{(s)} + \varepsilon}
\right)
]

where ( \varepsilon > 0 ) is a small constant for numerical stability.

---

### 3.4 CMC Loss

Let ( S^+ = { s : \mu_y^{(s)} > \delta } ), where ( \delta ) is a small threshold excluding true zero-mass series.

The **CMC loss** is defined as:

[
\boxed{
\mathcal{L}_{CMC}
=================

\frac{1}{|S^+|}
\sum_{s \in S^+} r_s^2
}
]

---

### 3.5 Total Training Objective

Let ( \mathcal{L}_{base} ) be the primary loss (e.g. MSLE).

[
\boxed{
\mathcal{L}
===========

\mathcal{L}*{base}
+
\lambda*{CMC} \cdot \mathcal{L}_{CMC}
}
]

where ( \lambda_{CMC} \ll 1 ).

---

## 4. Algorithmic Description

For each training step:

1. Group predictions by `series_id`
2. Compute unconditional mean prediction per series
3. Compute unconditional mean target per series
4. Exclude series with negligible target mass
5. Compute squared log mean-ratio per series
6. Average across series
7. Add weighted auxiliary loss to base loss

The loss operates at **series level**, not individual time steps.

---

## 5. PyTorch Implementation

### 5.1 Loss Module

```python
import torch
import torch.nn as nn

class TimeSeriesWiseCMC(nn.Module):
    """
    Conditional Mean Calibration (CMC) loss.
    Aligns unconditional predicted mean per time series
    with observed mean.
    """

    def __init__(self, zero_threshold=0.0, eps=1e-6):
        super().__init__()
        self.zero_threshold = float(zero_threshold)
        self.eps = float(eps)

    def forward(self, y_hat, y, series_id):
        y_hat = y_hat.reshape(-1)
        y = y.reshape(-1)
        series_id = series_id.reshape(-1)

        losses = []
        for sid in torch.unique(series_id):
            idx = series_id == sid
            mu_y = y[idx].mean()
            if mu_y <= self.zero_threshold:
                continue

            mu_yhat = y_hat[idx].mean()
            log_ratio = torch.log(
                (mu_yhat + self.eps) / (mu_y + self.eps)
            )
            losses.append(log_ratio.pow(2))

        if len(losses) == 0:
            return y_hat.new_tensor(0.0)

        return torch.stack(losses).mean()
```

---

## 6. Integration with DARTS and PyTorch Lightning

### 6.1 Integration Strategy

CMC **must not** be implemented as a DARTS `loss_fn`, because:

* DARTS losses do not receive `series_id`
* CMC requires aggregation across series

CMC must be applied **inside `training_step`**.

---

### 6.2 Lightning Training Step (Example)

```python
def training_step(self, batch, batch_idx):
    y = batch["y"]
    series_id = batch["series_id"]

    y_hat = self(batch)
    y_hat = torch.nn.functional.softplus(y_hat)

    loss_base = msle_loss(y_hat, y)
    loss_cmc = self.cmc_loss(y_hat, y, series_id)

    loss = loss_base + self.lambda_cmc * loss_cmc

    self.log_dict({
        "loss_base": loss_base,
        "loss_cmc": loss_cmc,
        "loss_total": loss,
    })

    return loss
```

---

## 7. Required Diagnostics and Logging

The following must be logged during training and evaluation:

### Core

* Base loss
* CMC loss
* Total loss

### Alignment Diagnostics

* Series-wise mean ratio
  ( \mu_{\hat{y}}^{(s)} / \mu_y^{(s)} )
* Distribution of ratios across series

### Guardrails

* MSLE on zero-only observations
* MSLE on non-zero observations
* Fraction of predicted mass on zero-mass series

---

## 8. Validation and Unit Testing

### Required Tests

| Test             | Expected Result      |
| ---------------- | -------------------- |
| ( \hat y = y )   | CMC = 0              |
| ( \hat y = c y ) | CMC ≈ ( (\log c)^2 ) |
| All-zero series  | Ignored              |
| Backprop         | Finite gradients     |
| Seeded runs      | Deterministic        |

These tests must pass before experimentation.

---

## 9. Hyperparameter Protocol (λ)

### Recommended Sweep

[
\lambda_{CMC} \in {0, 0.01, 0.03, 0.1, 0.3}
]

### Interpretation

| Outcome                     | Interpretation                |
| --------------------------- | ----------------------------- |
| Mean ratio → 1, MSLE stable | Loss-induced bias corrected   |
| MSLE ↑ slightly             | Acceptable trade-off          |
| MSLE collapses              | Model insufficient            |
| No effect                   | Calibration pressure too weak |

λ is **diagnostic**, not a performance knob.

---

## 10. Interpretation and Failure Modes

### If CMC works:

> The model was adequate; the objective was incomplete.

### If CMC fails:

> Structural model changes are required (e.g. hurdle or mixture models).

### Common Pitfalls

* Over-weighting λ
* Hiding false positives
* Sparse positive series per batch

---

## 11. Scope and Extensions (Out of Scope for v1)

* Tail-only CMC
* Exposure-weighted CMC
* Multi-moment calibration

These are explicitly excluded from this guide.

---

## 12. Summary

**Conditional Mean Calibration (CMC)** is a minimal, series-wise auxiliary loss that corrects unconditional mean shrinkage induced by log-scale losses. It aligns training with evaluation without altering model structure or likelihood assumptions and provides a clean diagnostic boundary between objective-induced bias and model inadequacy.

---

