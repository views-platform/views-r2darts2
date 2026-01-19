# Loss Specification Sheet: WeightedPenaltyHuberLoss v1.0

## 1. Intended Statistical Model & Purpose

- **Name:** `WeightedPenaltyHuberLoss`
- **Version:** 1.0
- **Purpose:** An extension of `WeightedHuberLoss` designed for imbalanced classification-and-regression tasks, common in conflict forecasting. It allows for asymmetric penalization of different error types:
    1.  **False Positives (FP):** Predicting a conflict event when none occurred.
    2.  **False Negatives (FN):** Failing to predict a conflict event that did occur.
- It achieves this by applying separate multiplicative penalties on top of the base `non_zero_weight`.

## 2. Canonical Formula & Code Mapping

The loss is a multi-level weighted version of the standard Huber Loss.

`loss = mean( w_final * L(preds - targets) )`

### Huber Loss Formula (`L(e)`):
`L(e) = 0.5 * e^2` if `|e| <= delta`
`L(e) = delta * (|e| - 0.5 * delta)` if `|e| > delta`

### Final Weight Calculation (`w_final`):
The final weight is determined by a nested logic based on the type of outcome.

1.  **Calculate Base Weight (`w_base`):**
    `w_base = non_zero_weight` if `|targets| > zero_threshold`
    `w_base = 1.0` if `|targets| <= zero_threshold`

2.  **Identify Error Type:**
    - `is_target_nonzero = |targets| > zero_threshold`
    - `is_pred_nonzero = |preds| > zero_threshold`
    - **Important:** The `is_pred_nonzero` check is detached from the computation graph (`.detach()`). This means the choice of penalty does not directly influence the gradient calculation; it only scales the gradient that comes from the underlying Huber loss.
    - `is_fp = (NOT is_target_nonzero) AND is_pred_nonzero`
    - `is_fn = is_target_nonzero AND (NOT is_pred_nonzero)`

3.  **Apply Multiplicative Penalties:**
    - If `is_fp`: `w_final = w_base * false_positive_weight`
    - If `is_fn`: `w_final = w_base * false_negative_weight`
    - Otherwise (TP/TN): `w_final = w_base`

This leads to the following weight outcomes:
| Outcome | Target | Prediction | Final Weight | Example (defaults) |
| :--- | :--- | :--- | :--- | :--- |
| True Negative (TN)| ~0 | ~0 | 1.0 | 1.0 |
| True Positive (TP)| > 0 | > 0 | `non_zero_weight` | 5.0 |
| False Positive (FP)| ~0 | > 0 | `1.0 * false_positive_weight` | 2.0 |
| False Negative (FN)| > 0 | ~0 | `non_zero_weight * false_negative_weight`| 15.0 |

### Parameters:

| Symbol | Code Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `zt` | `self.threshold` | 0.01 | (`zero_threshold`) The threshold for considering a value non-zero. |
| `delta` | `self.delta` | 0.5 | The Huber loss threshold. |
| `nzw` | `self.non_zero_weight`| 5.0 | The base weight for non-zero targets. |
| `fpw` | `self.false_positive_weight` | 2.0 | The multiplicative penalty for false positives. |
| `fnw` | `self.false_negative_weight` | 3.0 | The multiplicative penalty for false negatives. |

## 3. Domain Constraints

- All parameters must be non-negative floats.

## 4. Edge Case Policy

- The classification of a prediction as zero/non-zero is a hard boundary at `zero_threshold` and is detached from the gradient computation.
- Underlying Huber loss properties at its `delta` threshold apply.

## 5. Known Equivalences & Invariants

- If `false_positive_weight = 1.0` and `false_negative_weight = 1.0`, the loss **must be identical** to `WeightedHuberLoss` with the same `delta`, `zero_threshold`, and `non_zero_weight`.
- If all weight parameters are 1.0, the loss **must be identical** to `torch.nn.HuberLoss`.
- The loss is always non-negative.

## 6. Practical Guidance & Parameter Tuning

- **Critical:** The `delta` parameter is highly sensitive to the scale of the pre-processed data that the loss function receives.
- If your data pipeline includes transformations (e.g., `log1p`) followed by a scaler (e.g., `MinMaxScaler` that scales data to `[0, 1]`), the errors seen by the loss function will be in that same `[0, 1]` range.
- In this scenario, the default `delta=0.5` may be too large. A large `delta` will cause most errors to fall into the quadratic (MSE-like) part of the loss, which can lead to instability when combined with high penalty weights.
- **Recommendation:** For data scaled to `[0, 1]`, start with a much smaller `delta` to ensure the robust, linear part of the loss is engaged for meaningful errors. A good starting range to explore for `delta` is **`[0.05, 0.1, 0.25]`**.
