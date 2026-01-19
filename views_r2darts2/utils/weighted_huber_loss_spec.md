# Loss Specification Sheet: WeightedHuberLoss v1.0

## 1. Intended Statistical Model & Purpose

- **Name:** `WeightedHuberLoss`
- **Version:** 1.0
- **Purpose:** To provide a robust regression loss (less sensitive to outliers than MSE) that gives higher importance to samples where the ground truth target is non-zero. This is particularly useful for zero-inflated datasets where correctly predicting small but significant events is more important than perfecting the prediction of zero-valued targets.

## 2. Canonical Formula & Code Mapping

The loss is a weighted version of the standard Huber Loss.

### Huber Loss Formula (`L(e)`):
The unweighted Huber loss for a given error `e` is defined as:
`L(e) = 0.5 * e^2` if `|e| <= delta`
`L(e) = delta * (|e| - 0.5 * delta)` if `|e| > delta`

### Weighting Formula (`w`):
The weight `w` for each sample is determined by the target value:
`w = non_zero_weight` if `|targets| > zero_threshold`
`w = 1.0` if `|targets| <= zero_threshold`

### Final Loss Calculation:
`loss = mean( w * L(preds - targets) )`

### Code Mapping:

| Symbol / Term | Formula | Code Variable | Description |
| :--- | :--- | :--- | :--- |
| `e` | `preds - targets` | `errors` | The difference between prediction and target. |
| `delta` | `delta` | `self.delta` | The threshold at which Huber loss changes from quadratic to linear. |
| `w` | `w` | `weights` | The calculated weight for each sample. |
| `L(e)`| Huber Loss | `huber_loss` | The unweighted Huber loss for each sample. |

### Parameters:

| Symbol | Code Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `delta` | `self.delta` | 0.5 | The Huber loss threshold. |
| `zt` | `self.threshold` | 0.01 | (`zero_threshold`) The threshold to determine if a target is non-zero. |
| `nzw` | `self.non_zero_weight`| 5.0 | The multiplicative weight applied to samples with non-zero targets. |

## 3. Domain Constraints

- **`preds`, `targets`**: `torch.Tensor`, should be real-valued numbers.
- **`delta`, `zero_threshold`, `non_zero_weight`**: `float`, must be non-negative.

## 4. Edge Case Policy

- **`|e| = delta` (Error at threshold):** The Huber loss is continuous at this point. Both pieces of the piecewise function yield `0.5 * delta^2`.
- **`|targets| = zero_threshold`:** This is the boundary for weighting. The implementation uses `torch.abs(targets) > self.threshold`, so a target exactly equal to the threshold is considered a "zero" and gets a weight of 1.0.

## 5. Known Equivalences & Invariants

- If `non_zero_weight = 1.0`, the `WeightedHuberLoss` must be mathematically identical to the standard `torch.nn.HuberLoss` with the same `delta`.
- For a fixed error `e`, the loss for a sample with a non-zero target should be exactly `non_zero_weight` times the loss for a sample with a zero target.
- The loss is always non-negative.
- The loss is convex and minimized at `preds = targets`.

## 6. Practical Guidance & Parameter Tuning

- **Critical: `delta` Parameter:** The `delta` parameter is highly sensitive to the scale of the pre-processed data that the loss function receives.
- If your data pipeline includes transformations that scale data to a `[0, 1]` range, the default `delta` values often used (e.g., 0.5, 1.0) are typically too large. This can cause most errors to fall into the quadratic (MSE-like) part of the loss, leading to instability when combined with `non_zero_weight`.
- **Recommendation:** For data scaled to `[0, 1]`, start with a much smaller `delta` to ensure the robust, linear part of the Huber loss is properly engaged for meaningful errors. A good starting range to explore for `delta` is **`[0.05, 0.1, 0.25]`**.
- **`non_zero_weight`:** This parameter controls the importance given to non-zero target events. Experiment with values from `[2.0, 5.0, 10.0]` or higher, depending on the severity of false negatives versus false positives in your application.
- **`zero_threshold`:** This threshold defines what is considered a "zero" target. It should be chosen carefully based on the smallest meaningful non-zero value in your scaled target data. The default `0.01` is often appropriate for `[0, 1]` scaled data.
