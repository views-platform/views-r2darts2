# Loss Specification Sheet: ShrinkageLoss v1.0

## 1. Intended Statistical Model & Purpose

- **Name:** `ShrinkageLoss`
- **Version:** 1.0
- **Source:** Inspired by "Deep Regression Tracking with Shrinkage Loss" by Lu et al. (2018).
- **Purpose:** To improve regression performance on datasets with imbalanced error magnitudes, particularly zero-inflated data. The loss "shrinks" the contribution of easy samples (small errors), forcing the model to focus on hard samples (large errors).
- **Customizations:** This implementation includes a custom `importance_weight` term (`exp(targets)`) that is not part of the original paper's formula. This weight is intended to give more importance to samples with larger target values, assuming the targets are log1p-transformed (`targets = log(1 + y_original)`).

## 2. Canonical Formula & Code Mapping

The loss is calculated as:
`loss = mean( (importance_weight * l^2) / shrinkage_factor )`

### Formula Components:

| Symbol | Formula | Code Variable | Description |
| :--- | :--- | :--- | :--- |
| `l` | `|preds - targets|` | `l` | The absolute error (L1 distance). |
| `shrinkage_factor` | `1 + exp(a * (c - l))` | `shrinkage_factor` | The core mechanism. It is a large value for small `l` (easy samples) and approaches 1 for large `l` (hard samples). |
| `importance_weight`| `exp(targets)` | `importance_weight` | A custom, non-standard weight. If `targets` are `log(1+y)`, this equals `1+y`, applying a linear weight based on the original target magnitude. |
| `base_loss` | `importance_weight * l^2` | `base_loss` | The weighted squared error before shrinkage is applied. |

### Parameters:

| Symbol | Code Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `a` | `self.a` | 10.0 | Controls the rate of shrinkage. Higher `a` means faster shrinkage for easy samples. |
| `c` | `self.c` | 0.2 | The error threshold. Errors below `c` are considered "easy" and are shrunk more aggressively. |

## 3. Domain Constraints

- **`preds`, `targets`**: `torch.Tensor`, should be real-valued numbers.
- **`a`, `c`**: `float`, must be positive.

## 4. Edge Case Policy

- **`l = c` (Error at threshold):** The `shrinkage_factor` is `1 + exp(0) = 2`. The loss is `base_loss / 2`.
- **`l -> 0` (Perfect prediction):** `shrinkage_factor` approaches its maximum value `1 + exp(a*c)`, maximally shrinking the loss. The `base_loss` approaches 0.
- **`l -> inf` (Very large error):** `shrinkage_factor` approaches 1. The loss behaves like a simple weighted squared error (`importance_weight * l^2`).

## 5. Known Equivalences & Invariants

- If `a = 0`, the `shrinkage_factor` is always 2, and the loss is equivalent to `0.5 * importance_weight * l^2`.
- For a fixed error `l`, the loss value increases as the `targets` value increases, due to the `importance_weight` term.
- The loss is always non-negative.
- The loss is minimized at `preds = targets`.

## 6. Practical Guidance & Parameter Tuning

- **CRITICAL: The `c` (error threshold) parameter is highly sensitive to the scale of the errors (`|preds-targets|`) produced by your specific data pipeline.** An effective `c` should be set relative to the distribution of errors you expect.
- **The `a` (shrinkage speed) parameter controls the aggressiveness of the shrinkage. A very high `a` can cause vanishing gradients for easy samples.**
- For detailed recommendations and a matrix of suggested `c` ranges for common pipelines, please refer to the central guide:
  - **[Loss Function Pipeline Tuning Guide](../loss_function_tuning_guide.md)**
