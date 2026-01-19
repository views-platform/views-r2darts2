# Loss Specification Sheet: AsymmetricQuantileLoss v1.0

## 1. Intended Statistical Model & Purpose

- **Name:** `AsymmetricQuantileLoss`, also known as Pinball Loss.
- **Version:** 1.0
- **Purpose:** To perform quantile regression. Unlike MSE or MAE which target the mean and median respectively, quantile regression allows for estimating arbitrary quantiles of the target distribution. This is useful for generating prediction intervals and for creating models where the cost of underestimation and overestimation is asymmetric.
- For conflict forecasting, using a high `tau` (e.g., 0.75, 0.9) penalizes underestimation (missing a conflict) more heavily than overestimation (predicting a conflict that doesn't occur).

## 2. Canonical Formula & Code Mapping

The loss is a weighted version of the standard quantile loss.

`loss = mean( w * L_quantile(preds - targets) )`

### Quantile Loss Formula (`L_quantile(e)`):
The loss for a given error `e = targets - preds` and a quantile `tau` is:
`L_quantile(e) = tau * e` if `e >= 0` (Underestimation)
`L_quantile(e) = (1 - tau) * -e` if `e < 0` (Overestimation)
*This can be written more compactly as `max(tau * e, (tau - 1) * e)`.*

The implementation uses `(tau - 1) * e` for the overestimation case, which is equivalent since `e` and `tau-1` are both negative.

### Weighting Formula (`w`):
`w = non_zero_weight` if `|targets| > zero_threshold`
`w = 1.0` if `|targets| <= zero_threshold`

### Code Mapping:

| Symbol / Term | Formula | Code Variable | Description |
| :--- | :--- | :--- | :--- |
| `e` | `targets - preds` | `errors` | The difference between target and prediction. |
| `tau`| `tau` | `self.tau` | The quantile to be estimated, in (0, 1). |
| `w` | `w` | `weights` | The weight for each sample based on target magnitude. |

### Parameters:

| Symbol | Code Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `tau`| `self.tau` | 0.75 | The quantile level. |
| `nzw`| `self.non_zero_weight`| 5.0 | The weight for non-zero targets. |
| `zt` | `self.threshold` | 0.01 | (`zero_threshold`) Threshold for considering a target non-zero. |

## 3. Domain Constraints

- **`preds`, `targets`**: `torch.Tensor`, real-valued numbers.
- **`tau`**: `float`, must be strictly between 0 and 1. The constructor enforces this.
- All other parameters must be non-negative.

## 4. Edge Case Policy

- **`e = 0` (Perfect prediction):** The loss is 0.
- The gradient is discontinuous at `e = 0`.

## 5. Known Equivalences & Invariants

- If `tau = 0.5`, the loss is equivalent to `0.5 * Mean Absolute Error (MAE)`.
- The loss is always non-negative.
- The loss is minimized when the predictions represent the `tau`-th quantile of the target distribution.

## 6. Practical Guidance & Parameter Tuning

- **`tau` Parameter:** The `tau` parameter (must be between 0 and 1) directly controls the asymmetry of the penalties for underestimation versus overestimation.
    - If `tau > 0.5`, underestimation is penalized more heavily than overestimation (e.g., `tau=0.75` means underestimation is penalized 3x more than overestimation). This is typically desired in conflict forecasting.
    - If `tau < 0.5`, overestimation is penalized more heavily.
    - If `tau = 0.5`, the loss becomes symmetric, equivalent to `0.5 * MAE`.
- **`non_zero_weight` Parameter:** This parameter applies an additional weighting to non-zero targets. While useful for some losses, quantile loss inherently handles asymmetric costs through `tau`. Adding `non_zero_weight` might be redundant or could interfere with the desired statistical properties.
    - **Recommendation:** When tuning, it is highly recommended to include `1.0` in the search space for `non_zero_weight`. This allows the sweep to test the "pure" quantile loss against versions with additional artificial weighting on the non-zero targets.
