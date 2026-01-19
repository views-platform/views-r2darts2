# Loss Specification Sheet: TweedieLoss v1.0

## 1. Intended Statistical Model & Purpose

- **Name:** `TweedieLoss`
- **Version:** 1.0
- **Purpose:** To model zero-inflated, non-negative continuous targets, which are common in conflict data (e.g., battle deaths). The Tweedie distribution with a power parameter `p` between 1 and 2 naturally handles a mix of exact zeros and positive, often skewed, continuous values.
- **Statistical Model:** The loss implements the unit deviance of a Tweedie distribution, which is equivalent to the Poisson deviance when `p->1` and the Gamma deviance when `p=2`. This implementation is for `1 < p < 2`.

## 2. Canonical Formula & Code Mapping

The loss is a weighted version of the Tweedie unit deviance.

`loss = mean( w * D(targets, mu) )`

### Tweedie Unit Deviance (`D(y, mu)`):
The formula for the unit deviance for `1 < p < 2` is:
`D(y, mu) = 2 * ( y^(2-p) / ((1-p)(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p) )`

The implementation uses a common simplified form which is equivalent up to scaling factors and terms that don't depend on the prediction `mu`. The form used is:
`D_simplified(y, mu) = mu^(2-p) / (2-p) - y * mu^(1-p) / (1-p)`

**Note:** The model's raw prediction (`preds`) is mapped to the positive mean `mu` via a `softplus` function to ensure `mu > 0`.
`mu = softplus(preds) + eps`

### Weighting Formula (`w`):
`w = non_zero_weight` if `|targets| > zero_threshold`
`w = 1.0` if `|targets| <= zero_threshold`

### Code Mapping:

| Symbol / Term | Formula | Code Variable | Description |
| :--- | :--- | :--- | :--- |
| `y` | `targets` | `targets` | The ground truth target values. |
| `mu`| `softplus(preds)+eps`| `preds_pos` | The predicted positive mean of the distribution. |
| `p` | `p` | `self.p` | The power parameter of the Tweedie distribution. |
| `w` | `w` | `weights` | The weight for each sample based on target magnitude. |

### Parameters:

| Symbol | Code Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `p` | `self.p` | 1.5 | Power parameter, must be in (1, 2). |
| `nzw` | `self.non_zero_weight`| 5.0 | The weight for non-zero targets. |
| `zt` | `self.threshold` | 0.01 | (`zero_threshold`) Threshold for considering a target non-zero. |
| `eps`| `self.eps` | 1e-8 | Small constant for numerical stability after `softplus`. |

## 3. Domain Constraints

- **`targets`**: Must be non-negative (`>= 0`).
- **`preds`**: Real-valued. The `softplus` function ensures the mean `mu` is positive.
- **`p`**: Must be strictly between 1 and 2. The constructor enforces this.
- All other parameters must be non-negative.

## 4. Edge Case Policy

- **`targets = 0`:** The deviance simplifies to `mu^(2-p) / (2-p)`. The loss is finite and well-defined.
- **`preds -> -inf`:** `softplus(preds)` approaches 0. `mu` approaches `eps`. The loss will be large but finite, dominated by the `mu^(1-p)` term if `y > 0`.

## 5. Known Equivalences & Invariants

- If `non_zero_weight = 1.0`, the weighting is disabled.
- The loss is always non-negative for valid inputs.
- For a fixed target `y > 0`, the loss should be minimized when the predicted mean `mu` is equal to `y`.
- The `softplus` function makes the loss a function of `preds` rather than `mu` directly, which is a standard technique for constraining the output of a neural network to be positive.

## 6. Practical Guidance & Parameter Tuning

- **`p` Parameter:** The `p` parameter (power) is the key to this loss, defining the distribution's shape. It must be in `(1, 2)`. Values closer to 1 are more Poisson-like, while values closer to 2 are more Gamma-like. `p=1.5` is a common starting point. This parameter is scale-invariant.
- **`non_zero_weight` Parameter:** Since the loss already handles zeros statistically, this weight should be tuned with care. It is recommended to test `1.0` (no additional weight) as a baseline.
- For more detailed guidance, see the central guide:
  - **[Loss Function Pipeline Tuning Guide](../loss_function_tuning_guide.md)**
