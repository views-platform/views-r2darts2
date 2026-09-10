# Loss Specification Sheet: TweedieLoss v2.0

## 1. Intended Statistical Model & Purpose

- **Name:** `TweedieLoss`
- **Version:** 2.0
- **Purpose:** To serve as a statistically robust loss function for zero-inflated, non-negative continuous targets, such as conflict fatalities. The loss is the negative log-likelihood (NLL) of the Tweedie distribution.
- **Statistical Model:** By minimizing the Tweedie NLL, the model is trained to predict the conditional mean `μ` of a Tweedie distribution, `Y ~ Tweedie(μ, φ, p)`. For `1 < p < 2`, this distribution naturally models a mixture of exact zeros and continuous, right-skewed positive values.

## 2. Canonical Formula & Code Mapping

The loss function implements the Tweedie Negative Log-Likelihood for `1 < p < 2`. The model's raw output is treated as the linear predictor `eta`.

For numerical stability, `eta` is mapped to the mean `μ` via the `softplus` link function: `μ = softplus(eta)`. While the canonical link for the Tweedie GLM is the log link (`μ = exp(eta)`), `softplus` is used as a more robust alternative that prevents the exploding gradients that can occur with `exp` in a deep learning context.

The loss, up to constants that do not depend on `μ`, is:
`L(y, μ) = (μ**(2-p) / (2-p)) - (y * μ**(1-p) / (1-p))`

> **What `forward()` actually returns (2026-09-10):** the NLL above multiplied by a per-sample weight built from `non_zero_weight` (target-dependent, via `zero_threshold`) and the FP/FN penalty genes (prediction-dependent). The *weighted* loss is not the canonical Tweedie NLL and the "proper scoring rule" property below holds for the unweighted core only. Also: `forward()` silently clamps negative targets to 0 (`views_r2darts2/math/tweedie_loss.py:67`) rather than rejecting them.

### Critical Property
Minimizing the expected value of this loss function with respect to `μ` recovers the true conditional mean, `arg min E[L(y, μ)] => μ = E[Y]`. This property makes the loss function ideal for tasks where mean calibration is a primary objective.

### Parameters:

| Symbol | Code Variable | Typical value (must be declared in DNA) | Description |
| :--- | :--- | :--- | :--- |
| `p` | `self.p` | 1.5 | The power parameter of the Tweedie distribution, controlling the variance structure `Var(Y) = φμ^p`. Must be in the interval `(1, 2)`. |
| `eps` | `self.eps` | 1e-6 | A small positive constant added to `μ` to ensure numerical stability (`mu = softplus(eta) + eps`). |
| `non_zero_weight` | `self.non_zero_weight` | — | Base weight multiplier applied to non-zero targets. Mandatory gene. |
| `zero_threshold` | `self.zero_threshold` | — | Threshold below which a target counts as zero for weighting. Mandatory gene. |
| `false_positive_weight` | `self.false_positive_weight` | — | Extra weight when the target is zero and the prediction is not. Mandatory gene. |
| `false_negative_weight` | `self.false_negative_weight` | — | Extra weight when the target is non-zero and the prediction is below threshold. Mandatory gene. |

*All six are in `LOSS_GENOMES["TweedieLoss"]`; the constructor has no defaults (ADR-003).*

## 3. Domain Constraints

- **`preds` (`eta`)**: `torch.Tensor`, should be real-valued numbers.
- **`targets` (`y`)**: `torch.Tensor`, must be non-negative (`y >= 0`).
- **`p`**: `float`, must be strictly between 1 and 2.

## 4. Edge Case Policy

- **`eta -> -inf`:** If the raw model output `eta` is a large negative number, `softplus(eta)` will approach zero. A small `eps` is added to the result to ensure `μ` is strictly positive, preventing the terms `μ**(1-p)` and `μ**(2-p)` from resulting in division by zero.

## 5. Known Equivalences & Invariants

- As `p -> 1`, the Tweedie distribution approaches a scaled Poisson distribution.
- As `p -> 2`, the Tweedie distribution approaches a Gamma distribution.
- The unweighted core is a proper scoring rule, meaning it is uniquely minimized in expectation when the predicted mean equals the true mean. The weighting applied in `forward()` breaks this guarantee by design (it is the point of the weights).
- The dispersion `φ` is treated as a constant nuisance parameter during training and can be fixed to 1 without affecting the optimization of model weights.

## 6. Practical Guidance & Parameter Tuning

- **The power parameter `p` is a critical hyperparameter.** It should be tuned based on the characteristics of the data. A common method is to perform a grid search over a range of `p` values (e.g., `[1.2, 1.5, 1.8]`) and select the value that yields the best performance on a validation set.
- The raw output of the neural network should be used directly as the `preds` (`eta`) input to this loss function. No final activation function (like `ReLU`) is required on the model's output layer, as the loss function itself applies the `softplus` link function.
