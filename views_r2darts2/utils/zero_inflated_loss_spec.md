# Loss Specification Sheet: ZeroInflatedLoss v1.0

## 1. Intended Statistical Model & Purpose

- **Name:** `ZeroInflatedLoss`
- **Version:** 1.0
- **Purpose:** To model zero-inflated (or "hurdle") processes, where the data generation has two stages:
    1. A binary outcome of whether an event occurs (a zero vs. non-zero value).
    2. A continuous outcome for the magnitude of the event, conditional on it having occurred.
- This is a good fit for conflict data where many time steps have zero events, and we want to model both the probability of conflict and its intensity separately.

## 2. Canonical Formula & Code Mapping

The loss is a weighted sum of two distinct loss components: a binary classification loss and a regression loss.

`loss = zero_weight * L_zero + count_weight * L_count`

### 1. Zero-Inflation Component (`L_zero`)

- **Purpose:** To classify whether the target is zero or non-zero.
- **Method:** This component uses Binary Cross-Entropy (BCE).
- **Formula:**
    1.  First, the raw prediction `preds` is transformed into a probability of the target being zero, `p_zero`.
        `p_zero = sigmoid(-10 * preds)`
        *This implies that a large positive prediction corresponds to a low probability of the target being zero.*
    2.  The binary ground truth `is_zero` is determined.
        `is_zero = 1` if `|targets| < zero_threshold`, else `0`.
    3.  The loss is the standard BCE loss.
        `L_zero = BCE(p_zero, is_zero) = -[is_zero * log(p_zero) + (1 - is_zero) * log(1 - p_zero)]`

### 2. Count/Magnitude Component (`L_count`)

- **Purpose:** To measure the regression error for non-zero targets.
- **Method:** This component uses a Pseudo-Huber loss, which is a smooth approximation of the standard Huber loss. The loss is only calculated for non-zero targets.
- **Formula:**
    1.  A mask `count_mask` is created for non-zero targets.
        `count_mask = 1 - is_zero`
    2.  The error `e` is calculated only for these targets.
        `e = (preds - targets) * count_mask`
    3.  The Pseudo-Huber loss is applied.
        `L_ph(e) = delta^2 * (sqrt(1 + (e/delta)^2) - 1)`
    4.  The final component is the mean of this loss.
        `L_count = mean(L_ph(e))`

### Code Mapping & Parameters:

| Term | Code Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `zw` | `self.zero_weight` | 1.0 | The weight for the binary (`L_zero`) component. |
| `cw` | `self.count_weight` | 1.0 | The weight for the regression (`L_count`) component. |
| `delta`| `self.delta` | 0.5 | The delta parameter for the Pseudo-Huber loss. |
| `zt` | `self.threshold`| 0.01 | (`zero_threshold`) Threshold for considering a value zero. |
| `eps`| `self.eps` | 1e-8 | Epsilon for clamping probabilities in BCE to avoid `log(0)`. |

## 3. Domain Constraints

- **`preds`, `targets`**: `torch.Tensor`, real-valued.
- All weight and delta parameters must be non-negative.

## 4. Edge Case Policy

- **`targets = 0`:** The `count_mask` is zero, so `L_count` is zero. The loss is determined entirely by the `L_zero` component.
- **`preds` is large and positive:** `p_zero` approaches 0. If the target is non-zero (`is_zero=0`), `L_zero` approaches 0. If the target is zero (`is_zero=1`), `L_zero` approaches infinity. This correctly penalizes confident misclassifications of zero-valued targets.
- **`preds` is large and negative:** `p_zero` approaches 1. If the target is zero (`is_zero=1`), `L_zero` approaches 0. If the target is non-zero (`is_zero=0`), `L_zero` approaches infinity. This correctly penalizes confident misclassifications of non-zero-valued targets.

## 5. Known Equivalences & Invariants

- If `count_weight = 0`, the total loss is determined only by the binary classification (zero vs. non-zero) component.
- If `zero_weight = 0`, the total loss is determined only by the Pseudo-Huber regression error on the non-zero targets.
- The loss is always non-negative.

## 6. Practical Guidance & Parameter Tuning

- **`delta` Parameter:** Similar to other Huber-type losses, the `delta` parameter of the Pseudo-Huber component is highly sensitive to the scale of the pre-processed data. If your data is scaled to `[0, 1]`, the default `delta=0.5` is likely too large, causing most errors to be treated quadratically (MSE-like) instead of linearly (MAE-like).
    - **Recommendation:** For `[0, 1]` scaled data, explore a smaller range for `delta`, such as **`[0.05, 0.1, 0.25]`**.
- **`zero_weight` and `count_weight`:** These parameters control the balance between the zero-inflation (classification) and count (regression) components. Given the often high zero-inflation in conflict data, careful tuning is required to prevent one component from dominating the other.
    - **Recommendation:** Explore a wider range for both `zero_weight` and `count_weight`, such as **`[0.5, 1.0, 2.0, 5.0]`**, to find the optimal balance for your dataset.
- **Hardcoded Sigmoid Multiplier (`sigmoid(-10 * preds)`):** The `10x` multiplier within the `sigmoid` function for `p_zero` makes the zero-component very sensitive to small changes in `preds`. This aggressive scaling means that even small differences in raw predictions can lead to large changes in the predicted probability of being zero. While effective for steep transitions, it can make initial training unstable or difficult to optimize.
    - **Consideration:** If the loss proves highly unstable, making this multiplier a configurable hyperparameter might be beneficial, but it would require modifying the loss function implementation itself. For now, be aware of its aggressive nature.
