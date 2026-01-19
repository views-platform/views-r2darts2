# Loss Specification Sheet: SpikeFocalLoss v1.0

## 1. Intended Statistical Model & Purpose

- **Name:** `SpikeFocalLoss`
- **Version:** 1.0
- **Purpose:** A custom loss function designed to focus a regression model on "spike" events. It combines Mean Squared Error (MSE) with a focal-style weighting mechanism. The weighting is designed to:
    1.  Upregulate the loss for "spike" targets (those exceeding a `spike_threshold`), especially when the error is large.
    2.  Downregulate the loss for "non-spike" targets, especially when the error is small.
- The focal weighting term is based on `exp(-error)`, which serves as a proxy for the prediction "confidence" used in the original Focal Loss paper.

## 2. Canonical Formula & Code Mapping

The loss is the mean of the element-wise focal-weighted squared error.

`loss = mean( w_focal * e^2 )`

### Formula Components:

| Symbol / Term | Formula | Code Variable | Description |
| :--- | :--- | :--- | :--- |
| `e` | `|preds - targets|` | `errors` | The absolute difference between prediction and target. |
| `is_spike`| `targets > spike_threshold`| `is_spike` | A boolean mask indicating if a target is a spike. |
| `w_focal` | see below | `focal_weights` | The focal weight, calculated differently for spike and non-spike samples. |

### Focal Weighting Formula (`w_focal`):
- If `is_spike` is **True**:
  `w_focal = alpha * (1 - exp(-e))^gamma`
- If `is_spike` is **False**:
  `w_focal = (1 - alpha) * (exp(-e))^gamma`

### Parameters:

| Symbol | Code Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `alpha` | `self.alpha` | 0.8 | A weighting factor to balance the importance between spike and non-spike classes. Similar to the alpha in the original Focal Loss paper. |
| `gamma` | `self.gamma` | 2.0 | The focusing parameter. Higher gamma more aggressively down-weights well-classified examples. |
| `st` | `self.spike_threshold`| 3.0445 | The threshold above which a target is considered a "spike". |

## 3. Domain Constraints

- **`preds`, `targets`**: `torch.Tensor`, real-valued numbers.
- **`alpha`**: `float`, must be in (0, 1).
- **`gamma`, `spike_threshold`**: `float`, must be non-negative.

## 4. Edge Case Policy

- **`e -> 0` (Perfect prediction):**
    - For a non-spike, `w_focal` approaches `(1 - alpha)`, its maximum value for non-spikes.
    - For a spike, `w_focal` approaches `0`.
    - In both cases, the total loss `w_focal * e^2` approaches `0`.
- **`e -> inf` (Very large error):**
    - For a non-spike, `w_focal` approaches `0`.
    - For a spike, `w_focal` approaches `alpha`.
    - The loss grows quadratically, scaled by `alpha` for spikes and suppressed for non-spikes.

## 5. Known Equivalences & Invariants

- The loss is always non-negative.
- The loss is minimized at `preds = targets` (where `e=0`).
- If `gamma = 0`, the focal modulation is disabled.
    - For spikes, `w_focal = alpha`.
    - For non-spikes, `w_focal = 1 - alpha`.
    - The loss becomes a simple weighted MSE, where the weight depends only on whether the target is a spike.

## 6. Practical Guidance & Parameter Tuning

- **CRITICAL: The `spike_threshold` parameter is entirely dependent on the scale of the data it receives after all pre-processing transformations.** A `spike_threshold` that is appropriate for raw count data will be non-functional for data scaled to a `[0, 1]` range.
- **The `is_spike` condition (`targets > spike_threshold`) will never be met if the threshold is set outside the bounds of the transformed data, disabling the core mechanism of this loss function.**
- For detailed recommendations and a matrix of suggested `spike_threshold` ranges for common pipelines, please refer to the central guide:
  - **[Loss Function Pipeline Tuning Guide](../loss_function_tuning_guide.md)**
