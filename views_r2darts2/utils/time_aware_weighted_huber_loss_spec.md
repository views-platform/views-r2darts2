# Loss Specification Sheet: TimeAwareWeightedHuberLoss v1.0

## 1. Intended Statistical Model & Purpose

- **Name:** `TimeAwareWeightedHuberLoss`
- **Version:** 1.0
- **Purpose:** This loss function extends the `WeightedHuberLoss` by adding a temporal dimension. It is designed for time-series or sequence data where:
    1. Non-zero events are more important than zero-valued events.
    2. More recent events are more important than older events.
- It combines a standard Huber loss with both event-based weights and a time-decay weighting factor.

## 2. Canonical Formula & Code Mapping

The final loss is the mean of the element-wise weighted Huber loss.

`loss = mean( w_event * w_time * L(preds - targets) )`

### Huber Loss Formula (`L(e)`):
The unweighted Huber loss for a given error `e` is defined as:
`L(e) = 0.5 * e^2` if `|e| <= delta`
`L(e) = delta * (|e| - 0.5 * delta)` if `|e| > delta`

### Event Weighting Formula (`w_event`):
`w_event = non_zero_weight` if `|targets| > 1e-4`
`w_event = zero_weight` if `|targets| <= 1e-4`

### Temporal Weighting Formula (`w_time`):
For a sequence of length `seq_len`, the weight at time step `i` (from 0 to `seq_len-1`) is:
`w_time[i] = decay_factor ^ (seq_len - 1 - i)`
*Note: The implementation uses this formulation, which correctly gives the highest weight (`decay_factor^0 = 1`) to the most recent event in the sequence (`i = seq_len-1`) and the lowest weight to the oldest event (`i = 0`).*

### Code Mapping:

| Symbol / Term | Formula | Code Variable | Description |
| :--- | :--- | :--- | :--- |
| `e` | `preds - targets` | `errors` (implicit) | The difference between prediction and target. |
| `delta` | `delta` | `self.delta` | The Huber loss threshold. |
| `w_event`| Event Weight | `event_weights` | The weight based on target magnitude. |
| `w_time` | Temporal Weight | `time_weights` | The weight based on position in the sequence. |
| `L(e)`| Huber Loss | `losses` | The unweighted Huber loss for each sample. |

### Parameters:

| Symbol | Code Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `zw` | `self.zero_weight` | N/A | The weight for zero-valued targets. |
| `nzw`| `self.non_zero_weight`| N/A | The weight for non-zero targets. |
| `df` | `self.decay_factor` | N/A | The base for the exponential time decay. Should be in (0, 1]. |
| `delta`| `self.delta` | N/A | The Huber loss threshold. |

## 3. Domain Constraints

- **`preds`, `targets`**: `torch.Tensor` with at least 2 dimensions (batch, sequence_length).
- **`delta`, `zero_weight`, `non_zero_weight`**: `float`, must be non-negative.
- **`decay_factor`**: `float`, must be in (0, 1].

## 4. Edge Case Policy

- The underlying Huber loss properties at its `delta` threshold apply.
- If `decay_factor = 1.0`, time weighting is disabled (all time weights are 1.0).

## 5. Known Equivalences & Invariants

- If `decay_factor = 1.0`, `zero_weight = 1.0`, and `non_zero_weight = 1.0`, the loss must be mathematically identical to the standard `torch.nn.HuberLoss` with the same `delta`.
- If `decay_factor = 1.0`, the loss should be equivalent to a simple weighted Huber loss where the weights are `zero_weight` and `non_zero_weight`.
- If `zero_weight` and `non_zero_weight` are equal, the event-based weighting has no differential effect.
- The loss is always non-negative.

## 6. Practical Guidance & Parameter Tuning

- **Critical:** The `delta` parameter is highly sensitive to the scale of the pre-processed data that the loss function receives.
- If your data pipeline includes transformations that scale data to a `[0, 1]` range, the default `delta` values used in some sweeps (e.g., 0.5, 1.0) may be too large. This can cause the loss to behave like a weighted MSE instead of a robust Huber loss, leading to instability.
- **Recommendation:** For data scaled to `[0, 1]`, start with a much smaller `delta` to ensure the robust, linear part of the loss is engaged for meaningful errors. A good starting range to explore for `delta` is **`[0.05, 0.1, 0.25]`**.
- The `decay_factor` controls how much to penalize older errors. A value close to 1.0 (e.g., 0.99) results in very little decay, while a smaller value (e.g., 0.8) decays older errors more aggressively.
