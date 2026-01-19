# Guide: Pipeline Robustness & Hyperparameter Tuning for Loss Functions

This guide provides a comprehensive overview of how to tune scale-dependent loss function hyperparameters for different data pre-processing pipelines. The recommendations below are validated by the test suite in `tests/test_loss_pipeline_robustness.py`.

## 1. The Challenge: Scale-Dependent Parameters

Several custom loss functions in this repository contain hyperparameters that are highly sensitive to the numerical range (scale) of the data they receive. These include `delta` in Huber-based losses, `c` in `ShrinkageLoss`, and `spike_threshold` in `SpikeFocalLoss`.

A parameter value that is optimal for data scaled to `[0, 1]` will be ineffective or cause numerical instability for raw count data, and vice-versa. **It is critical to tune these parameters based on the specific data transformation pipeline being used.**

## 2. Common Data Transformation Pipelines

The following table defines common pipelines and the typical data range they produce.

| ID | Pipeline | Output Range | Description |
|:---|:---|:---|:---|
| A | `log1p` + `MinMax(0,1)` | `[0, 1]` | Project default. Log-transforms then scales to `[0, 1]`. |
| B | `raw_counts` | `[0, N]` | No transformation. Data remains as integer/float counts. |
| C | `asinh` + `StandardScaler`| `~[-3, 3]` | Log-like transform robust to zeros, then standardized to mean=0, std=1. |
| D | `log1p` + `MinMax(-1,1)`| `[-1, 1]` | Log-transforms then scales to `[-1, 1]`. |
| E | `pure_log1p` | `[0, ~8]` | Pure log-transform. Output scale depends on max raw value. |
| F | `pure_asinh` | `[0, ~8]` | Pure `arcsinh` transform. Similar to log. |
| G | `pure_minmax(0,1)` | `[0, 1]` | Direct scaling without log-transform. Skewness is preserved. |
| H | `pure_minmax(-1,1)` | `[-1, 1]` | Direct scaling to `[-1, 1]`. Skewness is preserved. |

## 3. Tuning Recommendations for Huber-Based Losses

This applies to **`WeightedHuberLoss`, `TimeAwareWeightedHuberLoss`, `WeightedPenaltyHuberLoss`,** and the Pseudo-Huber component of **`ZeroInflatedLoss`**.

| Pipeline ID | Recommended `delta` Range | Default `delta=0.5` Behavior |
| :--- | :--- | :--- |
| A, G | `[0.05, 0.25]` | **Likely Unstable.** Too large; acts like MSE. |
| B | `5-10% of max count` | **Ineffective.** Too small; treats all errors as large. |
| C | `[0.5, 1.5]` | **Good Default.** A reasonable starting point. |
| D, H | `[0.1, 0.4]` | **Likely Unstable.** Too large; acts like MSE. |
| E, F | `[0.2, 1.0]` | **Good Default.** A reasonable starting point. |

## 4. Tuning Recommendations for Threshold-Based Losses

### `SpikeFocalLoss` (`spike_threshold`)

The `spike_threshold` must be set to a value **within** the bounds of the transformed data. It's recommended to set it to a high quantile of the target distribution.

| Pipeline ID | Data Range | Recommended `spike_threshold` Range |
| :--- | :--- | :--- |
| A, G | `[0, 1]` | `[0.75, 0.95]` |
| B | `[0, N]` | `[90th, 99th]` percentile of raw counts. |
| C | `~[-3, 3]` | `[1.5, 2.5]` |
| D, H | `[-1, 1]` | `[0.7, 0.95]` |
| E, F | `[0, ~8]` | `[90th, 99th]` percentile of transformed values. |

### `ShrinkageLoss` (`c`)

The `c` parameter is an **error threshold**. It should be set relative to the expected distribution of errors (`|preds - targets|`).

| Pipeline ID | Data Range | Recommended `c` Range |
| :--- | :--- | :--- |
| A, D, G, H | `[0, 1]` or `[-1, 1]`| `[0.05, 0.3]` |
| B | `[0, N]` | `5-10% of N` |
| C, E, F | `~[-3, 3]` | `[0.2, 1.0]` |

The `a` parameter (shrinkage speed) should be tuned via sweeps, with `[1.0, 5.0, 15.0]` being a good starting range.

