# Sweep Configurations Directory Description

This directory contains `wandb` hyperparameter sweep configurations. These configurations are designed to systematically explore the parameter space of various loss functions when applied to different data transformation pipelines.

## Purpose

The primary goal of these sweep configurations is to facilitate hyperparameter optimization for forecasting models, specifically focusing on how custom loss functions behave under diverse data pre-processing scenarios. Each sweep configuration is tailored to a specific loss function and a particular data pipeline to ensure effective and stable model training.

## Data Transformation Pipelines

The configurations are categorized by five representative data transformation pipelines, each with distinct scaling characteristics:

*   **Pipeline A (`_a_sweep.py`): `log1p` + `MinMax(0,1)`**
    *   **Description:** The project's current standard. Data is transformed using `log1p` (natural logarithm of 1 + input) and then scaled to the `[0, 1]` range using `MinMaxScaler`.
    *   **Model Config:** `log_targets: True`, `target_scaler: MinMaxScaler`.

*   **Pipeline B (`_b_sweep.py`): `raw_counts`**
    *   **Description:** No transformation or scaling is applied to the target variable. Data remains as raw integer or float counts, typically with a large and variable range.
    *   **Model Config:** `log_targets: False`, `target_scaler: None`.

*   **Pipeline C (`_c_sweep.py`): `asinh` + `StandardScaler`**
    *   **Description:** The `arcsinh` (inverse hyperbolic sine) transformation is applied (a robust log-like transform), followed by `StandardScaler` to produce data with a mean of 0 and a standard deviation of 1.
    *   **Model Config:** `log_targets: False`, `target_scaler: StandardScaler`.

*   **Pipeline E (`_e_sweep.py`): `pure_log1p`**
    *   **Description:** Only the `log1p` transformation is applied. No additional scaling. The output range depends on the max raw value.
    *   **Model Config:** `log_targets: True`, `target_scaler: None`.

*   **Pipeline F (`_f_sweep.py`): `pure_asinh`**
    *   **Description:** Only the `arcsinh` transformation is applied. No additional scaling. The output range depends on the max raw value.
    *   **Model Config:** `log_targets: False`, `target_scaler: None`.

## Catalog of Sweep Configurations

The following table provides a catalog of the sweep configuration files, detailing the loss function, the pipeline it targets, and the key hyperparameters being explored.

| File Name | Loss Function | Pipeline | Key Loss Hyperparameters Swept | Target Scaler | Log Targets |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `asymmetric_quantile_a_sweep.py` | `AsymmetricQuantileLoss` | A | `tau: [0.75, 0.85, 0.95]`, `non_zero_weight: [1.0, 5.0, 10.0]` | `MinMaxScaler` | `True` |
| `asymmetric_quantile_b_sweep.py` | `AsymmetricQuantileLoss` | B | `tau: [0.75, 0.85, 0.95]`, `non_zero_weight: [1.0, 5.0, 10.0]` | `None` | `False` |
| `asymmetric_quantile_c_sweep.py` | `AsymmetricQuantileLoss` | C | `tau: [0.75, 0.85, 0.95]`, `non_zero_weight: [1.0, 5.0, 10.0]` | `StandardScaler` | `False` |
| `asymmetric_quantile_e_sweep.py` | `AsymmetricQuantileLoss` | E | `tau: [0.75, 0.85, 0.95]`, `non_zero_weight: [1.0, 5.0, 10.0]` | `None` | `True` |
| `asymmetric_quantile_f_sweep.py` | `AsymmetricQuantileLoss` | F | `tau: [0.75, 0.85, 0.95]`, `non_zero_weight: [1.0, 5.0, 10.0]` | `None` | `False` |
| `shrinkage_a_sweep.py` | `ShrinkageLoss` | A | `a: [1.0, 5.0, 15.0, 30.0]`, `c: [0.05, 0.15, 0.3, 0.6]` | `MinMaxScaler` | `True` |
| `shrinkage_b_sweep.py` | `ShrinkageLoss` | B | `a: [1.0, 5.0, 15.0, 30.0]`, `c: [50.0, 150.0, 300.0]` | `None` | `False` |
| `shrinkage_c_sweep.py` | `ShrinkageLoss` | C | `a: [1.0, 5.0, 15.0, 30.0]`, `c: [0.2, 1.0, 2.0]` | `StandardScaler` | `False` |
| `shrinkage_e_sweep.py` | `ShrinkageLoss` | E | `a: [1.0, 5.0, 15.0, 30.0]`, `c: [0.2, 1.0, 2.0]` | `None` | `True` |
| `shrinkage_f_sweep.py` | `ShrinkageLoss` | F | `a: [1.0, 5.0, 15.0, 30.0]`, `c: [0.2, 1.0, 2.0]` | `None` | `False` |
| `spikefocal_a_sweep.py` | `SpikeFocalLoss` | A | `alpha: [0.7, 0.8, 0.9]`, `gamma: [1.5, 2.0]`, `spike_threshold: [0.75, 0.9, 0.95]` | `MinMaxScaler` | `True` |
| `spikefocal_b_sweep.py` | `SpikeFocalLoss` | B | `alpha: [0.7, 0.8, 0.9]`, `gamma: [1.5, 2.0]`, `spike_threshold: [400.0, 450.0, 480.0]` | `None` | `False` |
| `spikefocal_c_sweep.py` | `SpikeFocalLoss` | C | `alpha: [0.7, 0.8, 0.9]`, `gamma: [1.5, 2.0]`, `spike_threshold: [1.5, 2.0, 2.5]` | `StandardScaler` | `False` |
| `spikefocal_e_sweep.py` | `SpikeFocalLoss` | E | `alpha: [0.7, 0.8, 0.9]`, `gamma: [1.5, 2.0]`, `spike_threshold: [1.5, 2.0, 2.5]` | `None` | `True` |
| `spikefocal_f_sweep.py` | `SpikeFocalLoss` | F | `alpha: [0.7, 0.8, 0.9]`, `gamma: [1.5, 2.0]`, `spike_threshold: [1.5, 2.0, 2.5]` | `None` | `False` |
| `timeaware_weighted_huber_a_sweep.py` | `TimeAwareWeightedHuberLoss` | A | `delta: [0.05, 0.1, 0.25]`, `decay_factor: [0.9, 0.95]`, `zero_weight: [1.0]`, `non_zero_weight: [5.0, 10.0, 20.0]` | `MinMaxScaler` | `True` |
| `timeaware_weighted_huber_b_sweep.py` | `TimeAwareWeightedHuberLoss` | B | `delta: [50.0, 100.0, 200.0]`, `decay_factor: [0.9, 0.95]`, `zero_weight: [1.0]`, `non_zero_weight: [5.0, 10.0, 20.0]` | `None` | `False` |
| `timeaware_weighted_huber_c_sweep.py` | `TimeAwareWeightedHuberLoss` | C | `delta: [0.5, 1.5]`, `decay_factor: [0.9, 0.95]`, `zero_weight: [1.0]`, `non_zero_weight: [5.0, 10.0, 20.0]` | `StandardScaler` | `False` |
| `timeaware_weighted_huber_e_sweep.py` | `TimeAwareWeightedHuberLoss` | E | `delta: [0.2, 0.5, 1.0]`, `decay_factor: [0.9, 0.95]`, `zero_weight: [1.0]`, `non_zero_weight: [5.0, 10.0, 20.0]` | `None` | `True` |
| `timeaware_weighted_huber_f_sweep.py` | `TimeAwareWeightedHuberLoss` | F | `delta: [0.2, 0.5, 1.0]`, `decay_factor: [0.9, 0.95]`, `zero_weight: [1.0]`, `non_zero_weight: [5.0, 10.0, 20.0]` | `None` | `False` |
| `tweedie_a_sweep.py` | `TweedieLoss` | A | `p: [1.2, 1.5, 1.8]` | `MinMaxScaler` | `True` |
| `tweedie_b_sweep.py` | `TweedieLoss` | B | `p: [1.2, 1.5, 1.8]` | `None` | `False` |
| `tweedie_c_sweep.py` | `TweedieLoss` | C | `p: [1.2, 1.5, 1.8]` | `StandardScaler` | `False` |
| `tweedie_e_sweep.py` | `TweedieLoss` | E | `p: [1.2, 1.5, 1.8]` | `None` | `True` |
| `tweedie_f_sweep.py` | `TweedieLoss` | F | `p: [1.2, 1.5, 1.8]` | `None` | `False` |
| `weightedhuber_a_sweep.py` | `WeightedHuberLoss` | A | `delta: [0.05, 0.1, 0.25]`, `non_zero_weight: [2.0, 5.0, 10.0]`, `zero_threshold: [0.01]` | `MinMaxScaler` | `True` |
| `weightedhuber_b_sweep.py` | `WeightedHuberLoss` | B | `delta: [50.0, 100.0, 200.0]`, `non_zero_weight: [2.0, 5.0, 10.0]`, `zero_threshold: [0.1, 0.5, 1.0]` | `None` | `False` |
| `weightedhuber_c_sweep.py` | `WeightedHuberLoss` | C | `delta: [0.5, 1.5]`, `non_zero_weight: [2.0, 5.0, 10.0]`, `zero_threshold: [0.001]` | `StandardScaler` | `False` |
| `weightedhuber_e_sweep.py` | `WeightedHuberLoss` | E | `delta: [0.2, 0.5, 1.0]`, `non_zero_weight: [2.0, 5.0, 10.0]`, `zero_threshold: [0.01, 0.1]` | `None` | `True` |
| `weightedhuber_f_sweep.py` | `WeightedHuberLoss` | F | `delta: [0.2, 0.5, 1.0]`, `non_zero_weight: [2.0, 5.0, 10.0]`, `zero_threshold: [0.01, 0.1]` | `None` | `False` |
| `weighted_penalty_huber_a_sweep.py` | `WeightedPenaltyHuberLoss` | A | `delta: [0.05, 0.1, 0.25]`, `zero_threshold: [0.01]`, `non_zero_weight: [2.0, 5.0, 10.0]`, `false_positive_weight: [2.0, 5.0, 10.0]`, `false_negative_weight: [3.0, 5.0, 10.0, 20.0]` | `MinMaxScaler` | `True` |
| `weighted_penalty_huber_b_sweep.py` | `WeightedPenaltyHuberLoss` | B | `delta: [50.0, 100.0, 200.0]`, `zero_threshold: [0.1, 0.5, 1.0]`, `non_zero_weight: [2.0, 5.0, 10.0]`, `false_positive_weight: [2.0, 5.0, 10.0]`, `false_negative_weight: [3.0, 5.0, 10.0, 20.0]` | `None` | `False` |
| `weighted_penalty_huber_c_sweep.py` | `WeightedPenaltyHuberLoss` | C | `delta: [0.5, 1.5]`, `zero_threshold: [0.001]`, `non_zero_weight: [2.0, 5.0, 10.0]`, `false_positive_weight: [2.0, 5.0, 10.0]`, `false_negative_weight: [3.0, 5.0, 10.0, 20.0]` | `StandardScaler` | `False` |
| `weighted_penalty_huber_e_sweep.py` | `WeightedPenaltyHuberLoss` | E | `delta: [0.2, 0.5, 1.0]`, `zero_threshold: [0.01, 0.1]`, `non_zero_weight: [2.0, 5.0, 10.0]`, `false_positive_weight: [2.0, 5.0, 10.0]`, `false_negative_weight: [3.0, 5.0, 10.0, 20.0]` | `None` | `True` |
| `weighted_penalty_huber_f_sweep.py` | `WeightedPenaltyHuberLoss` | F | `delta: [0.2, 0.5, 1.0]`, `zero_threshold: [0.01, 0.1]`, `non_zero_weight: [2.0, 5.0, 10.0]`, `false_positive_weight: [2.0, 5.0, 10.0]`, `false_negative_weight: [3.0, 5.0, 10.0, 20.0]` | `None` | `False` |
| `zero_inflated_a_sweep.py` | `ZeroInflatedLoss` | A | `zero_weight: [0.5, 1.0, 2.0, 5.0]`, `count_weight: [0.5, 1.0, 2.0, 5.0]`, `delta: [0.05, 0.1, 0.25]` | `MinMaxScaler` | `True` |
| `zero_inflated_b_sweep.py` | `ZeroInflatedLoss` | B | `zero_weight: [0.5, 1.0, 2.0, 5.0]`, `count_weight: [0.5, 1.0, 2.0, 5.0]`, `delta: [50.0, 100.0, 200.0]`, `zero_threshold: [0.1, 0.5, 1.0]` | `None` | `False` |
| `zero_inflated_c_sweep.py` | `ZeroInflatedLoss` | C | `zero_weight: [0.5, 1.0, 2.0, 5.0]`, `count_weight: [0.5, 1.0, 2.0, 5.0]`, `delta: [0.5, 1.5]`, `zero_threshold: [0.001]` | `StandardScaler` | `False` |
| `zero_inflated_e_sweep.py` | `ZeroInflatedLoss` | E | `zero_weight: [0.5, 1.0, 2.0, 5.0]`, `count_weight: [0.5, 1.0, 2.0, 5.0]`, `delta: [0.2, 0.5, 1.0]`, `zero_threshold: [0.01, 0.1]` | `None` | `True` |
| `zero_inflated_f_sweep.py` | `ZeroInflatedLoss` | F | `zero_weight: [0.5, 1.0, 2.0, 5.0]`, `count_weight: [0.5, 1.0, 2.0, 5.0]`, `delta: [0.2, 0.5, 1.0]`, `zero_threshold: [0.01, 0.1]` | `None` | `False` |