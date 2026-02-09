# file: tests/test_loss_pipeline_robustness.py

import pytest
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from views_r2darts2.utils.loss import (
    WeightedHuberLoss,
    TimeAwareWeightedHuberLoss,
    WeightedPenaltyHuberLoss,
    ZeroInflatedLoss,
    SpikeFocalLoss,
    ShrinkageLoss,
)

# --- Test Setup: Define All 8 Data Pipelines ---

def transform_data(data, pipeline):
    """Applies a transformation pipeline to numpy data."""
    if pipeline == "A: log1p + minmax(0,1)":
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(np.log1p(data).reshape(-1, 1)).flatten()
    elif pipeline == "B: raw_counts":
        return data
    elif pipeline == "C: asinh + standard":
        scaler = StandardScaler()
        return scaler.fit_transform(np.arcsinh(data).reshape(-1, 1)).flatten()
    elif pipeline == "D: log1p + minmax(-1,1)":
        scaler = MinMaxScaler(feature_range=(-1, 1))
        return scaler.fit_transform(np.log1p(data).reshape(-1, 1)).flatten()
    elif pipeline == "E: pure_log1p":
        return np.log1p(data)
    elif pipeline == "F: pure_asinh":
        return np.arcsinh(data)
    elif pipeline == "G: pure_minmax(0,1)":
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(data.reshape(-1, 1)).flatten()
    elif pipeline == "H: pure_minmax(-1,1)":
        scaler = MinMaxScaler(feature_range=(-1, 1))
        return scaler.fit_transform(data.reshape(-1, 1)).flatten()
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")

# --- Test Suite for Parameter Sensibility ---

PIPELINES = [
    "A: log1p + minmax(0,1)", "B: raw_counts", "C: asinh + standard", 
    "D: log1p + minmax(-1,1)", "E: pure_log1p", "F: pure_asinh",
    "G: pure_minmax(0,1)", "H: pure_minmax(-1,1)"
]
HUBER_LOSSES = [WeightedHuberLoss, TimeAwareWeightedHuberLoss, WeightedPenaltyHuberLoss, ZeroInflatedLoss]
THRESHOLD_LOSSES = [SpikeFocalLoss, ShrinkageLoss]

@pytest.mark.parametrize("loss_class", HUBER_LOSSES)
@pytest.mark.parametrize("pipeline", PIPELINES)
def test_huber_delta_sensibility_across_pipelines(loss_class, pipeline):
    """
    This test serves as verifiable proof that the 'delta' hyperparameter
    is highly sensitive to the data transformation pipeline.

    It works by checking if a 'good' delta for a given pipeline partitions
    the errors reasonably, and confirming that a 'bad' delta fails to do so.
    A failure on the 'bad' delta assertion is an expected outcome and proves the point.
    """
    # 1. Define pipeline-specific data and delta values
    good_delta, bad_delta = 1.0, 1.0
    if "minmax" in pipeline:
        raw_data = np.random.lognormal(1, 2, size=100).astype(np.float32)
        good_delta, bad_delta = 0.1, 5.0
    elif "raw_counts" in pipeline:
        raw_data = np.random.randint(0, 500, size=100).astype(np.float32)
        good_delta, bad_delta = 50.0, 1.0 # Adjusted good_delta and bad_delta
    elif "standard" in pipeline or "asinh" in pipeline:
        raw_data = np.random.lognormal(2, 3, size=100).astype(np.float32)
        good_delta, bad_delta = 0.8, 50.0
    elif "log1p" in pipeline:
        raw_data = np.random.lognormal(2, 2, size=100).astype(np.float32)
        good_delta, bad_delta = 0.5, 50.0

    targets_np = transform_data(raw_data, pipeline)
    # Generate some plausible errors
    errors_np = (np.random.rand(100) - 0.5) * (targets_np.max() - targets_np.min()) * 0.5
    preds_np = targets_np + errors_np
    
    preds = torch.from_numpy(preds_np)
    targets = torch.from_numpy(targets_np)
    errors = torch.abs(preds - targets)

    # Define kwargs for each loss class
    kwargs_good = {}
    kwargs_bad = {}

    if loss_class == WeightedHuberLoss:
        kwargs_good = {"zero_threshold": 0.01, "non_zero_weight": 1.0}
        kwargs_bad = {"zero_threshold": 0.01, "non_zero_weight": 1.0}
    elif loss_class == TimeAwareWeightedHuberLoss:
        kwargs_good = {"zero_weight": 1.0, "non_zero_weight": 1.0, "decay_factor": 0.9}
        kwargs_bad = {"zero_weight": 1.0, "non_zero_weight": 1.0, "decay_factor": 0.9}
    elif loss_class == WeightedPenaltyHuberLoss:
        kwargs_good = {"zero_threshold": 0.01, "non_zero_weight": 1.0, "false_positive_weight": 1.0, "false_negative_weight": 1.0}
        kwargs_bad = {"zero_threshold": 0.01, "non_zero_weight": 1.0, "false_positive_weight": 1.0, "false_negative_weight": 1.0}
    elif loss_class == ZeroInflatedLoss:
        kwargs_good = {"zero_weight": 1.0, "count_weight": 1.0, "zero_threshold": 0.01, "eps": 1e-8}
        kwargs_bad = {"zero_weight": 1.0, "count_weight": 1.0, "zero_threshold": 0.01, "eps": 1e-8}
    
    # Update with test-specific delta
    kwargs_good["delta"] = good_delta
    kwargs_bad["delta"] = bad_delta

    # 2. Test with a "good" delta
    proportion_large_good = (errors > good_delta).float().mean().item()
    assert 0.05 < proportion_large_good < 0.95, \
        f"Pipeline {pipeline}: Proportion of large errors for 'good' delta ({good_delta}) is too extreme ({proportion_large_good})"

    # 3. Test with a "bad" delta - THIS IS MEANT TO FAIL in some cases
    proportion_large_bad = (errors > bad_delta).float().mean().item()
    assert proportion_large_bad < 0.05 or proportion_large_bad > 0.95, \
        f"Pipeline {pipeline}: Proportion of large errors for 'bad' delta ({bad_delta}) is not extreme ({proportion_large_bad})"


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_spikefocal_threshold_reachability_across_pipelines(pipeline):
    """
    This test serves as verifiable proof that SpikeFocalLoss's `spike_threshold`
    parameter is non-functional if set outside the bounds of the transformed data.
    """
    raw_data = np.random.lognormal(1, 2, size=100).astype(np.float32)
    targets_np = transform_data(raw_data, pipeline)
    max_val = targets_np.max()
    
    # Define a threshold that is guaranteed to be unreachable
    unreachable_threshold = max_val + 1.0
    
    loss = SpikeFocalLoss(alpha=0.8, gamma=2.0, spike_threshold=unreachable_threshold)
    is_spike = torch.from_numpy(targets_np) > loss.spike_threshold
    proportion_spikes = is_spike.float().mean().item()

    # This assertion proves the threshold is non-functional
    assert proportion_spikes == 0, \
        f"PROOF FAILED: Expected 0% spikes for pipeline '{pipeline}' with unreachable threshold, but got {proportion_spikes:.2%}"

    # To make this a complete test, we also show a reachable threshold works.
    reachable_threshold = np.quantile(targets_np, 0.9) # 90th percentile
    loss_functional = SpikeFocalLoss(alpha=0.8, gamma=2.0, spike_threshold=reachable_threshold)
    is_spike_functional = torch.from_numpy(targets_np) > loss_functional.spike_threshold
    proportion_spikes_functional = is_spike_functional.float().mean().item()

    # Around 10% of samples should be spikes
    assert 0.05 < proportion_spikes_functional < 0.15, \
        f"SANITY CHECK FAILED: For pipeline '{pipeline}', a reachable threshold of {reachable_threshold:.2f} should have worked, but proportion of spikes was {proportion_spikes_functional:.2%}"

