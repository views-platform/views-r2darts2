
import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from views_r2darts2.utils.reproducibility_gate import ReproducibilityGate
from views_r2darts2.utils.exceptions import NumericalSanityError
from views_r2darts2.model.darts_forecaster import DartsForecaster
from views_r2darts2.utils.loss.loss_catalog import LossCatalog

def test_entropy_locking_parity():
    """Green Team: Verify entropy locking parity."""
    seed = 42
    def get_sample():
        ReproducibilityGate.Data.lock_entropy(seed)
        return torch.randn(5)
    sample1 = get_sample()
    sample2 = get_sample()
    assert torch.equal(sample1, sample2)
    print("✓ Entropy Locking Parity Verified.")

def test_loss_numerical_purge_red_team_all_losses():
    """Red Team: Verify ALL custom losses fail loudly on NaNs."""
    losses_to_test = {
        "WeightedPenaltyHuberLoss": {
            "zero_threshold": 0.01, "delta": 0.5, "non_zero_weight": 5.0,
            "false_positive_weight": 2.0, "false_negative_weight": 3.0
        },
        "ShrinkageLoss": {"a": 10.0, "c": 0.2},
        "WeightedHuberLoss": {"zero_threshold": 0.01, "delta": 0.5, "non_zero_weight": 5.0},
        "TimeAwareWeightedHuberLoss": {"zero_weight": 1.0, "non_zero_weight": 5.0, "decay_factor": 0.9, "delta": 0.5},
        "SpikeFocalLoss": {"alpha": 0.8, "gamma": 2.0, "spike_threshold": 3.0},
        "TweedieLoss": {"non_zero_weight": 5.0, "zero_threshold": 0.01, "p": 1.5, "false_positive_weight": 1.0, "false_negative_weight": 1.0, "eps": 1e-8},
        "AsymmetricQuantileLoss": {"tau": 0.75, "non_zero_weight": 5.0, "zero_threshold": 0.01},
        "ZeroInflatedLoss": {"zero_weight": 1.0, "count_weight": 1.0, "delta": 0.5, "zero_threshold": 0.01, "eps": 1e-8}
    }

    for name, kwargs in losses_to_test.items():
        loss_fn = LossCatalog({**kwargs, "loss_function": name}).get_loss()
        
        # Test NaN in Predictions
        preds_nan = torch.tensor([1.0, float('nan')], requires_grad=True)
        targets = torch.tensor([1.0, 1.0])
        with pytest.raises(NumericalSanityError, match="NaN or Inf detected in predictions"):
            loss_fn(preds_nan, targets)
            
        # Test NaN in Targets
        preds = torch.tensor([1.0, 1.0], requires_grad=True)
        targets_nan = torch.tensor([1.0, float('nan')])
        with pytest.raises(NumericalSanityError, match="NaN or Inf detected in targets"):
            loss_fn(preds, targets_nan)
            
        print(f"✓ {name}: Numerical Purge Verified.")

def test_loss_gradients_green_team():
    """Green Team: Verify gradient flow for custom losses."""
    losses_to_test = {
        "WeightedPenaltyHuberLoss": {
            "zero_threshold": 0.01, "delta": 0.5, "non_zero_weight": 5.0,
            "false_positive_weight": 2.0, "false_negative_weight": 3.0
        },
        "ShrinkageLoss": {"a": 10.0, "c": 0.2},
        "WeightedHuberLoss": {"zero_threshold": 0.01, "delta": 0.5, "non_zero_weight": 5.0},
        "SpikeFocalLoss": {"alpha": 0.8, "gamma": 2.0, "spike_threshold": 3.0},
        "TweedieLoss": {"non_zero_weight": 5.0, "zero_threshold": 0.01, "p": 1.5, "false_positive_weight": 1.0, "false_negative_weight": 1.0, "eps": 1e-8},
        "AsymmetricQuantileLoss": {"tau": 0.75, "non_zero_weight": 5.0, "zero_threshold": 0.01},
        "ZeroInflatedLoss": {"zero_weight": 1.0, "count_weight": 1.0, "delta": 0.5, "zero_threshold": 0.01, "eps": 1e-8}
    }

    for name, kwargs in losses_to_test.items():
        loss_fn = LossCatalog({**kwargs, "loss_function": name}).get_loss()
        preds = torch.tensor([0.5, 1.5, 0.0], requires_grad=True)
        targets = torch.tensor([0.0, 2.0, 0.0])
        
        loss = loss_fn(preds, targets)
        loss.backward()
        
        assert preds.grad is not None
        assert not torch.isnan(preds.grad).any()
        assert not torch.isinf(preds.grad).any()
        print(f"✓ {name}: Gradient Flow Verified.")

def test_forecaster_beige_team_constraints():
    """Beige Team: Verify DartsForecaster mandatory parameter rule."""
    dataset = MagicMock()
    model = MagicMock()
    partition = {"train": [0, 10], "test": [11, 20]}
    
    with pytest.raises(ValueError, match="MANDATORY PARAMETER MISSING"):
        DartsForecaster(
            dataset=dataset,
            model=model,
            partition_dict=partition,
            random_state=None
        )
    print("✓ Forecaster: No-Defaults Rule Verified.")

def test_forecaster_nan_airlock_red_team():
    """Red Team: Verify Forecaster fails if NaNs detected in output DF."""
    dataset = MagicMock()
    dataset._time_id = "time"
    dataset._entity_id = "entity"
    dataset.targets = ["target"]
    dataset.features = []
    dataset.as_darts_timeseries.return_value = []
    
    from darts import TimeSeries
    nan_series = TimeSeries.from_times_and_values(
        pd.date_range("2000-01-01", periods=1),
        np.array([[[np.nan]]]),
        static_covariates=pd.DataFrame({"entity": [1]})
    )
    model = MagicMock()
    model.predict.return_value = [nan_series]
    model.input_chunk_length = 1
    
    forecaster = DartsForecaster(
        dataset=dataset,
        model=model,
        partition_dict={"train": [0, 10], "test": [11, 20]},
        random_state=42
    )
    forecaster.scaler_fitted = True
    forecaster.device = "cpu"
    
    with pytest.raises(NumericalSanityError, match="NaN detected in Model Predictions"):
        forecaster.predict(0)
    print("✓ Forecaster: NaN Airlock Verified.")

if __name__ == "__main__":
    try:
        test_entropy_locking_parity()
        test_loss_numerical_purge_red_team_all_losses()
        test_loss_gradients_green_team()
        test_forecaster_beige_team_constraints()
        test_forecaster_nan_airlock_red_team()
        print("\nALL FORTRESS HARDENING TESTS PASSED. 🖖")
    except Exception as e:
        print(f"\nTEST FAILURE: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
