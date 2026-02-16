import torch
import inspect
import logging
from .shrinkage import ShrinkageLoss
from .weighted_huber import WeightedHuberLoss
from .time_aware_huber import TimeAwareWeightedHuberLoss
from .spike_focal import SpikeFocalLoss
from .weighted_penalty_huber import WeightedPenaltyHuberLoss
from .tweedie import TweedieLoss
from .quantile import AsymmetricQuantileLoss
from .zero_inflated import ZeroInflatedLoss

logger = logging.getLogger(__name__)

class LossSelector:
    @staticmethod
    def get_loss_function(loss_name, **kwargs):
        """
        Returns an instance of the specified loss function class with provided keyword arguments.

        Parameters:
            loss_name (str): The name of the loss function to instantiate. 
            **kwargs: Arbitrary keyword arguments to pass to the loss function's constructor.

        Returns:
            An instance of the requested loss function class.

        Raises:
            ValueError: If the provided loss_name is not recognized.
        """
        loss_classes = {
            "WeightedHuberLoss": WeightedHuberLoss,
            "TimeAwareWeightedHuberLoss": TimeAwareWeightedHuberLoss,
            "SpikeFocalLoss": SpikeFocalLoss,
            "WeightedPenaltyHuberLoss": WeightedPenaltyHuberLoss,
            "TweedieLoss": TweedieLoss,
            "AsymmetricQuantileLoss": AsymmetricQuantileLoss,
            "ZeroInflatedLoss": ZeroInflatedLoss,
            "ShrinkageLoss": ShrinkageLoss,
            # Standard PyTorch losses
            "MSELoss": torch.nn.MSELoss,
            "L1Loss": torch.nn.L1Loss,
            "HuberLoss": torch.nn.HuberLoss,
            "SmoothL1Loss": torch.nn.SmoothL1Loss,
            "PoissonNLLLoss": torch.nn.PoissonNLLLoss,
        }

        if loss_name not in loss_classes:
            raise ValueError(f"Unknown loss function: {loss_name}")

        cls = loss_classes[loss_name]
        
        # Standard PyTorch losses might not need filtering if they take standard args
        if cls.__module__.startswith("torch.nn"):
             return cls(**kwargs)

        # For our custom losses, filter kwargs to only include valid parameters
        params = inspect.signature(cls).parameters
        valid_kwargs = {k: v for k, v in kwargs.items() if k in params}
        
        # Check for missing parameters (NO DEFAULTS rule enforcement at instantiation)
        required_params = [
            k for k, v in params.items() 
            if v.default is inspect.Parameter.empty and v.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]
        missing = [p for p in required_params if p not in valid_kwargs]
        if missing:
            raise ValueError(f"MANDATORY LOSS PARAMETERS MISSING for {loss_name}: {missing}")

        return cls(**valid_kwargs)
