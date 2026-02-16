import torch
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

class LossCatalog:
    """
    Genome Translator for mathematical loss functions.
    
    This class is responsible for mapping configuration DNA to concrete torch.nn.Module 
    instances, ensuring that all mandatory loss hyperparameters are present and valid.
    """
    def __init__(self, config: dict):
        self.config = config
        self.loss_name = self.config["loss_function"]
        
        self._all_potential_args = {
            "zero_threshold": self.config.get("zero_threshold"),
            "delta": self.config.get("delta"),
            "non_zero_weight": self.config.get("non_zero_weight"),
            "false_negative_weight": self.config.get("false_negative_weight"),
            "false_positive_weight": self.config.get("false_positive_weight"),
            "tau": self.config.get("tau"),
            "a": self.config.get("a"),
            "c": self.config.get("c"),
            "alpha": self.config.get("alpha"),
            "gamma": self.config.get("gamma"),
            "spike_threshold": self.config.get("spike_threshold"),
            "zero_weight": self.config.get("zero_weight"),
            "count_weight": self.config.get("count_weight"),
            "p": self.config.get("p"),
            "eps": self.config.get("eps"),
            "decay_factor": self.config.get("decay_factor"),
        }

    def get_loss(self) -> torch.nn.Module:
        from views_r2darts2.utils.gates import ReproducibilityGate
        
        loss_classes = {
            "WeightedHuberLoss": WeightedHuberLoss,
            "TimeAwareWeightedHuberLoss": TimeAwareWeightedHuberLoss,
            "SpikeFocalLoss": SpikeFocalLoss,
            "WeightedPenaltyHuberLoss": WeightedPenaltyHuberLoss,
            "TweedieLoss": TweedieLoss,
            "AsymmetricQuantileLoss": AsymmetricQuantileLoss,
            "ZeroInflatedLoss": ZeroInflatedLoss,
            "ShrinkageLoss": ShrinkageLoss,
            "MSELoss": torch.nn.MSELoss,
            "L1Loss": torch.nn.L1Loss,
            "HuberLoss": torch.nn.HuberLoss,
            "SmoothL1Loss": torch.nn.SmoothL1Loss,
            "PoissonNLLLoss": torch.nn.PoissonNLLLoss,
        }

        if self.loss_name not in loss_classes:
            available_losses = list(loss_classes.keys())
            error_msg = (
                f"Unknown loss function: {self.loss_name}.\n"
                f"Available loss functions: {available_losses}"
            )
            logger.critical(error_msg)
            raise ValueError(error_msg)

        cls = loss_classes[self.loss_name]
        
        if cls.__module__.startswith("torch.nn"):
             if self.loss_name == "HuberLoss":
                 delta = self.config.get("delta")
                 if delta is None:
                     raise ValueError("MANDATORY LOSS GENES MISSING for HuberLoss: ['delta']")
                 return cls(delta=delta)
             return cls()

        loss_genome = ReproducibilityGate.Config.LOSS_GENOMES.get(self.loss_name, [])
        valid_kwargs = {k: v for k, v in self._all_potential_args.items() if k in loss_genome}
        
        missing = [k for k in loss_genome if valid_kwargs.get(k) is None]
        if missing:
            error_msg = f"MANDATORY LOSS GENES MISSING for {self.loss_name}: {missing}"
            logger.critical(error_msg)
            raise ValueError(error_msg)

        return cls(**valid_kwargs)
