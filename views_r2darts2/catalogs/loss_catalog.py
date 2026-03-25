import torch
import logging
from views_r2darts2.math.shrinkage_loss import ShrinkageLoss
from views_r2darts2.math.weighted_huber_loss import WeightedHuberLoss
from views_r2darts2.math.time_aware_weighted_huber_loss import TimeAwareWeightedHuberLoss
from views_r2darts2.math.spike_focal_loss import SpikeFocalLoss
from views_r2darts2.math.weighted_penalty_huber_loss import WeightedPenaltyHuberLoss
from views_r2darts2.math.tweedie_loss import TweedieLoss
from views_r2darts2.math.asymmetric_quantile_loss import AsymmetricQuantileLoss
from views_r2darts2.math.zero_inflated_loss import ZeroInflatedLoss
from views_r2darts2.math.spotlight_loss import SpotlightLoss
from views_r2darts2.math.sentinel_loss import SentinelLoss

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
            "beta": self.config.get("beta"),
            "kappa": self.config.get("kappa"),
        }

    def get_loss(self) -> torch.nn.Module:
        from views_r2darts2.infrastructure.reproducibility_gate import ReproducibilityGate
        
        loss_classes = {
            "WeightedHuberLoss": WeightedHuberLoss,
            "TimeAwareWeightedHuberLoss": TimeAwareWeightedHuberLoss,
            "SpikeFocalLoss": SpikeFocalLoss,
            "WeightedPenaltyHuberLoss": WeightedPenaltyHuberLoss,
            "TweedieLoss": TweedieLoss,
            "AsymmetricQuantileLoss": AsymmetricQuantileLoss,
            "ZeroInflatedLoss": ZeroInflatedLoss,
            "ShrinkageLoss": ShrinkageLoss,
            "SpotlightLoss": SpotlightLoss,
            "SentinelLoss": SentinelLoss,
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
