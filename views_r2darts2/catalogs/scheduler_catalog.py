import torch
import logging
from typing import Dict, Any, Type

logger = logging.getLogger(__name__)


class SchedulerCatalog:
    """
    Genome Translator for PyTorch LR schedulers.

    Maps configuration DNA to concrete torch.optim.lr_scheduler classes
    and their required hyperparameters, ensuring strict genomic compliance.
    """

    # Maps config key prefixes → the kwarg name expected by each scheduler class.
    _KWARG_MAP = {
        "ReduceLROnPlateau": {
            "lr_scheduler_factor": "factor",
            "lr_scheduler_patience": "patience",
            "lr_scheduler_min_lr": "min_lr",
        },
        "CosineAnnealingWarmRestarts": {
            "lr_scheduler_T_0": "T_0",
            "lr_scheduler_T_mult": "T_mult",
            "lr_scheduler_eta_min": "eta_min",
        },
        "WarmupCAWR": {
            "lr_scheduler_warmup_epochs": "warmup_epochs",
            "lr_scheduler_T_0": "T_0",
            "lr_scheduler_T_mult": "T_mult",
            "lr_scheduler_eta_min": "eta_min",
        },
        "StepLR": {
            "lr_scheduler_step_size": "step_size",
            "lr_scheduler_gamma": "gamma",
        },
        "ExponentialLR": {
            "lr_scheduler_gamma": "gamma",
        },
    }

    # Extra static kwargs injected per scheduler (not part of the DNA genome).
    _STATIC_KWARGS = {
        "ReduceLROnPlateau": {"mode": "min", "monitor": "train_loss"},
    }

    def __init__(self, config: dict):
        self.config = config
        self.sched_name = self.config["lr_scheduler_cls"]

    # Custom scheduler classes not in torch.optim.lr_scheduler.
    _CUSTOM_SCHEDULERS = {}

    @classmethod
    def _load_custom_schedulers(cls):
        if not cls._CUSTOM_SCHEDULERS:
            from views_r2darts2.math.warmup_cawr import WarmupCAWR
            cls._CUSTOM_SCHEDULERS["WarmupCAWR"] = WarmupCAWR

    def get_scheduler_cls(self) -> Type:
        """Returns the torch.optim.lr_scheduler class based on the DNA name."""
        self._load_custom_schedulers()
        if self.sched_name in self._CUSTOM_SCHEDULERS:
            return self._CUSTOM_SCHEDULERS[self.sched_name]
        try:
            return getattr(torch.optim.lr_scheduler, self.sched_name)
        except AttributeError:
            error_msg = (
                f"INVALID SCHEDULER: '{self.sched_name}' is not a valid "
                f"torch.optim.lr_scheduler class."
            )
            logger.critical(error_msg)
            raise ValueError(error_msg)

    def get_scheduler_kwargs(self) -> Dict[str, Any]:
        """
        Extracts and validates scheduler hyperparameters using the registered Genome.
        """
        from views_r2darts2.infrastructure.reproducibility_gate import ReproducibilityGate

        if self.sched_name not in ReproducibilityGate.Config.SCHEDULER_GENOMES:
            available = list(ReproducibilityGate.Config.SCHEDULER_GENOMES.keys())
            error_msg = (
                f"UNREGISTERED SCHEDULER: '{self.sched_name}' is not in the "
                f"Fortress whitelist.\nRegistered schedulers: {available}"
            )
            logger.critical(error_msg)
            raise ValueError(error_msg)

        kwarg_map = self._KWARG_MAP.get(self.sched_name, {})
        genome_keys = ReproducibilityGate.Config.SCHEDULER_GENOMES[self.sched_name]

        kwargs: Dict[str, Any] = {}
        missing = []
        for config_key in genome_keys:
            val = self.config.get(config_key)
            if val is None:
                missing.append(config_key)
            else:
                torch_key = kwarg_map.get(config_key, config_key)
                kwargs[torch_key] = val

        if missing:
            error_msg = (
                f"MANDATORY SCHEDULER GENES MISSING for "
                f"{self.sched_name}: {missing}"
            )
            logger.critical(error_msg)
            raise ValueError(error_msg)

        # Inject any static kwargs for the scheduler
        kwargs.update(self._STATIC_KWARGS.get(self.sched_name, {}))

        return kwargs
