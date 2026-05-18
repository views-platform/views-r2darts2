import torch
import logging
from typing import Dict, Any, Type

logger = logging.getLogger(__name__)

class OptimizerCatalog:
    """
    Genome Translator for PyTorch optimizers.
    
    This class is responsible for mapping configuration DNA to concrete torch.optim 
    classes and their required hyperparameters, ensuring strict genomic compliance.
    """
    def __init__(self, config: dict):
        self.config = config
        self.opt_name = self.config["optimizer_cls"]
        
        self._all_potential_args = {
            "lr": self.config.get("lr"),
            "weight_decay": self.config.get("weight_decay"),
            "momentum": self.config.get("momentum"),
            "alpha": self.config.get("alpha"),
        }

    def get_optimizer_cls(self) -> Type[torch.optim.Optimizer]:
        """Returns the torch.optim class based on the DNA name."""
        try:
            return getattr(torch.optim, self.opt_name)
        except AttributeError:
            error_msg = f"INVALID OPTIMIZER: '{self.opt_name}' is not a valid torch.optim class."
            logger.critical(error_msg)
            raise ValueError(error_msg)

    def get_optimizer_kwargs(self) -> Dict[str, Any]:
        """
        Extracts and validates optimizer hyperparameters using the registered Genome.
        """
        from views_r2darts2.infrastructure.reproducibility_gate import ReproducibilityGate
        
        if self.opt_name not in ReproducibilityGate.Config.OPTIMIZER_GENOMES:
            available_opts = list(ReproducibilityGate.Config.OPTIMIZER_GENOMES.keys())
            error_msg = (
                f"UNREGISTERED OPTIMIZER: '{self.opt_name}' is not in the Fortress whitelist.\n"
                f"Registered optimizers: {available_opts}"
            )
            logger.critical(error_msg)
            raise ValueError(error_msg)
        
        opt_genome = ReproducibilityGate.Config.OPTIMIZER_GENOMES[self.opt_name]
        valid_kwargs = {k: v for k, v in self._all_potential_args.items() if k in opt_genome}
        
        missing = [k for k in opt_genome if valid_kwargs.get(k) is None]
        if missing:
            error_msg = f"MANDATORY OPTIMIZER GENES MISSING for {self.opt_name}: {missing}"
            logger.critical(error_msg)
            raise ValueError(error_msg)
            
        return valid_kwargs
