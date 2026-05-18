import math
import logging
import torch
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class WarmupCosine(_LRScheduler):
    """Linear warmup → cosine decay to eta_min. No restarts.

    Phase 1 (epochs 0..warmup_epochs-1):
        lr = eta_min + (base_lr - eta_min) * epoch / warmup_epochs

    Phase 2 (epochs warmup_epochs..T_max-1):
        Standard half-cosine decay from base_lr to eta_min over
        (T_max - warmup_epochs) epochs.

    Unlike CAWR/WarmupCAWR, there are no LR restarts. Once cosine decay
    reaches eta_min it stays there. This prevents the curriculum-disrupting
    LR spikes that de-calibrate per-series means after SpotlightLoss's
    level anchor has already converged.

    Unlike ReduceLROnPlateau, decay is deterministic — no false plateau
    detection from batch-noise-amplified loss variance. The model never
    gets trapped at a prematurely low LR.

    Set T_max = n_epochs for standard use. The scheduler gracefully
    clamps to eta_min for any epoch >= T_max.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Epochs for linear warmup. 0 = pure cosine.
        T_max: Total scheduler length (including warmup). Should match
            n_epochs. After T_max epochs, lr stays at eta_min.
        eta_min: Minimum lr (warmup floor and cosine trough).
        last_epoch: Index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 10,
        T_max: int = 300,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

        logger.info(
            "WarmupCosine | warmup=%d epochs | cosine T_max=%d eta_min=%.2e | "
            "base_lrs=%s",
            warmup_epochs, T_max, eta_min,
            [f"{lr:.4e}" for lr in self.base_lrs],
        )

    def get_lr(self):
        epoch = self.last_epoch

        # Phase 1: linear warmup
        if epoch < self.warmup_epochs:
            alpha = epoch / max(1, self.warmup_epochs)
            lrs = [
                self.eta_min + (base_lr - self.eta_min) * alpha
                for base_lr in self.base_lrs
            ]
            logger.debug(
                "WarmupCosine warmup | epoch=%d/%d alpha=%.3f lr=%s",
                epoch, self.warmup_epochs, alpha,
                [f"{lr:.4e}" for lr in lrs],
            )
            return lrs

        # Phase 2: cosine decay (no restarts)
        cosine_epochs = self.T_max - self.warmup_epochs
        t = min(epoch - self.warmup_epochs, cosine_epochs)
        lrs = [
            self.eta_min
            + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / max(1, cosine_epochs))) / 2
            for base_lr in self.base_lrs
        ]

        if epoch == self.warmup_epochs:
            logger.info(
                "WarmupCosine | warmup complete → cosine decay begins | "
                "epoch=%d cosine_epochs=%d lr=%s",
                epoch, cosine_epochs,
                [f"{lr:.4e}" for lr in lrs],
            )
        else:
            logger.debug(
                "WarmupCosine cosine | epoch=%d t=%d/%d lr=%s",
                epoch, t, cosine_epochs,
                [f"{lr:.4e}" for lr in lrs],
            )

        return lrs
