import math
import logging
import torch
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class WarmupCAWR(_LRScheduler):
    """CosineAnnealingWarmRestarts with linear warmup.

    For the first ``warmup_epochs``, lr ramps linearly from ``eta_min``
    to the optimizer's base lr. After that, standard CAWR cosine
    annealing with warm restarts takes over.

    Without warmup, CAWR starts at peak lr from step 0. For Transformers
    this causes 30 epochs of maximum-strength updates before any decay,
    leading to NaN (weight explosion) or peace-attractor collapse in
    narrow viable lr ranges.

    Warmup widens the viable lr window: the model trains at low lr for
    the first few epochs while weights are random, then ramps to full
    lr once representations have stabilised.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of epochs for linear warmup. 0 = no warmup
            (equivalent to plain CAWR).
        T_0: CAWR first cycle length (in epochs).
        T_mult: CAWR cycle length multiplier.
        eta_min: Minimum lr for both warmup start and CAWR troughs.
        last_epoch: The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        T_0: int = 30,
        T_mult: int = 2,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

        logger.info(
            "WarmupCAWR | warmup=%d epochs | CAWR T_0=%d T_mult=%d eta_min=%.2e | "
            "base_lrs=%s",
            warmup_epochs, T_0, T_mult, eta_min,
            [f"{lr:.4e}" for lr in self.base_lrs],
        )

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            lrs = [
                self.eta_min + (base_lr - self.eta_min) * alpha
                for base_lr in self.base_lrs
            ]
            logger.debug(
                "WarmupCAWR warmup | epoch=%d/%d alpha=%.3f lr=%s",
                self.last_epoch, self.warmup_epochs, alpha,
                [f"{lr:.4e}" for lr in lrs],
            )
            return lrs

        # CAWR phase: compute position within cosine cycle.
        epoch = self.last_epoch - self.warmup_epochs
        T_i = self.T_0
        offset = 0
        while offset + T_i <= epoch:
            offset += T_i
            T_i = int(T_i * self.T_mult)
        T_cur = epoch - offset

        lrs = [
            self.eta_min
            + (base_lr - self.eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
            for base_lr in self.base_lrs
        ]

        if epoch == 0:
            logger.info(
                "WarmupCAWR | warmup complete → CAWR begins | "
                "epoch=%d T_0=%d T_mult=%d lr=%s",
                self.last_epoch, self.T_0, self.T_mult,
                [f"{lr:.4e}" for lr in lrs],
            )
        elif T_cur == 0:
            logger.info(
                "WarmupCAWR | CAWR restart | epoch=%d T_i=%d lr=%s",
                self.last_epoch, T_i,
                [f"{lr:.4e}" for lr in lrs],
            )
        else:
            logger.debug(
                "WarmupCAWR CAWR | epoch=%d T_cur=%d/%d lr=%s",
                self.last_epoch, T_cur, T_i,
                [f"{lr:.4e}" for lr in lrs],
            )

        return lrs
