import torch
import numpy as np
import logging
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)

class NaNDetectionCallback(Callback):
    """
    Callback to detect NaN loss and stop training early.
    """
    def __init__(self, patience: int = 3):
        super().__init__()
        self.patience = patience
        self.nan_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is not None and torch.isnan(loss):
            self.nan_count += 1
            logger.warning(
                f"NaN loss detected at epoch {trainer.current_epoch}, batch {batch_idx} "
                f"(consecutive NaN count: {self.nan_count}/{self.patience})"
            )
            if self.nan_count >= self.patience:
                logger.error("Training stopped due to persistent NaN loss.")
                trainer.should_stop = True
        else:
            self.nan_count = 0 

class GradientHealthCallback(Callback):
    """
    Callback to monitor gradient health after each epoch.
    """
    def __init__(
        self,
        log_every_n_epochs: int = 1,
        warn_threshold: float = 1e-7,
        explode_threshold: float = 100.0,
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.warn_threshold = warn_threshold
        self.explode_threshold = explode_threshold

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        grad_norms = []
        nan_count = 0
        inf_count = 0
        zero_count = 0
        total_params = 0

        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                total_params += 1
                grad = param.grad.detach()
                norm = grad.norm().item()

                if np.isnan(norm):
                    nan_count += 1
                elif np.isinf(norm):
                    inf_count += 1
                elif norm == 0:
                    zero_count += 1
                else:
                    grad_norms.append(norm)

        if not grad_norms and total_params == 0:
            return 

        if grad_norms:
            grad_norms = np.array(grad_norms)
            stats = {
                "min": grad_norms.min(),
                "max": grad_norms.max(),
                "mean": grad_norms.mean(),
                "median": np.median(grad_norms),
            }
        else:
            stats = {"min": 0, "max": 0, "mean": 0, "median": 0}

        status = "✅ healthy"
        if nan_count > 0:
            status = f"🚨 {nan_count} NaN grads!"
        elif inf_count > 0:
            status = f"🚨 {inf_count} Inf grads!"
        elif stats["max"] > self.explode_threshold:
            status = f"🚨 exploding (max={stats['max']:.1f})"
        elif stats["max"] < self.warn_threshold:
            status = f"🚨 vanishing (max={stats['max']:.2e})"

        logger.info(
            f"[Epoch {trainer.current_epoch}] Gradients {status} | "
            f"norm: min={stats['min']:.2e}, max={stats['max']:.2e}, "
            f"mean={stats['mean']:.2e}, median={stats['median']:.2e} | "
            f"zero={zero_count}/{total_params}"
        )
