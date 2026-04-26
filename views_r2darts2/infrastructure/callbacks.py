import time
import torch
import numpy as np
import logging
from collections import deque
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NaN Detection
# ---------------------------------------------------------------------------


class TrainingStepPatchCallback(Callback):
    """
    Patches Darts' training_step to expose predictions to downstream callbacks.

    Intent Contract:
        - Purpose: Darts' ``training_step`` returns only the scalar loss, so
          callbacks like ``YHatBarCallback`` and ``PredictionSanityCallback`` that
          need access to ``y_pred`` and ``y_true`` during training receive nothing.
          This callback monkey-patches ``training_step`` at ``on_fit_start`` to
          store predictions on ``pl_module.last_predictions`` and truth on
          ``pl_module.last_targets`` after each batch.
        - Guarantees: Downstream callbacks that check ``hasattr(pl_module,
          'last_predictions')`` will get fresh per-batch tensors. Original
          ``training_step`` behaviour (loss, logging, metrics) is unchanged.
        - Non-Goals: Does not add memory cost beyond one batch of detached tensors.

    Must be placed FIRST in the callback list so the patch is applied before
    other callbacks run.
    """

    def on_fit_start(self, trainer, pl_module):
        if hasattr(pl_module, "_original_training_step"):
            return  # Already patched

        original = pl_module.training_step

        def patched_training_step(train_batch, batch_idx):
            # Darts convention: batch[-1] = future target, batch[-2] = sample weights
            output = pl_module._produce_train_output(train_batch[:-2])
            sample_weight = train_batch[-2]
            target = train_batch[-1]
            loss = pl_module._compute_loss(
                output, target, pl_module.train_criterion, sample_weight
            )
            pl_module.log(
                "train_loss",
                loss,
                batch_size=train_batch[0].shape[0],
                prog_bar=True,
                sync_dist=True,
            )
            pl_module._update_metrics(output, target, pl_module.train_metrics)

            # ── Store predictions & truth for downstream callbacks ────
            # Squeeze likelihood dimension: (B, T, C, n_params) → (B, T, C)
            preds = output.detach()
            if preds.dim() == 4:
                preds = preds[..., 0]  # point forecast (first likelihood param)
            pl_module.last_predictions = preds
            pl_module.last_targets = target.detach()

            return loss

        pl_module._original_training_step = original
        pl_module.training_step = patched_training_step
        logger.info("TrainingStepPatchCallback: patched training_step to expose predictions")


class NaNDetectionCallback(Callback):
    """
    Batch-level NaN sentinel that halts training after consecutive NaN losses.

    Intent Contract:
        - Purpose: Catch diverged training runs as fast as possible, before they
          waste GPU hours producing garbage checkpoints.
        - Guarantees: Training is stopped after ``patience`` consecutive NaN-loss
          batches. Counter resets as soon as one valid batch is seen.
        - Failure Behavior: Sets ``trainer.should_stop = True`` and logs an ERROR.

    Parameters
    ----------
    patience : int, default 3
        Number of consecutive NaN-loss batches tolerated before stopping.
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


# ---------------------------------------------------------------------------
# Gradient Health
# ---------------------------------------------------------------------------


class GradientHealthCallback(Callback):
    """
    Epoch-level gradient norm auditor.

    Intent Contract:
        - Purpose: Surface vanishing, exploding, NaN, or Inf gradients so the
          operator can adjust clipping / learning rate before the run is lost.
        - Guarantees: Logs per-epoch gradient statistics and a human-readable
          health verdict. Also pushes scalar metrics to the PL logger (e.g. wandb).
        - Non-Goals: Does not modify gradients or stop training.

    Parameters
    ----------
    log_every_n_epochs : int, default 1
        How often to run the audit.
    warn_threshold : float, default 1e-7
        Maximum gradient norm below which gradients are flagged as vanishing.
    explode_threshold : float, default 100.0
        Minimum gradient norm above which gradients are flagged as exploding.
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
                elif norm == 0 and not name.endswith(".bias"):
                    # Skip bias parameters: LayerNorm/Linear biases are zero-initialized
                    # by PyTorch, and on zero-inflated data (90% peace cells) their
                    # gradients underflow to exactly 0.0 in early training — same false
                    # alarm as the collapsed-bias fix in WeightNormCallback.
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

        # Push scalars to PL logger (wandb) if available
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    "grad_norm/min": stats["min"],
                    "grad_norm/max": stats["max"],
                    "grad_norm/mean": stats["mean"],
                    "grad_norm/median": stats["median"],
                    "grad_norm/nan_count": nan_count,
                    "grad_norm/inf_count": inf_count,
                    "grad_norm/zero_count": zero_count,
                },
                step=trainer.global_step,
            )

        logger.info(
            f"[Epoch {trainer.current_epoch}] Gradients {status} | "
            f"norm: min={stats['min']:.2e}, max={stats['max']:.2e}, "
            f"mean={stats['mean']:.2e}, median={stats['median']:.2e} | "
            f"zero={zero_count}/{total_params}"
        )


# ---------------------------------------------------------------------------
# Weight Norm Monitor
# ---------------------------------------------------------------------------


class WeightNormCallback(Callback):
    """
    Epoch-level parameter weight-magnitude auditor.

    Intent Contract:
        - Purpose: Detect slow weight explosion or collapse that gradient norms
          alone cannot catch (gradients may look healthy while weights drift to
          extreme values over many epochs).
        - Guarantees: Logs per-epoch weight-norm statistics, flags layers whose
          norms exceed ``explode_threshold`` or fall below ``collapse_threshold``,
          and pushes scalars to the PL logger for wandb tracking.
        - Non-Goals: Does not modify weights or stop training.

    Parameters
    ----------
    log_every_n_epochs : int, default 1
        How often to run the audit.
    explode_threshold : float, default 1e4
        Layers with weight norm above this are flagged as exploding.
    collapse_threshold : float, default 1e-8
        Layers with weight norm below this are flagged as collapsed.
    """

    def __init__(
        self,
        log_every_n_epochs: int = 1,
        explode_threshold: float = 1e4,
        collapse_threshold: float = 1e-8,
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.explode_threshold = explode_threshold
        self.collapse_threshold = collapse_threshold

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        weight_norms = []
        exploding_layers = []
        collapsed_layers = []

        for name, param in pl_module.named_parameters():
            if not param.requires_grad:
                continue
            norm = param.data.detach().norm().item()
            weight_norms.append(norm)

            if norm > self.explode_threshold:
                exploding_layers.append((name, norm))
            elif norm < self.collapse_threshold and not name.endswith(".bias"):
                # Skip bias parameters: LayerNorm, BatchNorm, and Linear biases are
                # zero-initialized by PyTorch convention (e.g. LayerNorm.bias = 0.0).
                # Flagging them as "collapsed" is a false alarm — the weight matrices
                # are what matters, and a zero bias norm is expected at epoch 0.
                collapsed_layers.append((name, norm))

        if not weight_norms:
            return

        arr = np.array(weight_norms)
        stats = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
        }

        # Determine health verdict
        status = "✅ healthy"
        if exploding_layers:
            status = f"🚨 {len(exploding_layers)} exploding layer(s)"
            for name, norm in exploding_layers[:3]:
                logger.warning(
                    f"  ↳ weight explosion: {name} norm={norm:.2e}"
                )
        if collapsed_layers:
            collapse_msg = f"🚨 {len(collapsed_layers)} collapsed layer(s)"
            status = collapse_msg if status.startswith("✅") else f"{status} + {collapse_msg}"
            for name, norm in collapsed_layers[:3]:
                logger.warning(
                    f"  ↳ weight collapse: {name} norm={norm:.2e}"
                )

        # Push scalars to PL logger (wandb) if available
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    "weight_norm/min": stats["min"],
                    "weight_norm/max": stats["max"],
                    "weight_norm/mean": stats["mean"],
                    "weight_norm/median": stats["median"],
                    "weight_norm/exploding_layers": len(exploding_layers),
                    "weight_norm/collapsed_layers": len(collapsed_layers),
                },
                step=trainer.global_step,
            )

        logger.info(
            f"[Epoch {trainer.current_epoch}] Weights {status} | "
            f"norm: min={stats['min']:.2e}, max={stats['max']:.2e}, "
            f"mean={stats['mean']:.2e}, median={stats['median']:.2e} | "
            f"layers={len(weight_norms)}"
        )


# ---------------------------------------------------------------------------
# Prediction Sanity (Mode-Collapse Detector)
# ---------------------------------------------------------------------------


class PredictionSanityCallback(Callback):
    """
    Epoch-level mode-collapse detector for imbalanced time-series regression.

    Intent Contract:
        - Purpose: Detect the most common silent failure mode on zero-inflated
          data — the model learning to predict a near-constant value (usually ≈ 0)
          for every sample. This looks fine on aggregate MSE but produces
          operationally useless forecasts.
        - Guarantees: At the end of every ``check_every_n_epochs`` epoch the
          callback hooks into the PL module's last training batch outputs to
          inspect prediction variance. If the standard deviation of predictions
          falls below ``variance_floor`` for ``patience`` consecutive checks,
          an ERROR is logged. Statistics are always pushed to wandb.
        - Non-Goals: Does not stop training (the operator decides). Does not
          require a held-out validation set.

    How it works:
        After each qualifying epoch, the callback reads the model's last
        recorded predictions from an internal buffer populated by
        ``on_train_batch_end``. It computes the standard deviation and the
        fraction of predictions within ``collapse_band`` of the mean.
        If std < ``variance_floor`` *and* the near-mean fraction exceeds 95 %,
        the model is flagged as collapsed.

    Parameters
    ----------
    check_every_n_epochs : int, default 1
        How often to run the check.
    variance_floor : float, default 1e-4
        Prediction std below this triggers the collapse flag.
    collapse_band : float, default 1e-3
        Absolute distance from the mean within which a prediction is counted
        as "near-constant".
    patience : int, default 5
        Number of consecutive collapsed epochs before an ERROR is emitted.
    """

    def __init__(
        self,
        check_every_n_epochs: int = 1,
        variance_floor: float = 1e-4,
        collapse_band: float = 1e-3,
        patience: int = 5,
    ):
        super().__init__()
        self.check_every_n_epochs = check_every_n_epochs
        self.variance_floor = variance_floor
        self.collapse_band = collapse_band
        self.patience = patience

        self._consecutive_collapses = 0
        self._last_preds: torch.Tensor | None = None

    # -- Capture the last batch predictions each step -----------------------

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Store the raw predictions from the last training batch."""
        if isinstance(outputs, dict) and "preds" in outputs:
            self._last_preds = outputs["preds"].detach()
        elif isinstance(outputs, dict) and "y_hat" in outputs:
            self._last_preds = outputs["y_hat"].detach()
        elif hasattr(pl_module, "last_predictions"):
            self._last_preds = pl_module.last_predictions.detach()

    # -- Epoch-end analysis -------------------------------------------------

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.check_every_n_epochs != 0:
            return

        # Fallback: try to read from the PL module if batch hook did not fire
        preds = self._last_preds
        if preds is None:
            logger.debug(
                "[PredictionSanity] No predictions captured this epoch — skipping."
            )
            return

        preds_flat = preds.float().flatten()
        pred_std = preds_flat.std().item()
        pred_mean = preds_flat.mean().item()
        pred_min = preds_flat.min().item()
        pred_max = preds_flat.max().item()
        near_mean_frac = (
            (preds_flat - pred_mean).abs() < self.collapse_band
        ).float().mean().item()

        is_collapsed = pred_std < self.variance_floor and near_mean_frac > 0.95

        if is_collapsed:
            self._consecutive_collapses += 1
        else:
            self._consecutive_collapses = 0

        # Determine verdict
        if self._consecutive_collapses >= self.patience:
            status = (
                f"🚨 MODE COLLAPSE for {self._consecutive_collapses} consecutive epochs"
            )
            logger.error(
                f"[Epoch {trainer.current_epoch}] Predictions {status} | "
                f"std={pred_std:.2e}, mean={pred_mean:.4f}, "
                f"near-mean fraction={near_mean_frac:.1%}"
            )
        elif is_collapsed:
            status = f"⚠️  low variance ({self._consecutive_collapses}/{self.patience})"
            logger.warning(
                f"[Epoch {trainer.current_epoch}] Predictions {status} | "
                f"std={pred_std:.2e}, mean={pred_mean:.4f}, "
                f"near-mean fraction={near_mean_frac:.1%}"
            )
        else:
            status = "✅ diverse"
            logger.info(
                f"[Epoch {trainer.current_epoch}] Predictions {status} | "
                f"std={pred_std:.2e}, range=[{pred_min:.4f}, {pred_max:.4f}], "
                f"mean={pred_mean:.4f}"
            )

        # Push scalars to PL logger (wandb) if available
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    "pred_sanity/std": pred_std,
                    "pred_sanity/mean": pred_mean,
                    "pred_sanity/min": pred_min,
                    "pred_sanity/max": pred_max,
                    "pred_sanity/near_mean_frac": near_mean_frac,
                    "pred_sanity/consecutive_collapses": self._consecutive_collapses,
                },
                step=trainer.global_step,
            )

        # Clear for next epoch
        self._last_preds = None


# ---------------------------------------------------------------------------
# Loss Stability Monitor
# ---------------------------------------------------------------------------


class LossStabilityCallback(Callback):
    """
    Rolling-window loss stability monitor with spike and oscillation detection.

    Intent Contract:
        - Purpose: Catch training instability patterns that EarlyStopping misses —
          sudden loss spikes, persistent high-frequency oscillation, or a slowly
          widening variance — all of which degrade final model quality even if the
          mean loss trend is still descending.
        - Guarantees: Maintains a rolling window of recent batch losses, computes
          mean / std / coefficient-of-variation (CV) at epoch end, detects spikes
          (any single loss > ``spike_factor`` × rolling mean), and logs everything
          to both the Python logger and wandb.
        - Non-Goals: Does not stop training. The operator or EarlyStopping decides.

    Parameters
    ----------
    window_size : int, default 100
        Number of recent batch losses to keep in the rolling buffer.
    spike_factor : float, default 5.0
        A batch loss exceeding ``spike_factor * rolling_mean`` is flagged as a spike.
    instability_cv : float, default 0.5
        Coefficient of variation (std / mean) above this threshold flags the
        training as oscillating.
    log_every_n_epochs : int, default 1
        How often to emit the epoch-level summary.
    """

    def __init__(
        self,
        window_size: int = 100,
        spike_factor: float = 5.0,
        instability_cv: float = 0.5,
        log_every_n_epochs: int = 1,
    ):
        super().__init__()
        self.window_size = window_size
        self.spike_factor = spike_factor
        self.instability_cv = instability_cv
        self.log_every_n_epochs = log_every_n_epochs

        self._buffer: deque[float] = deque(maxlen=window_size)
        self._epoch_losses: list[float] = []
        self._spikes_this_epoch: int = 0

    # -- Batch hook: accumulate losses --------------------------------------

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is None:
            return

        val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
        if np.isnan(val) or np.isinf(val):
            return  # NaN/Inf handled by NaNDetectionCallback

        # Spike detection against rolling mean
        if len(self._buffer) >= 10:
            rolling_mean = np.mean(self._buffer)
            if rolling_mean > 0 and val > self.spike_factor * rolling_mean:
                self._spikes_this_epoch += 1
                logger.warning(
                    f"[Epoch {trainer.current_epoch}, batch {batch_idx}] "
                    f"Loss spike: {val:.4f} vs rolling mean {rolling_mean:.4f} "
                    f"({val / rolling_mean:.1f}×)"
                )

        self._buffer.append(val)
        self._epoch_losses.append(val)

    # -- Epoch hook: summary statistics -------------------------------------

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            self._epoch_losses.clear()
            self._spikes_this_epoch = 0
            return

        if not self._epoch_losses:
            return

        arr = np.array(self._epoch_losses)
        epoch_mean = float(arr.mean())
        epoch_std = float(arr.std())
        epoch_min = float(arr.min())
        epoch_max = float(arr.max())
        cv = epoch_std / epoch_mean if epoch_mean > 0 else 0.0

        # Determine verdict
        verdicts = []
        if self._spikes_this_epoch > 0:
            verdicts.append(f"⚠️  {self._spikes_this_epoch} spike(s)")
        if cv > self.instability_cv:
            verdicts.append(f"⚠️  oscillating (CV={cv:.2f})")
        status = " | ".join(verdicts) if verdicts else "✅ stable"

        # Push scalars to PL logger (wandb) if available
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    "loss_stability/epoch_mean": epoch_mean,
                    "loss_stability/epoch_std": epoch_std,
                    "loss_stability/epoch_min": epoch_min,
                    "loss_stability/epoch_max": epoch_max,
                    "loss_stability/cv": cv,
                    "loss_stability/spikes": self._spikes_this_epoch,
                },
                step=trainer.global_step,
            )

        logger.info(
            f"[Epoch {trainer.current_epoch}] Loss {status} | "
            f"mean={epoch_mean:.4f}, std={epoch_std:.4f}, "
            f"range=[{epoch_min:.4f}, {epoch_max:.4f}], CV={cv:.3f}, "
            f"spikes={self._spikes_this_epoch}"
        )

        # Reset per-epoch accumulators (rolling buffer persists across epochs)
        self._epoch_losses.clear()
        self._spikes_this_epoch = 0


# ---------------------------------------------------------------------------
# Epoch Timing & ETA
# ---------------------------------------------------------------------------


class EpochTimingCallback(Callback):
    """
    Wall-clock timer with epoch duration tracking and remaining-time estimation.

    Intent Contract:
        - Purpose: Give the operator real-time visibility into how long each epoch
          takes and when the run will finish. Essential for long sweeps on shared
          GPU clusters where deciding whether to kill a slow run saves money.
        - Guarantees: Logs epoch wall-clock duration in human-readable format,
          maintains a rolling average, and estimates time-to-completion based on
          ``trainer.max_epochs``. Pushes duration to wandb.
        - Non-Goals: Does not modify training behaviour.

    Parameters
    ----------
    log_every_n_epochs : int, default 1
        How often to log timing information.
    """

    def __init__(self, log_every_n_epochs: int = 1):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self._epoch_start: float = 0.0
        self._durations: list[float] = []

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        duration = time.perf_counter() - self._epoch_start
        self._durations.append(duration)

        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        avg_duration = np.mean(self._durations)
        current_epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs or 0
        remaining_epochs = max(0, max_epochs - current_epoch)
        eta_seconds = remaining_epochs * avg_duration

        # Push to PL logger (wandb) if available
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    "timing/epoch_seconds": duration,
                    "timing/avg_epoch_seconds": avg_duration,
                    "timing/eta_seconds": eta_seconds,
                },
                step=trainer.global_step,
            )

        logger.info(
            f"[Epoch {trainer.current_epoch}] "
            f"Duration: {self._format_time(duration)} "
            f"(avg {self._format_time(avg_duration)}) | "
            f"ETA: {self._format_time(eta_seconds)} "
            f"({remaining_epochs} epochs remaining)"
        )

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into a human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.0f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


# ---------------------------------------------------------------------------
# Y-Hat Bar (Prediction Calibration Monitor)
# ---------------------------------------------------------------------------


class YHatBarCallback(Callback):
    """
    Epoch-level mean-prediction calibration monitor in raw (un-transformed) space.

    Intent Contract:
        - Purpose: Track ``y_hat_bar`` — the mean prediction across all cells in
          raw space — as the primary calibration diagnostic for zero-inflated
          conflict forecasting. Complements MSLE, which rewards mild upward bias
          and cannot distinguish a well-calibrated model from an overpredicting one.
        - Guarantees: At the end of every ``log_every_n_epochs`` epoch, logs the
          overall mean and median raw prediction, the mean raw truth, the
          overprediction ratio (mean_pred / mean_truth), and per-channel means.
          All metrics are pushed to wandb.
        - Non-Goals: Does not stop training. Does not replace a proper calibration
          evaluation on held-out data.

    How it works:
        ``on_train_batch_end`` accumulates (prediction, truth) pairs in (B, T, C)
        form. ``on_train_epoch_end`` applies the appropriate inverse transform, then
        computes calibration statistics overall and per output channel.

        When truth is available, also computes event/peace series split diagnostics
        to detect Jensen's inequality bias amplification through sinh (or expm1)
        on high-variance event series.

    Parameters
    ----------
    target_scaler : str or None, default None
        Name of the target scaler used by the model. Determines the inverse
        transform applied to convert predictions back to raw space:
        - ``"AsinhTransform"`` → ``torch.sinh``
        - ``"LogTransform"`` or ``None`` → ``torch.expm1``
    non_zero_threshold : float, default 0.88
        Threshold in transformed space for classifying a series as "event"
        vs "peace". Default 0.88 ≈ asinh(1). Use 0.693 for log1p space.
    log_every_n_epochs : int, default 1
        How often to compute and log calibration statistics.
    """

    def __init__(
        self,
        target_scaler: str | None = None,
        non_zero_threshold: float = 0.88,
        log_every_n_epochs: int = 1,
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.non_zero_threshold = non_zero_threshold
        self._preds: list[torch.Tensor] = []
        self._truths: list[torch.Tensor] = []
        if target_scaler == "AsinhTransform":
            self._inverse_fn = torch.sinh
        else:
            self._inverse_fn = torch.expm1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Grab predictions from outputs dict or pl_module fallback
        preds = None
        if isinstance(outputs, dict) and "preds" in outputs:
            preds = outputs["preds"].detach().float()
        elif isinstance(outputs, dict) and "y_hat" in outputs:
            preds = outputs["y_hat"].detach().float()
        elif hasattr(pl_module, "last_predictions"):
            preds = pl_module.last_predictions.detach().float()

        # Grab truth from batch. Darts passes batch as (past_target, ..., future_target)
        # where future_target is the last element (a tuple/list) or a tensor directly.
        truth = None
        if preds is not None and batch is not None:
            try:
                future = batch[-1]
                if isinstance(future, (list, tuple)):
                    future = future[0]
                if isinstance(future, torch.Tensor):
                    truth = future.detach().float()
            except Exception:
                truth = None

        if preds is not None:
            # Normalise to (B, T, C): unsqueeze trailing dim if missing
            if preds.dim() == 2:
                preds = preds.unsqueeze(-1)
            self._preds.append(preds)

        if truth is not None:
            if truth.dim() == 2:
                truth = truth.unsqueeze(-1)
            self._truths.append(truth)

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            self._preds.clear()
            self._truths.clear()
            return

        if not self._preds:
            return

        # Shape: (N, T, C) on CPU
        all_preds = torch.cat(self._preds, dim=0).cpu()
        has_truth = len(self._truths) == len(self._preds)
        all_truths = torch.cat(self._truths, dim=0).cpu() if has_truth else None

        # Convert from transformed space to raw space — keep (N, T, C)
        raw_preds = self._inverse_fn(all_preds)
        raw_truths = self._inverse_fn(all_truths) if all_truths is not None else None

        # ── Overall stats ──────────────────────────────────────────────
        y_hat_bar_mean = raw_preds.mean().item()
        y_hat_bar_median = raw_preds.median().item()

        metrics: dict[str, float] = {
            "y_hat_bar/mean": y_hat_bar_mean,
            "y_hat_bar/median": y_hat_bar_median,
        }
        log_parts = [
            f"mean={y_hat_bar_mean:.2f}",
            f"median={y_hat_bar_median:.2f}",
        ]

        if raw_truths is not None:
            y_bar_mean = raw_truths.mean().item()
            ratio = y_hat_bar_mean / y_bar_mean if y_bar_mean > 1e-6 else float("nan")
            metrics["y_hat_bar/y_bar_mean"] = y_bar_mean
            metrics["y_hat_bar/ratio"] = ratio
            log_parts += [f"y_bar={y_bar_mean:.2f}", f"ratio={ratio:.2f}x"]

        # ── Per-channel stats ──────────────────────────────────────────
        # raw_preds shape: (N, T, C). Iterate channels, log as ch_0, ch_1, ...
        n_channels = raw_preds.size(-1)
        ch_parts = []
        for c in range(n_channels):
            ch_pred = raw_preds[:, :, c]
            ch_mean = ch_pred.mean().item()
            metrics[f"y_hat_bar/ch_{c}"] = ch_mean
            if raw_truths is not None:
                ch_truth = raw_truths[:, :, c]
                ch_y_bar = ch_truth.mean().item()
                ch_ratio = ch_mean / ch_y_bar if ch_y_bar > 1e-6 else float("nan")
                metrics[f"y_hat_bar/ch_{c}_y_bar"] = ch_y_bar
                metrics[f"y_hat_bar/ch_{c}_ratio"] = ch_ratio
                ch_parts.append(f"ch{c}={ch_mean:.2f}(×{ch_ratio:.2f})")
            else:
                ch_parts.append(f"ch{c}={ch_mean:.2f}")

        if n_channels > 1:
            log_parts.append("[" + " ".join(ch_parts) + "]")

        # ── Event/peace split (Jensen's inequality diagnostic) ─────────
        # Detect systematic raw-space bias on event series, which would
        # indicate sinh (or expm1) convexity amplifying residual level
        # error through RevIN denormalization.
        if raw_truths is not None:
            # A series is "event" if any timestep in truth exceeds threshold
            # all_truths shape: (N, T, C) in transformed space
            is_event_series = (
                torch.abs(all_truths) > self.non_zero_threshold
            ).any(dim=1).any(dim=1)  # (N,) bool

            n_event = is_event_series.sum().item()
            n_peace = (~is_event_series).sum().item()

            if n_event > 0:
                event_raw_pred_mean = raw_preds[is_event_series].mean().item()
                event_raw_truth_mean = raw_truths[is_event_series].mean().item()
                event_bias = event_raw_pred_mean - event_raw_truth_mean
                event_ratio = (
                    event_raw_pred_mean / event_raw_truth_mean
                    if event_raw_truth_mean > 1e-6
                    else float("nan")
                )
                # Per-series σ in transformed space (proxy for RevIN σ)
                event_sigma = all_truths[is_event_series].std(dim=1).mean().item()

                metrics["y_hat_bar/event_mean"] = event_raw_pred_mean
                metrics["y_hat_bar/event_truth"] = event_raw_truth_mean
                metrics["y_hat_bar/event_bias"] = event_bias
                metrics["y_hat_bar/event_ratio"] = event_ratio
                metrics["y_hat_bar/event_sigma"] = event_sigma
                metrics["y_hat_bar/n_event_series"] = n_event

                log_parts.append(
                    f"event({n_event}): bias={event_bias:+.2f} "
                    f"ratio={event_ratio:.2f}x σ={event_sigma:.2f}"
                )

            if n_peace > 0:
                peace_raw_pred_mean = raw_preds[~is_event_series].mean().item()
                metrics["y_hat_bar/peace_mean"] = peace_raw_pred_mean
                metrics["y_hat_bar/n_peace_series"] = n_peace

        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

        logger.info(
            f"[Epoch {trainer.current_epoch}] Calibration | " + ", ".join(log_parts)
        )

        self._preds.clear()
        self._truths.clear()

