import time
import math
import torch
import numpy as np
import logging
from collections import deque, defaultdict
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NaN Detection
# ---------------------------------------------------------------------------


class _PatchedTrainingStep:
    """
    Top-level callable that replaces ``pl_module.training_step``.

    Defined at module level (not as a local function) so that ``torch.save``
    can pickle it — pickle requires callables to be importable by module + name,
    which local/nested functions are not.

    Exposes ``__code__`` (delegating to ``__call__.__code__``) so that
    PyTorch Lightning's ``is_overridden`` check — which does
    ``instance_attr.__code__ != parent_attr.__code__`` — sees this as a
    genuinely overridden method rather than raising ``AttributeError``.
    """

    def __init__(self, pl_module):
        self.pl_module = pl_module

    @property
    def __code__(self):
        return self.__call__.__code__

    def __call__(self, train_batch, batch_idx):
        pl_module = self.pl_module
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

        pl_module._original_training_step = pl_module.training_step
        pl_module.training_step = _PatchedTrainingStep(pl_module)
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
    explode_threshold : float, default 500.0
        Minimum gradient norm above which gradients are flagged as exploding.
    """

    def __init__(
        self,
        log_every_n_epochs: int = 1,
        warn_threshold: float = 1e-7,
        explode_threshold: float = 500.0,
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.warn_threshold = warn_threshold
        self.explode_threshold = explode_threshold
        self._last_logged_epoch = -1
        self._last_grad_stats = None

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Capture gradient norms BEFORE optimizer.step() zeroes them.

        PyTorch Lightning's execution order is:
            backward → on_before_optimizer_step → optimizer.step → zero_grad
        So this is the only hook where param.grad is populated.
        We snapshot the stats here and log them later in on_train_epoch_end.
        """
        # Only snapshot on the last batch of each logged epoch
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
                    zero_count += 1
                else:
                    grad_norms.append(norm)

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

        # Overwrite each batch — the last batch of the epoch wins
        self._last_grad_stats = {
            "stats": stats,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "zero_count": zero_count,
            "total_params": total_params,
        }

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        if self._last_grad_stats is None:
            return

        s = self._last_grad_stats
        stats = s["stats"]
        nan_count = s["nan_count"]
        inf_count = s["inf_count"]
        zero_count = s["zero_count"]
        total_params = s["total_params"]

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
        metrics = {
            "grad_norm/min": stats["min"],
            "grad_norm/max": stats["max"],
            "grad_norm/mean": stats["mean"],
            "grad_norm/median": stats["median"],
            "grad_norm/nan_count": nan_count,
            "grad_norm/inf_count": inf_count,
            "grad_norm/zero_count": zero_count,
        }

        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

        logger.info(
            f"[Epoch {trainer.current_epoch}] Gradients {status} | "
            f"norm: min={stats['min']:.2e}, max={stats['max']:.2e}, "
            f"mean={stats['mean']:.2e}, median={stats['median']:.2e} | "
            f"zero={zero_count}/{total_params}"
        )

        if nan_count > 0 or inf_count > 0:
            logger.error(
                f"[Epoch {trainer.current_epoch}] Stopping: {nan_count} NaN + "
                f"{inf_count} Inf gradients detected — run is unrecoverable."
            )
            trainer.should_stop = True


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
            # Skip loss-criterion parameters (e.g. SpotlightLoss channel_log_var):
            # these are learnable loss hyperparameters, not model weights. They are
            # zero-initialized by design (e.g. log-variance 0 = equal channel
            # weighting) and only the training-criterion copy receives gradients,
            # so a zero norm on the criterion/val_criterion copies is expected and
            # is not a layer collapse.
            if "criterion" in name:
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
# RevIN Monitor
# ---------------------------------------------------------------------------


class RevINMonitorCallback(Callback):
    """
    Epoch-level RevIN statistics monitor.

    Intent Contract:
        - Purpose: Track the per-series μ and σ_eff values computed by the
          (potentially patched) RINorm module. Surfaces whether the balanced
          conditioning patch is working as expected — σ_eff should be small
          for high-μ entities and ≈ σ_raw for low-μ entities.
        - Guarantees: Logs per-epoch RevIN statistics (μ range, σ_eff range,
          max z-magnitude, compression ratio) to the PL logger (wandb).
          Silently no-ops if RevIN is disabled (pl_module.rin is None).
        - Non-Goals: Does not modify RevIN behavior or stop training.

    Parameters
    ----------
    log_every_n_epochs : int, default 5
        How often to log RevIN statistics.
    """

    def __init__(self, log_every_n_epochs: int = 5):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only capture on the last batch of the epoch to avoid overhead.
        # We store the most recent batch's RevIN state for epoch-end logging.
        rin = getattr(pl_module, "rin", None)
        if rin is None:
            return

        mean = getattr(rin, "mean", None)
        stdev = getattr(rin, "stdev", None)
        if mean is not None and stdev is not None:
            self._last_mean = mean.detach()
            self._last_stdev = stdev.detach()

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        rin = getattr(pl_module, "rin", None)
        if rin is None:
            return

        mean = getattr(self, "_last_mean", None)
        stdev = getattr(self, "_last_stdev", None)
        if mean is None or stdev is None:
            return

        # Raw-space RevIN: self.mean and self.stdev are in raw count space
        mu_flat = mean.flatten()
        sigma_flat = stdev.flatten()

        mu_min = mu_flat.min().item()
        mu_max = mu_flat.max().item()
        mu_mean = mu_flat.mean().item()

        sigma_min = sigma_flat.min().item()
        sigma_max = sigma_flat.max().item()
        sigma_mean = sigma_flat.mean().item()

        # For raw-space RevIN, Jensen bias ≈ exp(σ_ẑ²/2) where σ_ẑ is
        # the z-space prediction variance (MC dropout, ~0.3). This is
        # independent of σ_raw/μ_raw — structurally eliminated.
        # Log sigma_raw/mu_raw ratio as a diagnostic for data distribution.
        ratio_max = sigma_max / max(abs(mu_max), 1.0)

        metrics = {
            "revin/mu_raw_min": mu_min,
            "revin/mu_raw_max": mu_max,
            "revin/mu_raw_mean": mu_mean,
            "revin/sigma_raw_min": sigma_min,
            "revin/sigma_raw_max": sigma_max,
            "revin/sigma_raw_mean": sigma_mean,
            "revin/cv_max": ratio_max,
        }

        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

        logger.info(
            f"[Epoch {trainer.current_epoch}] RevIN (raw-space) | "
            f"μ_raw∈[{mu_min:.1f}, {mu_max:.1f}] mean={mu_mean:.1f} | "
            f"σ_raw∈[{sigma_min:.3f}, {sigma_max:.1f}] mean={sigma_mean:.2f} | "
            f"CV_max={ratio_max:.2f}"
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
        variance_floor: float = 0.2,
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


# ---------------------------------------------------------------------------
# Validation MSLE / MSE
# ---------------------------------------------------------------------------


class ValMetricsCallback(Callback):
    """
    Epoch-level MSLE and MSE on the validation set in raw (un-transformed) space.

    Intent Contract:
        - Purpose: Provide calibration-quality metrics (MSLE, RMSLE, MSE) on the
          held-out validation fold so that overprediction bias visible in training
          calibration is also tracked on unseen windows.
                - Mechanism: ``on_validation_batch_end`` runs a fresh ``no_grad`` forward
                    pass using the batch that was just processed.  The model is already in
                    ``eval()`` mode, so RevIN and dropout behave identically to what the
                    ``validation_step`` already computed — the extra overhead is one
                    forward pass per val batch with no gradient tape.
                - Guarantees: Logs ``val_metrics/MSLE``, ``val_metrics/RMSLE``,
                    ``val_metrics/MSE``, ``val_metrics/GM_MSLE_MSE``,
                    ``val_metrics/PARETO_GATE`` (strict both-must-improve), and
                    ``val_metrics/BALANCE_GM_RATIO`` (trade-off balance) at epoch end.
        - Non-Goals: Does not recompute val_loss or interfere with early stopping.

    Parameters
    ----------
    target_scaler : str, optional
        ``"AsinhTransform"`` → inverse = ``sinh``; anything else → ``expm1``.
    log_every_n_epochs : int
        Skip logging on epochs that are not multiples of this value (still
        clears the accumulators to avoid stale data).
    """

    def __init__(
        self,
        target_scaler: str | None = None,
        log_every_n_epochs: int = 1,
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self._preds: list = []
        self._truths: list = []
        self._inverse_fn = (
            torch.sinh if target_scaler == "AsinhTransform" else torch.expm1
        )
        self._best_msle = float("inf")
        self._best_mse = float("inf")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if batch is None:
            return
        try:
            # Resolve input slice: use model's own input_tuple_size when available,
            # else mirror the training_step patch which excludes sample_weight (-2)
            # and future_target (-1).
            n_inputs = getattr(pl_module, "input_tuple_size", None)
            if n_inputs is not None:
                input_batch = batch[:n_inputs]
            else:
                input_batch = batch[:-2]

            target = batch[-1]

            with torch.no_grad():
                output = pl_module._produce_train_output(input_batch)

            preds = output.detach().float()
            if preds.dim() == 4:
                preds = preds[..., 0]  # point forecast (first likelihood param)
            if preds.dim() == 2:
                preds = preds.unsqueeze(-1)

            truth = target.detach().float()
            if truth.dim() == 2:
                truth = truth.unsqueeze(-1)

            self._preds.append(preds.cpu())
            self._truths.append(truth.cpu())
        except Exception as exc:
            logger.debug(
                f"ValMetricsCallback: skipped batch {batch_idx} — {exc}"
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        preds_buf = self._preds[:]
        truths_buf = self._truths[:]
        self._preds.clear()
        self._truths.clear()

        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        if not preds_buf or not truths_buf:
            return

        all_preds = torch.cat(preds_buf, dim=0).float()   # (N, T, C)
        all_truths = torch.cat(truths_buf, dim=0).float()  # (N, T, C)

        # Inverse-transform to raw (count) space and floor at 0
        raw_preds = self._inverse_fn(all_preds).clamp(min=0.0)
        raw_truths = self._inverse_fn(all_truths).clamp(min=0.0)

        # MSLE: mean((log(1 + y_hat) - log(1 + y))^2)
        log_pred = torch.log1p(raw_preds)
        log_true = torch.log1p(raw_truths)
        msle = ((log_pred - log_true) ** 2).mean().item()
        rmsle = msle ** 0.5

        # MSE in raw space
        mse = ((raw_preds - raw_truths) ** 2).mean().item()

        # Composite objective in raw/log space trade-off.
        gm_msle_mse = (msle * mse) ** 0.5

        # Ratios are computed against BEST PREVIOUS values (before update).
        msle_ref = max(self._best_msle, 1e-12)
        mse_ref = max(self._best_mse, 1e-12)
        ratio_msle = msle / msle_ref if msle_ref < float("inf") else 1.0
        ratio_mse = mse / mse_ref if mse_ref < float("inf") else 1.0

        # Strict "both must not regress" gate: < 1 only when both are better.
        pareto_gate = max(ratio_msle, ratio_mse)
        # Balanced trade-off gate: allows one-up one-down if joint GM improves.
        balance_gm_ratio = (ratio_msle * ratio_mse) ** 0.5

        metrics = {
            "val_metrics/MSLE": msle,
            "val_metrics/RMSLE": rmsle,
            "val_metrics/MSE": mse,
            "val_metrics/GM_MSLE_MSE": gm_msle_mse,
            "val_metrics/PARETO_GATE": pareto_gate,
            "val_metrics/BALANCE_GM_RATIO": balance_gm_ratio,
        }

        # Update per-metric bests after ratio computation.
        self._best_msle = min(self._best_msle, msle)
        self._best_mse = min(self._best_mse, mse)

        # Log to PyTorch Lightning callback_metrics so EarlyStopping/checkpointing can monitor them
        for k, v in metrics.items():
            pl_module.log(k, v, on_epoch=True, prog_bar=True, logger=False)

        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

        logger.info(
            f"[Epoch {trainer.current_epoch}] Val Metrics | "
            f"MSLE={msle:.5f}  RMSLE={rmsle:.4f}  MSE={mse:.2f}  "
            f"GM={gm_msle_mse:.4f}  Pareto={pareto_gate:.4f}  Balance={balance_gm_ratio:.4f}"
        )


# ---------------------------------------------------------------------------
# Input Batch Monitor
# ---------------------------------------------------------------------------


class InputBatchMonitorCallback(Callback):
    """
    Logs key statistics about the input batch data.

    This helps correlate training dynamics (e.g., loss spikes) with the
    composition of the data being processed.
    """

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Log statistics of the training batch."""
        if not hasattr(pl_module, "log"):
            return

        # For Darts PL modules, the batch is a tuple/list.
        # (past_target, past_covariates, ..., future_target)
        past_target = batch[0]
        if past_target is None:
            return

        # Assuming past_target is a tensor of shape (batch_size, sequence_length, features)
        # Flatten over sequence_length and features for batch-wide stats
        flat_targets = past_target.view(past_target.size(0), -1)

        # --- Key Metrics ---
        # Sparsity: fraction of zero values
        sparsity = (flat_targets == 0).float().mean()

        # Magnitude stats
        mean_val = flat_targets.mean()
        max_val = flat_targets.max()
        
        # Number of "event" series (at least one non-zero value)
        event_series_mask = torch.any(flat_targets != 0, dim=1)
        n_event_series = event_series_mask.sum()
        
        # Log metrics to WandB
        pl_module.log(
            "input_batch/sparsity",
            sparsity,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        pl_module.log(
            "input_batch/mean",
            mean_val,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        pl_module.log(
            "input_batch/max",
            max_val,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        pl_module.log(
            "input_batch/n_event_series",
            n_event_series,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )


# ---------------------------------------------------------------------------
# Loss Component Monitor
# ---------------------------------------------------------------------------


class LossComponentCallback(Callback):
    """
    Per-channel composite-loss component auditor (SpotlightLossLogcosh).

    Intent Contract:
                - Purpose: Make the internal structure of the multi-task loss visible —
                    for each target channel, how big its shape / level / spectral terms
                    are, what fraction of that channel each term contributes, the
                    per-channel active weight(s), and each channel's weighted
                    contribution to the total loss. This surfaces both the within-channel
                    term mix and whether cross-channel budgeting is behaving as intended.
        - Guarantees: Reads only detached, parameter-free telemetry stashed on
          ``pl_module.train_criterion._last_components`` (never touches the
          graph), accumulates per batch, and pushes per-epoch means to wandb.
          No-op for losses that do not expose ``_last_components``.
        - Non-Goals: Does not modify the loss or training.

    wandb keys (per channel c, 0-indexed = sb/ns/os):
        loss_components/ch_{c}/{shape,level,spec}            raw term magnitudes
        loss_components/ch_{c}/frac_{shape,level,spec}       within-channel share
        loss_components/ch_{c}/weight                         primary active weight
        loss_components/ch_{c}/cal_score                      secondary active weight
        loss_components/ch_{c}/cal_ratio                      calibration ratio (loss-defined)
        loss_components/ch_{c}/contribution                   weighted channel contribution
        loss_components/ch_{c}/budget_won                     contribution share in [0,1]
        loss_components/contribution_spread                   max/min contribution

    Notes:
        - Field names are intentionally generic to stay backward-compatible
          across loss revisions.
        - For uncertainty-weighted SpotlightLossLogcosh:
            * ``weight`` = shape precision term (0.5 * exp(-s_shape))
            * ``cal_score`` = level precision term (0.5 * exp(-s_level))
            * ``cal_ratio`` = event calibration ratio (sum |y_hat| / sum |y|)
    """

    def __init__(self):
        super().__init__()
        self._buf = defaultdict(list)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        crit = getattr(pl_module, "train_criterion", None)
        comp = getattr(crit, "_last_components", None) if crit is not None else None
        if not comp:
            return

        shape = comp.get("shape", [])
        level = comp.get("level", [])
        spec = comp.get("spec", [])
        C = len(shape)
        if C == 0:
            return
        weight = comp.get("weight", [1.0] * C)
        ema = comp.get("ema", [float("nan")] * C)
        contribution = comp.get("contribution")
        cal_ratio = comp.get("cal_ratio", [1.0] * C)
        cal_score = comp.get("cal_score", [1.0] * C)
        gates = comp.get("gates", [1.0] * C)

        contrib_sum = None
        if contribution is not None and len(contribution) == C:
            contrib_sum = sum(abs(x) for x in contribution) + 1e-12

        for c in range(C):
            tot_c = shape[c] + level[c] + spec[c]
            denom = tot_c if abs(tot_c) > 1e-12 else 1e-12
            self._buf[f"loss_components/ch_{c}/shape"].append(shape[c])
            self._buf[f"loss_components/ch_{c}/level"].append(level[c])
            self._buf[f"loss_components/ch_{c}/spec"].append(spec[c])
            self._buf[f"loss_components/ch_{c}/frac_shape"].append(shape[c] / denom)
            self._buf[f"loss_components/ch_{c}/frac_level"].append(level[c] / denom)
            self._buf[f"loss_components/ch_{c}/frac_spec"].append(spec[c] / denom)
            self._buf[f"loss_components/ch_{c}/weight"].append(weight[c])
            self._buf[f"loss_components/ch_{c}/cal_ratio"].append(cal_ratio[c])
            self._buf[f"loss_components/ch_{c}/cal_score"].append(cal_score[c])
            self._buf[f"loss_components/ch_{c}/gate_weight"].append(gates[c])
            if contrib_sum is not None:
                self._buf[f"loss_components/ch_{c}/budget_won"].append(
                    abs(contribution[c]) / contrib_sum
                )
            else:
                # Backward-compatible fallback when only gate-like telemetry exists.
                self._buf[f"loss_components/ch_{c}/budget_won"].append(gates[c] / C)
            if not math.isnan(ema[c]):
                self._buf[f"loss_components/ch_{c}/ema"].append(ema[c])
            if contribution is not None:
                self._buf[f"loss_components/ch_{c}/contribution"].append(contribution[c])

        # How equal are the channel contributions after balancing? ~1.0 means
        # balanced; ≫1 means one channel dominates the joint objective.
        if contribution is not None and len(contribution) > 1:
            cmax = max(contribution)
            cmin = min(contribution)
            self._buf["loss_components/contribution_spread"].append(
                cmax / (abs(cmin) + 1e-12)
            )

    def on_train_epoch_end(self, trainer, pl_module):
        if not self._buf:
            return

        metrics = {k: float(np.mean(v)) for k, v in self._buf.items() if v}

        # Mean shape-vs-level active weight ratio across channels. This is a
        # compact monitor for uncertainty-weight drift: >1 means shape gets
        # more budget than level; <1 means level dominates.
        n_ch = sum(1 for k in metrics if k.endswith("/shape"))
        if n_ch > 0:
            ratios = []
            for c in range(n_ch):
                w_shape = metrics.get(f"loss_components/ch_{c}/weight")
                w_level = metrics.get(f"loss_components/ch_{c}/cal_score")
                if w_shape is None or w_level is None:
                    continue
                ratios.append(w_shape / (abs(w_level) + 1e-12))
            if ratios:
                metrics["loss_components/shape_level_weight_ratio"] = float(np.mean(ratios))

        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

        # Concise per-channel summary line for the console.
        parts = []
        for c in range(n_ch):
            sh = metrics.get(f"loss_components/ch_{c}/shape", float("nan"))
            lv = metrics.get(f"loss_components/ch_{c}/level", float("nan"))
            sp = metrics.get(f"loss_components/ch_{c}/spec", float("nan"))
            w_shape = metrics.get(f"loss_components/ch_{c}/weight", float("nan"))
            w_level = metrics.get(f"loss_components/ch_{c}/cal_score", float("nan"))
            cal = metrics.get(f"loss_components/ch_{c}/cal_ratio", float("nan"))
            won = metrics.get(f"loss_components/ch_{c}/budget_won", float("nan"))
            parts.append(
                f"ch{c}[sh={sh:.2f} lv={lv:.2f} sp={sp:.2f} w_sh={w_shape:.2f} w_lv={w_level:.2f} cal={cal:.2f} won_frac={won:.2%}]"
            )
        spread = metrics.get("loss_components/contribution_spread")
        spread_str = f" | spread={spread:.2f}" if spread is not None else ""
        ratio = metrics.get("loss_components/shape_level_weight_ratio")
        ratio_str = f" | w_sh/w_lv={ratio:.2f}" if ratio is not None else ""
        logger.info(
            f"[Epoch {trainer.current_epoch}] LossComponents | "
            + " ".join(parts)
            + spread_str
            + ratio_str
        )

        self._buf.clear()


# ---------------------------------------------------------------------------
# Loss Gradient Diagnostics
# ---------------------------------------------------------------------------


class LossGradientDiagnosticsCallback(Callback):
    """
    Detailed per-term gradient diagnostics for SpotlightLossLogcosh.

    Combines three diagnostic sources each batch/epoch:

    1. **Loss telemetry** — reads extended ``_last_components`` keys written by
       ``SpotlightLossLogcosh.forward``:
       - DRO weight distribution per channel (mean, std, max, fraction > 1).
         A high max (> 5) or high frac_up means DRO is concentrating gradient
         on a few cells; a low mean (≪ 1) after renorm means DRO is mostly idle.
       - Level gap magnitude per channel.  Decreasing gap_mean = level converging.
         Stuck gap_mean = level not learning (gradient budget stolen or LR too low).
       - Shape DC leak per channel.  ``shape_dc`` is the batch-mean of the
         per-series gated mean of e_shape.  Should be ≈ 0 for a DC-neutral shape
         term; a large value indicates gate weighting is reintroducing a DC bias
         that fights the level anchor.
       - Event cell fraction per channel (sparsity diagnostic).

    2. **Input-gradient decomposition** — reads ``_last_input_grad`` tensor
       (``d(loss)/d(y_pred)``), set by a backward hook registered in
       ``SpotlightLossLogcosh.forward``:
       - DC fraction: ``|mean_t(grad)| / (|mean_t(grad)| + RMS_AC(grad))``.
         Should be ≈ 0 for a pure shape loss; will be > 0 when level dominates.
         Target for this loss design: dc_frac ≈ 0.30–0.60 (neither starves).
       - AC fraction: complement of DC fraction.
       - Sign fraction: fraction of gradient entries > 0.  < 0.5 means the loss
         is predominantly pushing predictions DOWN (under-prediction correction).
       - DC and AC gradient magnitudes per channel (absolute scale).

    3. **Output-layer gradient** — captures ``fc_out.weight.grad`` in
       ``on_before_optimizer_step``:
       - Row-norm mean/max/std over the output projection's weight matrix.
       - Sign fraction of the fc_out gradient (direction of output update).

    wandb keys:
        ``grad_diag/ch_{c}/{dro_w_mean, dro_w_std, dro_w_max, dro_frac_up,
        event_frac, level_gap_mean, level_gap_max, shape_dc, grad_dc_frac,
        grad_ac_frac, grad_sign_frac, grad_dc_mag, grad_ac_mag, grad_total_norm}``
        ``grad_diag/{fcout_grad_norm_mean, fcout_grad_norm_max, fcout_grad_norm_std,
        fcout_grad_sign_frac}``
    """

    def __init__(self, log_every_n_epochs: int = 1):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self._buf: dict = defaultdict(list)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        crit = getattr(pl_module, "train_criterion", None)
        comp = getattr(crit, "_last_components", None) if crit is not None else None
        if not comp:
            return

        C = len(comp.get("shape", []))
        if C == 0:
            return

        # ── 1. Loss telemetry from _last_components ─────────────────────
        for key in (
            "dro_w_mean", "dro_w_std", "dro_w_max", "dro_frac_up",
            "event_frac", "level_gap_mean", "level_gap_max", "shape_dc",
            "level_gap_ev_mean", "level_gap_ev_max", "level_gap_sat",
            "shape_level_ratio",
            "gap_v13_mean", "gap_v13_max", "dilution",
            "hit_frac", "false_alarm_of_mask", "missed_frac",
            "mean_pred_lm", "mean_true_lm", "lm_per_series",
            "e_fa_mean", "e_me_mean",
        ):
            vals = comp.get(key)
            if vals is None:
                continue
            for c, v in enumerate(vals):
                self._buf[f"grad_diag/ch_{c}/{key}"].append(v)

        # ── 2. Input-gradient decomposition ─────────────────────────────
        # _last_input_grad = d(loss)/d(y_pred), shape (B,T) or (B,T,C).
        # Decompose per-channel into:
        #   DC = mean_over_time component  → driven by level (uniform gradient)
        #   AC = demeaned component        → driven by shape (zero-mean gradient)
        grad = getattr(crit, "_last_input_grad", None)
        if grad is not None:
            g = grad.float().cpu()
            if g.dim() == 2:
                g = g.unsqueeze(-1)              # → (B, T, 1)

            dc = g.mean(dim=1, keepdim=True)     # (B, 1, C) — DC per series
            ac = g - dc                          # (B, T, C) — AC per series

            # Per-series norm magnitudes
            dc_norm  = dc.squeeze(1).abs()                        # (B, C)
            ac_norm  = ac.norm(dim=1) / (g.shape[1] ** 0.5 + 1e-8)  # (B, C) RMS
            tot_norm = dc_norm + ac_norm + 1e-12

            dc_frac   = (dc_norm / tot_norm).mean(dim=0)          # (C,)
            ac_frac   = (ac_norm / tot_norm).mean(dim=0)          # (C,)
            sign_frac = (g > 0).float().mean(dim=(0, 1))          # (C,)
            dc_mag    = dc_norm.mean(dim=0)                       # (C,)
            ac_mag    = ac_norm.mean(dim=0)                       # (C,)
            total_g   = g.norm(dim=1).mean(dim=0)                 # (C,)

            for c in range(min(g.shape[-1], C)):
                self._buf[f"grad_diag/ch_{c}/grad_dc_frac"].append(dc_frac[c].item())
                self._buf[f"grad_diag/ch_{c}/grad_ac_frac"].append(ac_frac[c].item())
                self._buf[f"grad_diag/ch_{c}/grad_sign_frac"].append(sign_frac[c].item())
                self._buf[f"grad_diag/ch_{c}/grad_dc_mag"].append(dc_mag[c].item())
                self._buf[f"grad_diag/ch_{c}/grad_ac_mag"].append(ac_mag[c].item())
                self._buf[f"grad_diag/ch_{c}/grad_total_norm"].append(total_g[c].item())

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Capture output-layer (fc_out) gradient statistics before weight update."""
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        # Try fc_out directly (TSMixer), then fall back to last Linear with a gradient.
        out_layer = getattr(pl_module, "fc_out", None)
        if (
            out_layer is None
            or not hasattr(out_layer, "weight")
            or out_layer.weight.grad is None
        ):
            out_layer = None
            for _, mod in reversed(list(pl_module.named_modules())):
                if (
                    isinstance(mod, torch.nn.Linear)
                    and hasattr(mod, "weight")
                    and mod.weight.grad is not None
                ):
                    out_layer = mod
                    break

        if out_layer is None:
            return

        g = out_layer.weight.grad.detach().float()   # (out_features, in_features)
        row_norms = g.norm(dim=1)                    # (out_features,)
        self._buf["grad_diag/fcout_grad_norm_mean"].append(row_norms.mean().item())
        self._buf["grad_diag/fcout_grad_norm_max"].append(row_norms.max().item())
        self._buf["grad_diag/fcout_grad_norm_std"].append(row_norms.std().item())
        self._buf["grad_diag/fcout_grad_sign_frac"].append((g > 0).float().mean().item())

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            self._buf.clear()
            return
        if not self._buf:
            return

        metrics = {k: float(np.mean(v)) for k, v in self._buf.items() if v}

        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

        # Determine number of channels from keys
        C = max(
            (int(k.split("/ch_")[1].split("/")[0]) for k in metrics if "/ch_" in k),
            default=-1,
        ) + 1

        if C > 0:
            parts = []
            for c in range(C):
                pfx = f"grad_diag/ch_{c}"
                dc_frac   = metrics.get(f"{pfx}/grad_dc_frac",    float("nan"))
                ac_frac   = metrics.get(f"{pfx}/grad_ac_frac",    float("nan"))
                sign_frac = metrics.get(f"{pfx}/grad_sign_frac",  float("nan"))
                dc_mag    = metrics.get(f"{pfx}/grad_dc_mag",     float("nan"))
                ac_mag    = metrics.get(f"{pfx}/grad_ac_mag",     float("nan"))
                dro_mean  = metrics.get(f"{pfx}/dro_w_mean",      float("nan"))
                dro_max   = metrics.get(f"{pfx}/dro_w_max",       float("nan"))
                dro_fup   = metrics.get(f"{pfx}/dro_frac_up",     float("nan"))
                gap_mean  = metrics.get(f"{pfx}/level_gap_mean",  float("nan"))
                gap_ev    = metrics.get(f"{pfx}/level_gap_ev_mean", float("nan"))
                gap_sat   = metrics.get(f"{pfx}/level_gap_sat",    float("nan"))
                gap_v13   = metrics.get(f"{pfx}/gap_v13_mean",     float("nan"))
                dilution  = metrics.get(f"{pfx}/dilution",         float("nan"))
                sl_ratio  = metrics.get(f"{pfx}/shape_level_ratio", float("nan"))
                hit_frac  = metrics.get(f"{pfx}/hit_frac",         float("nan"))
                fa_mask   = metrics.get(f"{pfx}/false_alarm_of_mask", float("nan"))
                miss_frac = metrics.get(f"{pfx}/missed_frac",      float("nan"))
                e_fa      = metrics.get(f"{pfx}/e_fa_mean",        float("nan"))
                e_me      = metrics.get(f"{pfx}/e_me_mean",        float("nan"))
                shape_dc  = metrics.get(f"{pfx}/shape_dc",        float("nan"))
                ev_frac   = metrics.get(f"{pfx}/event_frac",      float("nan"))
                parts.append(
                    f"ch{c}["
                    f"dc%={dc_frac:.0%} ac%={ac_frac:.0%} sign↑={sign_frac:.2f} "
                    f"dcMag={dc_mag:.4f} acMag={ac_mag:.4f} | "
                    f"dro={dro_mean:.2f}±{dro_max:.1f}× ↑{dro_fup:.0%} | "
                    f"gap={gap_mean:.3f}/{gap_ev:.3f} sat={gap_sat:.0%} v13={gap_v13:.3f} dil={dilution:.2f} sl={sl_ratio:.2f} | "
                    f"hit={hit_frac:.0%} faM={fa_mask:.0%} miss={miss_frac:.0%} e_fa={e_fa:.3f} e_me={e_me:.3f} | "
                    f"shDC={shape_dc:.4f} ev={ev_frac:.2f}]"
                )
            fcout_norm = metrics.get("grad_diag/fcout_grad_norm_mean", float("nan"))
            fcout_sign = metrics.get("grad_diag/fcout_grad_sign_frac", float("nan"))
            logger.info(
                f"[Epoch {trainer.current_epoch}] GradDiag | "
                + " ".join(parts)
                + f" | fc_out[norm={fcout_norm:.4f} sign↑={fcout_sign:.2f}]"
            )

        self._buf.clear()

# ---------------------------------------------------------------------------
# Rich Loss Diagnostics (Visual Console Output)
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

if _HAS_RICH:
    _console = Console()


class RichLossDiagnosticsCallback(Callback):
    """
    Visual, color-coded console diagnostics for SpotlightLossLogcosh.

    Prints a rich table every N epochs with:
    - Shape vs Level balance (green/yellow/red)
    - Gap diagnostics (raw vs density-scaled)
    - Hájek weights and batch active fraction
    - Gradient direction on events vs non-events (green=up, red=down)
    - DRO weight distribution
    - Prediction statistics

    Install: pip install rich
    """

    def __init__(self, log_every_n_epochs: int = 1):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        crit = getattr(pl_module, "train_criterion", None)
        comp = getattr(crit, "_last_components", None) if crit is not None else None
        if not comp:
            return

        epoch = trainer.current_epoch
        n_ch = len(comp.get("shape", []))

        # Build the table
        table = Table(
            title=f"[bold cyan]🔬 SpotlightLoss Diagnostics — Epoch {epoch}[/]",
            show_header=True,
            header_style="bold cyan",
            border_style="blue",
        )
        table.add_column("Channel", style="bold", width=6)
        table.add_column("Shape", justify="right", width=8)
        table.add_column("Level", justify="right", width=8)
        table.add_column("S/L Ratio", justify="right", width=9)
        table.add_column("Gap (raw)", justify="right", width=9)
        table.add_column("Gap (scaled)", justify="right", width=11)
        table.add_column("Density", justify="right", width=8)
        table.add_column("W_level", justify="right", width=8)
        table.add_column("Active%", justify="right", width=8)
        table.add_column("Grad Ev", justify="right", width=10)
        table.add_column("Grad NEv", justify="right", width=10)
        table.add_column("DRO μ/max", justify="right", width=10)

        for c in range(n_ch):
            sh = comp.get("shape", [0])[c] if c < len(comp.get("shape", [])) else 0
            lv = comp.get("level", [0])[c] if c < len(comp.get("level", [])) else 0
            ratio = sh / max(lv, 1e-6)

            gap_raw = comp.get("level_gap_mean", [0])[c] if c < len(comp.get("level_gap_mean", [])) else 0
            gap_scaled = comp.get("gap_scaled_mean", [0])[c] if c < len(comp.get("gap_scaled_mean", [])) else 0
            density = comp.get("density_scale_mean", [0])[c] if c < len(comp.get("density_scale_mean", [])) else 0
            w_lvl = comp.get("w_level_mean", [0])[c] if c < len(comp.get("w_level_mean", [])) else 0
            active = comp.get("batch_active_frac", [0])[c] if c < len(comp.get("batch_active_frac", [])) else 0

            grad_ev = comp.get("grad_event", [0])[0]
            grad_nev = comp.get("grad_nonevent", [0])[0]

            dro_mu_val = comp.get("dro_w_mean", [0])[c] if c < len(comp.get("dro_w_mean", [])) else 0
            dro_max_val = comp.get("dro_w_max", [0])[c] if c < len(comp.get("dro_w_max", [])) else 0

            # Color coding
            def _ratio_color(v):
                if 0.3 <= v <= 3.0:
                    return "green"
                elif 0.1 <= v <= 10.0:
                    return "yellow"
                else:
                    return "red"

            def _gap_color(v):
                if v > 0.1:
                    return "green"
                elif v > 0.01:
                    return "yellow"
                else:
                    return "red"

            def _grad_color(v):
                if abs(v) < 1e-6:
                    return "dim"
                elif v < 0:
                    return "green"  # pushing UP (negative gradient = increase)
                else:
                    return "red"    # pushing DOWN

            def _active_color(v):
                if v > 0.01:
                    return "green"
                elif v > 0.001:
                    return "yellow"
                else:
                    return "red"

            ratio_str = Text(f"{ratio:.2f}", style=_ratio_color(ratio))
            gap_scaled_str = Text(f"{gap_scaled:.4f}", style=_gap_color(gap_scaled))
            grad_ev_str = Text(f"{grad_ev:+.6f}", style=_grad_color(grad_ev))
            grad_nev_str = Text(f"{grad_nev:+.6f}", style=_grad_color(grad_nev))
            active_str = Text(f"{active*100:.1f}%", style=_active_color(active))

            table.add_row(
                f"ch{c}",
                f"{sh:.3f}",
                f"{lv:.3f}",
                ratio_str,
                f"{gap_raw:.4f}",
                gap_scaled_str,
                f"{density:.2f}x",
                f"{w_lvl:.3f}",
                active_str,
                grad_ev_str,
                grad_nev_str,
                f"{dro_mu_val:.2f}/{dro_max_val:.1f}",
            )

        _console.print(table)

        # Summary panel
        summary_lines = []
        for c in range(n_ch):
            grad_ev = comp.get("grad_event", [0])[0]
            grad_nev = comp.get("grad_nonevent", [0])[0]
            if grad_ev > 0:
                summary_lines.append(f"[red]⚠ ch{c}: Event gradient is POSITIVE (pushing DOWN) — collapse risk![/]")
            elif abs(grad_ev) < 1e-6:
                summary_lines.append(f"[yellow]⚠ ch{c}: Event gradient is near ZERO — shape loss may be dead[/]")
            else:
                summary_lines.append(f"[green]✓ ch{c}: Event gradient is NEGATIVE (pushing UP) — healthy[/]")

        if summary_lines:
            _console.print(Panel(
                "\n".join(summary_lines),
                title="[bold]Gradient Health[/]",
                border_style="blue",
            ))


# ---------------------------------------------------------------------------
# Extended Loss Gradient Diagnostics (updated for new telemetry keys)
# ---------------------------------------------------------------------------

class LossGradientDiagnosticsCallbackV2(Callback):
    """
    Updated gradient diagnostics that include the new density-scaled gap,
    Hájek weights, and gradient direction telemetry.

    wandb keys (new):
        grad_diag/ch_{c}/density_scale_mean
        grad_diag/ch_{c}/gap_scaled_mean
        grad_diag/ch_{c}/w_level_mean
        grad_diag/ch_{c}/batch_active_frac
        grad_diag/grad_event
        grad_diag/grad_nonevent
    """

    def __init__(self, log_every_n_epochs: int = 1):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self._buf: dict = defaultdict(list)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        crit = getattr(pl_module, "train_criterion", None)
        comp = getattr(crit, "_last_components", None) if crit is not None else None
        if not comp:
            return

        C = len(comp.get("shape", []))
        if C == 0:
            return

        # ── 1. Loss telemetry from _last_components ─────────────────────
        for key in (
            "dro_w_mean", "dro_w_std", "dro_w_max", "dro_frac_up",
            "event_frac", "level_gap_mean", "level_gap_max", "shape_dc",
            "level_gap_ev_mean", "level_gap_ev_max", "level_gap_sat",
            "shape_level_ratio",
            # NEW keys
            "density_scale_mean", "gap_scaled_mean",
            "w_level_mean", "batch_active_frac",
        ):
            vals = comp.get(key)
            if vals is None:
                continue
            for c, v in enumerate(vals):
                self._buf[f"grad_diag/ch_{c}/{key}"].append(v)

        # NEW: Gradient direction (scalar, not per-channel)
        grad_ev = comp.get("grad_event", [0])[0]
        grad_nev = comp.get("grad_nonevent", [0])[0]
        self._buf["grad_diag/grad_event"].append(grad_ev)
        self._buf["grad_diag/grad_nonevent"].append(grad_nev)

        # ── 2. Input-gradient decomposition ─────────────────────────────
        grad = getattr(crit, "_last_input_grad", None)
        if grad is not None:
            g = grad.float().cpu()
            if g.dim() == 2:
                g = g.unsqueeze(-1)

            dc = g.mean(dim=1, keepdim=True)
            ac = g - dc

            dc_norm = dc.squeeze(1).abs()
            ac_norm = ac.norm(dim=1) / (g.shape[1] ** 0.5 + 1e-8)
            tot_norm = dc_norm + ac_norm + 1e-12

            dc_frac = (dc_norm / tot_norm).mean(dim=0)
            ac_frac = (ac_norm / tot_norm).mean(dim=0)
            sign_frac = (g > 0).float().mean(dim=(0, 1))
            dc_mag = dc_norm.mean(dim=0)
            ac_mag = ac_norm.mean(dim=0)
            total_g = g.norm(dim=1).mean(dim=0)

            for c in range(min(g.shape[-1], C)):
                self._buf[f"grad_diag/ch_{c}/grad_dc_frac"].append(dc_frac[c].item())
                self._buf[f"grad_diag/ch_{c}/grad_ac_frac"].append(ac_frac[c].item())
                self._buf[f"grad_diag/ch_{c}/grad_sign_frac"].append(sign_frac[c].item())
                self._buf[f"grad_diag/ch_{c}/grad_dc_mag"].append(dc_mag[c].item())
                self._buf[f"grad_diag/ch_{c}/grad_ac_mag"].append(ac_mag[c].item())
                self._buf[f"grad_diag/ch_{c}/grad_total_norm"].append(total_g[c].item())

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        out_layer = getattr(pl_module, "fc_out", None)
        if (
            out_layer is None
            or not hasattr(out_layer, "weight")
            or out_layer.weight.grad is None
        ):
            out_layer = None
            for _, mod in reversed(list(pl_module.named_modules())):
                if (
                    isinstance(mod, torch.nn.Linear)
                    and hasattr(mod, "weight")
                    and mod.weight.grad is not None
                ):
                    out_layer = mod
                    break

        if out_layer is None:
            return

        g = out_layer.weight.grad.detach().float()
        row_norms = g.norm(dim=1)
        self._buf["grad_diag/fcout_grad_norm_mean"].append(row_norms.mean().item())
        self._buf["grad_diag/fcout_grad_norm_max"].append(row_norms.max().item())
        self._buf["grad_diag/fcout_grad_norm_std"].append(row_norms.std().item())
        self._buf["grad_diag/fcout_grad_sign_frac"].append((g > 0).float().mean().item())

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            self._buf.clear()
            return
        if not self._buf:
            return

        metrics = {k: float(np.mean(v)) for k, v in self._buf.items() if v}

        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

        C = max(
            (int(k.split("/ch_")[1].split("/")[0]) for k in metrics if "/ch_" in k),
            default=-1,
        ) + 1

        if C > 0:
            parts = []
            for c in range(C):
                pfx = f"grad_diag/ch_{c}"
                dc_frac = metrics.get(f"{pfx}/grad_dc_frac", float("nan"))
                ac_frac = metrics.get(f"{pfx}/grad_ac_frac", float("nan"))
                sign_frac = metrics.get(f"{pfx}/grad_sign_frac", float("nan"))
                dc_mag = metrics.get(f"{pfx}/grad_dc_mag", float("nan"))
                ac_mag = metrics.get(f"{pfx}/grad_ac_mag", float("nan"))
                dro_mean = metrics.get(f"{pfx}/dro_w_mean", float("nan"))
                dro_max = metrics.get(f"{pfx}/dro_w_max", float("nan"))
                dro_fup = metrics.get(f"{pfx}/dro_frac_up", float("nan"))
                gap_mean = metrics.get(f"{pfx}/level_gap_mean", float("nan"))
                gap_ev = metrics.get(f"{pfx}/level_gap_ev_mean", float("nan"))
                gap_sat = metrics.get(f"{pfx}/level_gap_sat", float("nan"))
                sl_ratio = metrics.get(f"{pfx}/shape_level_ratio", float("nan"))
                shape_dc = metrics.get(f"{pfx}/shape_dc", float("nan"))
                ev_frac = metrics.get(f"{pfx}/event_frac", float("nan"))
                # NEW
                density = metrics.get(f"{pfx}/density_scale_mean", float("nan"))
                gap_scaled = metrics.get(f"{pfx}/gap_scaled_mean", float("nan"))
                w_lvl = metrics.get(f"{pfx}/w_level_mean", float("nan"))
                active = metrics.get(f"{pfx}/batch_active_frac", float("nan"))
                parts.append(
                    f"ch{c}["
                    f"dc%={dc_frac:.0%} ac%={ac_frac:.0%} sign↑={sign_frac:.2f} "
                    f"dcMag={dc_mag:.4f} acMag={ac_mag:.4f} | "
                    f"dro={dro_mean:.2f}±{dro_max:.1f}× ↑{dro_fup:.0%} | "
                    f"gap={gap_mean:.3f}/{gap_ev:.3f} sat={gap_sat:.0%} "
                    f"sl={sl_ratio:.2f} | "
                    f"density={density:.2f}x gap_s={gap_scaled:.4f} "
                    f"w_lvl={w_lvl:.3f} act={active:.1%} | "
                    f"shDC={shape_dc:.4f} ev={ev_frac:.2f}]"
                )
            fcout_norm = metrics.get("grad_diag/fcout_grad_norm_mean", float("nan"))
            fcout_sign = metrics.get("grad_diag/fcout_grad_sign_frac", float("nan"))
            grad_ev = metrics.get("grad_diag/grad_event", 0.0)
            grad_nev = metrics.get("grad_diag/grad_nonevent", 0.0)
            logger.info(
                f"[Epoch {trainer.current_epoch}] GradDiag | "
                + " ".join(parts)
                + f" | fc_out[norm={fcout_norm:.4f} sign↑={fcout_sign:.2f}]"
                + f" | grad_ev={grad_ev:+.6f} grad_nev={grad_nev:+.6f}"
            )

        self._buf.clear()