"""
Defines the model structures for the experiment, including the baseline and N-BEATS wrapper.
"""

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import NBEATSModel

# from darts.models.components import activation_fns # Removed: For dynamic activation
from pytorch_lightning.callbacks.early_stopping import EarlyStopping  # Explicit import
import torch.optim

from views_r2darts2.math import WeightedPenaltyHuberLoss, AsymmetricQuantileLoss


class BaseModel:
    """Abstract base class for models in this experiment."""

    def fit(self, train_ts: TimeSeries):
        raise NotImplementedError

    def predict(self, n: int) -> TimeSeries:
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        raise NotImplementedError


class HeuristicBaseline(BaseModel):
    """Implements the training set mean baseline."""

    def __init__(self):  # Removed window_size
        super().__init__()
        self.prediction_value = 0.0
        self._last_train_time = None  # To store end time of training data
        self._freq = None  # To store frequency of training data

    def fit(self, train_ts: TimeSeries):
        """Calculates the mean of the entire training series."""
        print(
            f"Fitting baseline: calculating mean of entire training series (length {len(train_ts)})."
        )
        self.prediction_value = np.mean(train_ts.values().flatten())
        self._last_train_time = train_ts.end_time()
        self._freq = train_ts.freq
        print(f"Baseline prediction value set to: {self.prediction_value:.4f}")

    def predict(self, n: int) -> TimeSeries:
        """Returns a constant prediction series in raw scale, with correct time index."""
        # Convert prediction_value from log1p scale back to raw scale and ensure non-negativity
        raw_prediction_value = np.expm1(self.prediction_value)
        raw_prediction_value = max(0.0, raw_prediction_value)  # Clip at 0

        # Construct the correct time index for the predictions
        if self._last_train_time is None or self._freq is None:
            raise RuntimeError(
                "Baseline not fitted: train_ts.end_time() and freq are not set."
            )

        prediction_start_time = self._last_train_time + self._freq
        time_index = pd.date_range(
            start=prediction_start_time, periods=n, freq=self._freq
        )

        prediction_series = np.full(shape=(n, 1), fill_value=raw_prediction_value)
        return TimeSeries.from_times_and_values(time_index, prediction_series)

    def save(self, path):
        # Baseline is simple enough to not require saving/loading
        pass

    @classmethod
    def load(cls, path):
        # Baseline is simple enough to not require saving/loading
        pass


class DartsNBEATS(BaseModel):
    """A wrapper for the Darts N-BEATS model to fit the experiment's interface."""

    def __init__(
        self,
        nbeats_hps,
        loss_fn_class,
        loss_fn_params,
        trainer_config,
        optimizer_config,
        lr_scheduler_config,
        early_stopping_config,
    ):
        super().__init__()

        # Dynamically get the loss function class
        loss_class_map = {
            "WeightedPenaltyHuberLoss": WeightedPenaltyHuberLoss,
            "AsymmetricQuantileLoss": AsymmetricQuantileLoss,
        }
        if loss_fn_class not in loss_class_map:
            raise ValueError(f"Unknown loss function class: {loss_fn_class}")
        loss_fn = loss_class_map[loss_fn_class](**loss_fn_params)

        # Dynamically get optimizer class
        optimizer_cls = getattr(torch.optim, optimizer_config["optimizer_cls"])

        # Dynamically get activation function
        # activation_fn = getattr(activation_fns, nbeats_hps["activation"]) # Removed this line

        # Construct PyTorch Lightning Trainer arguments (pl_trainer_kwargs)
        pl_trainer_kwargs = trainer_config.copy()
        callbacks = []

        # Add EarlyStopping callback
        if early_stopping_config.get("early_stopping_patience") is not None:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_config["early_stopping_patience"],
                    min_delta=early_stopping_config["early_stopping_min_delta"],
                    mode="min",
                )
            )

        # Add LR Scheduler callback if specified. Darts NBEATS usually handles this via `lr_scheduler_cls` directly.
        # If a custom callback is needed, this section would be more complex.
        # For now, Darts NBEATS takes lr_scheduler_cls, lr_scheduler_kwargs directly.
        # This part requires careful mapping:
        # Darts NBEATS takes lr_scheduler_cls, lr_scheduler_kwargs
        # The provided config uses standard PyTorch LR schedulers that might require a wrapper.
        # For simplicity, I'll pass relevant lr_scheduler params directly to NBEATSModel.

        pl_trainer_kwargs["callbacks"] = callbacks

        # Construct NBEATSModel
        self.model = NBEATSModel(
            # Core NBEATS parameters
            input_chunk_length=nbeats_hps["input_chunk_length"],
            output_chunk_length=nbeats_hps["output_chunk_length"],
            num_stacks=nbeats_hps["num_stacks"],
            num_blocks=nbeats_hps["num_blocks"],
            dropout=nbeats_hps["dropout"],
            layer_widths=nbeats_hps["layer_widths"],
            num_layers=nbeats_hps["num_layers"],
            activation=nbeats_hps["activation"],  # Pass the string directly
            generic_architecture=nbeats_hps["generic_architecture"],
            batch_size=nbeats_hps["batch_size"],
            output_chunk_shift=nbeats_hps["output_chunk_shift"],
            random_state=nbeats_hps["random_state"],
            # mc_dropout=nbeats_hps["mc_dropout"], # Removed: Not a direct NBEATSModel constructor arg
            # Scalers and log transforms
            # target_scaler=nbeats_hps["target_scaler"], # Removed: Not a direct NBEATSModel constructor arg
            # feature_scaler=nbeats_hps["feature_scaler"], # Removed: Not a direct NBEATSModel constructor arg
            # log_targets=nbeats_hps["log_targets"], # Removed: Not a direct NBEATSModel constructor arg
            # log_features=nbeats_hps["log_features"], # Removed: Not a direct NBEATSModel constructor arg
            # Loss function
            loss_fn=loss_fn,
            # Trainer parameters (passed through)
            n_epochs=nbeats_hps["n_epochs"],  # NBEATSModel also takes this
            # Optimizer parameters
            optimizer_cls=optimizer_cls,
            optimizer_kwargs={
                "lr": optimizer_config["lr"],
                "weight_decay": optimizer_config["weight_decay"],
            },
            # LR Scheduler parameters (NBEATSModel takes these directly)
            lr_scheduler_cls=getattr(
                torch.optim.lr_scheduler, lr_scheduler_config["lr_scheduler_cls"]
            ),
            lr_scheduler_kwargs={
                "patience": lr_scheduler_config["lr_scheduler_patience"],
                "factor": lr_scheduler_config["lr_scheduler_factor"],
                "min_lr": lr_scheduler_config["lr_scheduler_min_lr"],
            },
            # PyTorch Lightning Trainer kwargs
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=nbeats_hps["force_reset"],  # Existing parameter
        )

    def fit(
        self, train_ts: TimeSeries, val_ts: TimeSeries = None
    ):  # Added val_ts parameter
        self.model.fit(
            series=train_ts, val_series=val_ts, verbose=False
        )  # Modified to pass val_series

    def predict(self, n: int) -> TimeSeries:
        return self.model.predict(n=n)

    def save(self, path):
        self.model.save(path)

    @classmethod
    def load(
        cls,
        path,
        nbeats_hps,
        loss_fn_class,
        loss_fn_params,
        trainer_config,
        optimizer_config,
        lr_scheduler_config,
        early_stopping_config,
    ):
        # We need to reconstruct the model with the same parameters
        # before loading the state.
        instance = cls(
            nbeats_hps,
            loss_fn_class,
            loss_fn_params,
            trainer_config,
            optimizer_config,
            lr_scheduler_config,
            early_stopping_config,
        )
        instance.model = NBEATSModel.load(path)
        return instance
