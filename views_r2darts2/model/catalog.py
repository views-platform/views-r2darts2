from darts.models.forecasting.nbeats import NBEATSModel
from darts.models.forecasting.nhits import NHiTSModel
from darts.models.forecasting.tft_model import TFTModel
from darts.models.forecasting.tcn_model import TCNModel
from darts.models.forecasting.block_rnn_model import BlockRNNModel
from darts.models.forecasting.transformer_model import TransformerModel
from darts.models.forecasting.tsmixer_model import TSMixerModel
from darts.models.forecasting.nlinear import NLinearModel
from darts.models.forecasting.tide_model import TiDEModel
from darts.models.forecasting.dlinear import DLinearModel
from pytorch_lightning.callbacks import EarlyStopping, Callback
from views_r2darts2.utils.loss import WeightedHuberLoss
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import WandbLogger
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NaNDetectionCallback(Callback):
    """
    Callback to detect NaN loss and stop training early.
    
    When a model becomes numerically unstable (producing NaN loss), continuing
    training is pointless and wastes compute. This callback:
    1. Detects NaN loss
    2. Logs useful debugging info
    3. Stops training immediately
    """
    
    def __init__(self, patience: int = 3):
        """
        Args:
            patience: Number of consecutive NaN batches before stopping.
                      Set to 1 for immediate stop, higher for transient NaN tolerance.
        """
        super().__init__()
        self.patience = patience
        self.nan_count = 0
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        if loss is not None and torch.isnan(loss):
            self.nan_count += 1
            logger.warning(
                f"NaN loss detected at epoch {trainer.current_epoch}, batch {batch_idx} "
                f"(consecutive NaN count: {self.nan_count}/{self.patience})"
            )
            if self.nan_count >= self.patience:
                logger.error(
                    "Training stopped due to persistent NaN loss. "
                    "Suggestions: lower learning rate, increase gradient clipping, "
                    "check data scaling, verify norm_type='LayerNorm'"
                )
                trainer.should_stop = True
        else:
            self.nan_count = 0  # Reset on valid loss


# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from views_r2darts2.model.forecaster import DartsForecaster
from views_r2darts2.utils.loss import LossSelector


class ModelCatalog:

    def __init__(self, config: dict):
        """
        Initializes the model catalog with configuration parameters.
        Args:
            config (dict): Configuration dictionary containing model and loss function parameters.
        Attributes:
            models (dict): Mapping of model names to their respective getter methods.
            config (dict): Stores the provided configuration dictionary.
            device: The device (CPU/GPU) to be used for model training and inference.
            loss_name (str): Name of the loss function to be used, defaults to 'WeightedPenaltyHuberLoss'.
            loss_args (dict): Arguments for the loss function, extracted from the configuration.
            loss_fn: The selected loss function instance, initialized with the specified arguments.
        """
        self.models = {
            "NBEATSModel": self._get_nbeats,
            "NHiTSModel": self._get_nhits,
            "TFTModel": self._get_tft_model,
            "TCNModel": self._get_tcn_model,
            "BlockRNNModel": self._get_rnn_model,
            "TransformerModel": self._get_transformer_model,
            "NLinearModel": self._get_nlinear_model,
            "TiDEModel": self._get_tide_model,
            "DLinearModel": self._get_dlinear_model,
            "TSMixerModel": self._get_tsmixer_model,
        }
        self.config = config
        self.device = DartsForecaster.get_device()

        self.loss_name = self.config["loss_function"]

        # Prepare loss arguments from config parameters
        # Use .get() with None defaults so missing params don't crash
        # LossSelector.get_loss_function filters to only valid params for each loss
        self.loss_args = {
            # Common to most losses
            "zero_threshold": self.config.get("zero_threshold"),
            "non_zero_weight": self.config.get("non_zero_weight"),
            # Huber-specific
            "delta": self.config.get("delta"),
            "false_negative_weight": self.config.get("false_negative_weight"),
            "false_positive_weight": self.config.get("false_positive_weight"),
            # Tweedie-specific
            "p": self.config.get("p"),
            "eps": self.config.get("eps"),
        }
        # Remove None values so LossSelector doesn't pass them
        self.loss_args = {k: v for k, v in self.loss_args.items() if v is not None}
        self.loss_fn = LossSelector.get_loss_function(self.loss_name, **self.loss_args)
        self.lr_scheduler_args = {
            "mode": "min",
            "factor": self.config["lr_scheduler_factor"],
            "patience": self.config["lr_scheduler_patience"],
            "min_lr": self.config["lr_scheduler_min_lr"],
            "monitor": "train_loss",
        }

    def get_model(self, model_name: str):
        """
        Get a model class by its name.

        Args:
            model_name (str): The name of the model.

        Returns:
            Model class corresponding to the provided name.
        """
        return self.models.get(model_name)()

    def list_models(self):
        """
        List all available models in the catalog.

        Returns:
            List of model names.
        """
        return list(self.models.keys())

    def _get_tsmixer_model(self):
        torch.serialization.add_safe_globals([TSMixerModel, LossSelector])
        return TSMixerModel(
            input_chunk_length=self.config["input_chunk_length"],
            output_chunk_length=self.config.get("output_chunk_length", len(self.config["steps"])),
            output_chunk_shift=self.config["output_chunk_shift"],
            num_blocks=self.config["num_blocks"],
            ff_size=self.config["ff_size"],
            hidden_size=self.config["hidden_size"],
            activation=self.config["activation"],
            dropout=self.config["dropout"],
            norm_type=self.config["norm_type"],
            normalize_before=self.config["normalize_before"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            loss_fn=self.loss_fn,
            model_name=self.config["name"],
            random_state=self.config["random_state"],
            force_reset=True,
            use_static_covariates=self.config["use_static_covariates"],
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "gradient_clip_val": self.config["gradient_clip_val"],
                "logger": WandbLogger(log_model="all"),
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config["early_stopping_patience"],
                        min_delta=self.config["early_stopping_min_delta"],
                        mode="min",
                    ),
                    LearningRateMonitor(log_momentum=True),
                ],
                "enable_progress_bar": True,
            },
            optimizer_kwargs={
                "lr": self.config["lr"],
                "weight_decay": self.config["weight_decay"],
            },
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    def _get_tft_model(self):
        torch.serialization.add_safe_globals([TFTModel, LossSelector])

        return TFTModel(
            input_chunk_length=self.config["input_chunk_length"],
            output_chunk_length=self.config.get("output_chunk_length", len(self.config["steps"])),
            output_chunk_shift=self.config["output_chunk_shift"],
            feed_forward=self.config["feed_forward"],
            add_relative_index=self.config["add_relative_index"],
            use_static_covariates=self.config["use_static_covariates"],
            full_attention=self.config["full_attention"],
            lstm_layers=self.config["lstm_layers"],
            num_attention_heads=self.config["num_attention_heads"],
            hidden_size=self.config["hidden_size"],
            dropout=self.config["dropout"],
            batch_size=self.config["batch_size"],
            loss_fn=self.loss_fn,
            model_name=self.config["name"],
            norm_type=self.config.get("norm_type", "RMSNorm"),
            n_epochs=self.config["n_epochs"],
            random_state=self.config["random_state"],
            use_reversible_instance_norm=self.config["use_reversible_instance_norm"],
            skip_interpolation=self.config["skip_interpolation"],
            hidden_continuous_size=self.config["hidden_continuous_size"],
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "logger": WandbLogger(log_model="all"),
                "gradient_clip_val": self.config["gradient_clip_val"],
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config["early_stopping_patience"],
                        min_delta=self.config["early_stopping_min_delta"],
                        mode="min",
                    ),
                    LearningRateMonitor(log_momentum=True),
                ],
                "enable_progress_bar": True,
            },
            optimizer_kwargs={
                "lr": self.config["lr"],
                "weight_decay": self.config["weight_decay"],
            },
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    def _get_nbeats(self):
        torch.serialization.add_safe_globals([NBEATSModel, LossSelector])
        return NBEATSModel(
            input_chunk_length=self.config["input_chunk_length"],
            output_chunk_length=self.config.get("output_chunk_length", len(self.config["steps"])),
            output_chunk_shift=self.config["output_chunk_shift"],
            generic_architecture=self.config["generic_architecture"],
            num_stacks=self.config["num_stacks"],
            num_blocks=self.config["num_blocks"],
            num_layers=self.config["num_layers"],
            layer_widths=self.config["layer_width"],
            activation=self.config["activation"],
            dropout=self.config["dropout"],
            random_state=self.config["random_state"],
            n_epochs=self.config["n_epochs"],
            batch_size=self.config["batch_size"],
            loss_fn=self.loss_fn,
            model_name=self.config["name"],
            force_reset=self.config["force_reset"],
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "logger": WandbLogger(log_model="all"),
                "gradient_clip_val": self.config["gradient_clip_val"],
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config["early_stopping_patience"],
                        min_delta=self.config["early_stopping_min_delta"],
                        mode="min",
                    ),
                    LearningRateMonitor(log_momentum=True),
                ],
                "enable_progress_bar": True,
            },
            optimizer_kwargs={
                "lr": self.config["lr"],
                "weight_decay": self.config["weight_decay"],
            },
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,
        )

    def _get_nhits(self):
        """N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.
        
        Similar to N-BEATS but with multi-rate sampling for better performance
        at lower computational cost. Uses MaxPooling for input downsampling
        and multi-scale interpolation for outputs.
        """
        torch.serialization.add_safe_globals([NHiTSModel, LossSelector])
        return NHiTSModel(
            input_chunk_length=self.config["input_chunk_length"],
            output_chunk_length=self.config.get("output_chunk_length", len(self.config["steps"])),
            output_chunk_shift=self.config["output_chunk_shift"],
            num_stacks=self.config["num_stacks"],
            num_blocks=self.config["num_blocks"],
            num_layers=self.config["num_layers"],
            layer_widths=self.config["layer_width"],
            pooling_kernel_sizes=self.config["pooling_kernel_sizes"],
            n_freq_downsample=self.config["n_freq_downsample"],
            activation=self.config["activation"],
            MaxPool1d=self.config["max_pool_1d"],
            dropout=self.config["dropout"],
            random_state=self.config["random_state"],
            n_epochs=self.config["n_epochs"],
            batch_size=self.config["batch_size"],
            loss_fn=self.loss_fn,
            model_name=self.config["name"],
            force_reset=self.config["force_reset"],
            use_reversible_instance_norm=self.config["use_reversible_instance_norm"],
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "logger": WandbLogger(log_model="all"),
                "gradient_clip_val": self.config["gradient_clip_val"],
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config["early_stopping_patience"],
                        min_delta=self.config["early_stopping_min_delta"],
                        mode="min",
                    ),
                    LearningRateMonitor(log_momentum=True),
                ],
                "enable_progress_bar": True,
            },
            optimizer_kwargs={
                "lr": self.config["lr"],
                "weight_decay": self.config["weight_decay"],
            },
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,
        )

    def _get_tcn_model(self):
        torch.serialization.add_safe_globals([TCNModel, LossSelector])
        return TCNModel(
            input_chunk_length=self.config["input_chunk_length"],
            output_chunk_length=self.config.get("output_chunk_length", len(self.config["steps"])),
            output_chunk_shift=self.config["output_chunk_shift"],
            kernel_size=self.config["kernel_size"],
            num_filters=self.config["num_filters"],
            dilation_base=self.config["dilation_base"],
            dropout=self.config["dropout"],
            force_reset=self.config["force_reset"],
            save_checkpoints=True,
            batch_size=self.config["batch_size"],
            model_name=self.config["name"],
            random_state=self.config["random_state"],
            n_epochs=self.config["n_epochs"],
            loss_fn=self.loss_fn,
            use_reversible_instance_norm=self.config["use_reversible_instance_norm"],
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "logger": WandbLogger(log_model="all"),
                "gradient_clip_val": self.config["gradient_clip_val"],
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config["early_stopping_patience"],
                        min_delta=self.config["early_stopping_min_delta"],
                        mode="min",
                    ),
                ],
            },
            optimizer_kwargs={
                "lr": self.config["lr"],
                "weight_decay": self.config["weight_decay"],
            },
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    def _get_rnn_model(self):
        torch.serialization.add_safe_globals([BlockRNNModel, LossSelector])
        return BlockRNNModel(
            input_chunk_length=self.config["input_chunk_length"],
            output_chunk_length=self.config.get("output_chunk_length", len(self.config["steps"])),
            output_chunk_shift=self.config["output_chunk_shift"],
            model=self.config["rnn_type"],
            hidden_dim=self.config["hidden_dim"],
            activation=self.config["activation"],
            n_rnn_layers=self.config["n_rnn_layers"],
            dropout=self.config["dropout"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            loss_fn=self.loss_fn,
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "logger": WandbLogger(log_model="all"),
                "gradient_clip_val": self.config["gradient_clip_val"],
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config["early_stopping_patience"],
                        min_delta=self.config["early_stopping_min_delta"],
                        mode="min",
                    ),
                ],
                "enable_progress_bar": True,
            },
            optimizer_kwargs={
                "lr": self.config["lr"],
                "weight_decay": self.config["weight_decay"],
            },
            use_reversible_instance_norm=self.config["use_reversible_instance_norm"],
            model_name=self.config["name"],
            random_state=self.config["random_state"],
            force_reset=True,
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    def _get_transformer_model(self):
        torch.serialization.add_safe_globals([TransformerModel, LossSelector])
        
        d_model = self.config["d_model"]
        nhead = self.config.get("nhead", self.config.get("num_attention_heads"))
        
        # Validate d_model is divisible by nhead
        if d_model % nhead != 0:
            import logging
            logging.warning(
                f"d_model ({d_model}) not divisible by nhead ({nhead}). "
                f"Adjusting nhead to {d_model // (d_model // nhead)}."
            )
            nhead = max(1, d_model // 32)  # Ensure at least 32 dims per head
        
        return TransformerModel(
            input_chunk_length=self.config["input_chunk_length"],
            output_chunk_length=self.config.get("output_chunk_length", len(self.config["steps"])),
            output_chunk_shift=self.config["output_chunk_shift"],
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=self.config["num_encoder_layers"],
            num_decoder_layers=self.config["num_decoder_layers"],
            dim_feedforward=self.config["dim_feedforward"],
            dropout=self.config["dropout"],
            activation=self.config["activation"],
            norm_type=self.config["norm_type"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            loss_fn=self.loss_fn,
            model_name=self.config["name"],
            random_state=self.config["random_state"],
            force_reset=True,
            use_reversible_instance_norm=self.config["use_reversible_instance_norm"],
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "gradient_clip_val": self.config["gradient_clip_val"],
                "logger": WandbLogger(log_model="all"),
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config["early_stopping_patience"],
                        min_delta=self.config["early_stopping_min_delta"],
                        mode="min",
                    ),
                    LearningRateMonitor(log_momentum=True),
                    NaNDetectionCallback(patience=5),
                ],
                "enable_progress_bar": True,
                "detect_anomaly": self.config["detect_anomaly"],
            },
            optimizer_kwargs={
                "lr": self.config["lr"],
                "weight_decay": self.config["weight_decay"],
            },
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    def _get_nlinear_model(self):
        torch.serialization.add_safe_globals([NLinearModel, LossSelector])
        return NLinearModel(
            input_chunk_length=self.config["input_chunk_length"],
            output_chunk_length=self.config.get("output_chunk_length", len(self.config["steps"])),
            output_chunk_shift=self.config["output_chunk_shift"],
            shared_weights=self.config["shared_weights"],
            const_init=self.config["const_init"],
            normalize=self.config["normalize"],
            use_static_covariates=self.config["use_static_covariates"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            loss_fn=self.loss_fn,
            model_name=self.config["name"],
            random_state=self.config["random_state"],
            force_reset=True,
            use_reversible_instance_norm=self.config["use_reversible_instance_norm"],
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "logger": WandbLogger(log_model="all"),
                "gradient_clip_val": self.config["gradient_clip_val"],
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config["early_stopping_patience"],
                        min_delta=self.config["early_stopping_min_delta"],
                        mode="min",
                    ),
                    LearningRateMonitor(log_momentum=True),
                ],
                "enable_progress_bar": True,
            },
            optimizer_kwargs={
                "lr": self.config["lr"],
                "weight_decay": self.config["weight_decay"],
            },
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    def _get_dlinear_model(self):
        torch.serialization.add_safe_globals([DLinearModel, LossSelector])
        return DLinearModel(
            input_chunk_length=self.config["input_chunk_length"],
            output_chunk_length=self.config.get("output_chunk_length", len(self.config["steps"])),
            output_chunk_shift=self.config["output_chunk_shift"],
            shared_weights=self.config["shared_weights"],
            kernel_size=self.config["kernel_size"],
            const_init=self.config["const_init"],
            use_static_covariates=self.config["use_static_covariates"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            loss_fn=self.loss_fn,
            model_name=self.config["name"],
            random_state=self.config["random_state"],
            force_reset=True,
            use_reversible_instance_norm=self.config["use_reversible_instance_norm"],
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "gradient_clip_val": self.config["gradient_clip_val"],
                "logger": WandbLogger(log_model="all"),
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config["early_stopping_patience"],
                        min_delta=self.config["early_stopping_min_delta"],
                        mode="min",
                    ),
                    LearningRateMonitor(log_momentum=True),
                ],
                "enable_progress_bar": True,
            },
            optimizer_kwargs={
                "lr": self.config["lr"],
                "weight_decay": self.config["weight_decay"],
            },
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    def _get_tide_model(self):
        torch.serialization.add_safe_globals([TiDEModel, LossSelector])
        return TiDEModel(
            input_chunk_length=self.config["input_chunk_length"],
            output_chunk_length=self.config.get("output_chunk_length", len(self.config["steps"])),
            output_chunk_shift=self.config["output_chunk_shift"],
            num_encoder_layers=self.config["num_encoder_layers"],
            num_decoder_layers=self.config["num_decoder_layers"],
            decoder_output_dim=self.config["decoder_output_dim"],
            hidden_size=self.config["hidden_size"],
            temporal_width_past=self.config["temporal_width_past"],
            temporal_width_future=self.config["temporal_width_future"],
            temporal_hidden_size_past=self.config["temporal_hidden_size_past"],
            temporal_hidden_size_future=self.config["temporal_hidden_size_future"],
            temporal_decoder_hidden=self.config["temporal_decoder_hidden"],
            use_layer_norm=self.config["use_layer_norm"],
            dropout=self.config["dropout"],
            use_static_covariates=self.config["use_static_covariates"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            loss_fn=self.loss_fn,
            model_name=self.config["name"],
            random_state=self.config["random_state"],
            force_reset=True,
            use_reversible_instance_norm=self.config["use_reversible_instance_norm"],
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "logger": WandbLogger(log_model="all"),
                "gradient_clip_val": self.config["gradient_clip_val"],
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config["early_stopping_patience"],
                        min_delta=self.config["early_stopping_min_delta"],
                        mode="min",
                    ),
                    LearningRateMonitor(log_momentum=True),
                ],
                "enable_progress_bar": True,
            },
            optimizer_kwargs={
                "lr": self.config["lr"],
                "weight_decay": self.config["weight_decay"],
            },
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )
