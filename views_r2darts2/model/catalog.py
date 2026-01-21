from darts.models.forecasting.nbeats import NBEATSModel
from darts.models.forecasting.tft_model import TFTModel
from darts.models.forecasting.tcn_model import TCNModel
from darts.models.forecasting.block_rnn_model import BlockRNNModel
from darts.models.forecasting.transformer_model import TransformerModel
from darts.models.forecasting.tsmixer_model import TSMixerModel
from darts.models.forecasting.nlinear import NLinearModel
from darts.models.forecasting.tide_model import TiDEModel
from darts.models.forecasting.dlinear import DLinearModel
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import WandbLogger
import torch

from views_r2darts2.model.forecaster import DartsForecaster
from views_r2darts2.utils.loss import LossSelector


class ModelCatalog:

    def __init__(self, config: dict):
        """
        Initializes the model catalog with configuration parameters.
        """
        self.models = {
            "NBEATSModel": self._get_nbeats,
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

        self.loss_name = self.config.get("loss_function", "WeightedPenaltyHuberLoss")

        # Prepare loss arguments from config parameters by dynamically grabbing all
        # potential loss-related keys from the config.
        self.loss_args = {
            # Huber-family params
            "zero_threshold": self.config.get("zero_threshold"),
            "delta": self.config.get("delta"),
            "non_zero_weight": self.config.get("non_zero_weight"),
            "false_negative_weight": self.config.get("false_negative_weight"),
            "false_positive_weight": self.config.get("false_positive_weight"),
            # Quantile-family params
            "tau": self.config.get("tau"),
            # Shrinkage-family params
            "a": self.config.get("a"),
            "c": self.config.get("c"),
            # SpikeFocal-family params
            "alpha": self.config.get("alpha"),
            "gamma": self.config.get("gamma"),
            "spike_threshold": self.config.get("spike_threshold"),
            # ZeroInflated-family params
            "zero_weight": self.config.get("zero_weight"),
            "count_weight": self.config.get("count_weight"),
        }
        # Filter out None values, so that loss function defaults can apply
        self.loss_args = {k: v for k, v in self.loss_args.items() if v is not None}

        self.loss_fn = LossSelector.get_loss_function(self.loss_name, **self.loss_args)

        self.lr_scheduler_args = {
            "mode": "min",
            "factor": self.config.get("lr_scheduler_factor", 0.1),
            "patience": self.config.get("lr_scheduler_patience", 3),
            "min_lr": self.config.get("lr_scheduler_min_lr", 1e-6),
            "monitor": "train_loss",
        }

    def get_model(self, model_name: str):
        """
        Get a model class by its name.
        """
        return self.models.get(model_name)()

    def list_models(self):
        """
        List all available models in the catalog.
        """
        return list(self.models.keys())

    def _get_tsmixer_model(self):
        torch.serialization.add_safe_globals([TSMixerModel, LossSelector])
        return TSMixerModel(
            input_chunk_length=self.config.get("input_chunk_length", 12 * 4),
            output_chunk_length=len(self.config["steps"]),
            output_chunk_shift=self.config.get("output_chunk_shift", 0),
            num_blocks=self.config.get("num_blocks", 2),
            ff_size=self.config.get("ff_size", 64),
            hidden_size=self.config.get("hidden_size", 64),
            activation=self.config.get("activation", "ReLU"),
            dropout=self.config.get("dropout", 0.1),
            norm_type=self.config.get("norm_type", "LayerNorm"),
            normalize_before=self.config.get("normalize_before", False),
            batch_size=self.config.get("batch_size", 64),
            n_epochs=self.config.get("n_epochs", 2),
            loss_fn=self.loss_fn,
            model_name=self.config.get("name", "TSMixerModel"),
            random_state=self.config.get("random_state", 42),
            force_reset=True,
            use_static_covariates=self.config.get(
                "use_static_covariates", True
            ),  # Default: True
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "gradient_clip_val": self.config.get("gradient_clip_val", 0.8),
                "logger": WandbLogger(log_model="all"),
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config.get("early_stopping_patience", 5),
                        min_delta=self.config.get("early_stopping_min_delta", 0.001),
                        mode="min",
                    ),
                    LearningRateMonitor(log_momentum=True),
                ],
                "enable_progress_bar": True,
            },
            optimizer_kwargs={
                "lr": self.config.get("lr", 3e-4),
                "weight_decay": self.config.get("weight_decay", 1e-3),
            },
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    def _get_tft_model(self):
        torch.serialization.add_safe_globals([TFTModel, LossSelector])

        # Revised training parameters
        batch_size = 256  # Reduced from 512 for better gradient variety

        return TFTModel(
            # Keep temporal configuration unchanged
            input_chunk_length=self.config.get("input_chunk_length", 12 * 4),
            output_chunk_length=len(self.config["steps"]),
            output_chunk_shift=self.config.get("output_chunk_shift", 0),  # Default: 0
            feed_forward=self.config.get(
                "feed_forward", "GatedResidualNetwork"
            ),  # Default: True
            add_relative_index=self.config.get(
                "add_relative_index", True
            ),  # Default: True
            use_static_covariates=self.config.get(
                "use_static_covariates", True
            ),  # Default: True
            full_attention=self.config.get("full_attention", False),  # Default: False
            lstm_layers=self.config.get("lstm_layers", 1),  # Default: 1
            num_attention_heads=self.config.get("num_attention_heads", 4),  # Default: 4
            hidden_size=self.config.get("hidden_size", 256),  # Default: 256
            dropout=self.config.get("dropout", 0.3),  # Default: 0.3
            # Critical training modifications
            batch_size=self.config.get("batch_size", batch_size),  # Default: 256
            loss_fn=self.loss_fn,
            model_name="TFTModel",
            norm_type="RMSNorm",  # Better for scaled outputs
            n_epochs=self.config.get("n_epochs", 2),
            use_reversible_instance_norm=self.config.get(
                "use_reversible_instance_norm", True
            ),  # https://openreview.net/forum?id=cGDAkQo1C0p - good for non-stationary conflict data
            # Training controls
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "logger": WandbLogger(log_model="all"),
                "gradient_clip_val": self.config.get("gradient_clip_val", 0.8),
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config.get("early_stopping_patience", 3),
                        min_delta=self.config.get("early_stopping_min_delta", 0.001),
                        mode="min",
                    ),
                    LearningRateMonitor(log_momentum=True),
                ],
                "enable_progress_bar": True,
            },
            optimizer_kwargs={
                "lr": self.config.get("lr", 3e-4),  # Default learning rate
                "weight_decay": self.config.get(
                    "weight_decay", 1e-3
                ),  # Default L2 regularization
            },
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    def _get_nbeats(self):
        torch.serialization.add_safe_globals([NBEATSModel, LossSelector])
        dropout_value = self.config.get("dropout", 0.2)
        print(f"DEBUG: N-BEATS dropout value: {dropout_value}")
        return NBEATSModel(
            input_chunk_length=self.config.get("input_chunk_length", 12 * 2),
            output_chunk_length=len(self.config["steps"]),
            output_chunk_shift=self.config.get("output_chunk_shift", 0),  # Default: 0
            generic_architecture=self.config.get("generic_architecture", True),
            num_stacks=self.config.get("num_stacks", 4),
            num_blocks=self.config.get("num_blocks", 2),
            num_layers=self.config.get("num_layers", 2),
            layer_widths=self.config.get("layer_widths", 128),
            activation=self.config.get("activation", "ReLU"),
            dropout=self.config.get("dropout", 0.2),
            random_state=self.config.get("random_state", 42),
            n_epochs=self.config.get("n_epochs", 2),
            batch_size=self.config.get("batch_size", 128),
            loss_fn=self.loss_fn,
            model_name=self.config.get("name", "NBEATSModel"),
            force_reset=self.config.get("force_reset", True),
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "logger": WandbLogger(log_model="all"),
                "gradient_clip_val": self.config.get("gradient_clip_val", 0.8),
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config.get("early_stopping_patience", 5),
                        min_delta=self.config.get("early_stopping_min_delta", 0.001),
                        mode="min",
                    ),
                    LearningRateMonitor(log_momentum=True),
                ],
                "enable_progress_bar": True,
            },
            optimizer_kwargs={
                "lr": self.config.get("lr", 3e-4),
                "weight_decay": self.config.get("weight_decay", 1e-3),
            },
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,
        )

    def _get_tcn_model(self):
        torch.serialization.add_safe_globals([TCNModel, LossSelector])
        return TCNModel(
            input_chunk_length=self.config.get("input_chunk_length", 12 * 6),
            output_chunk_length=len(self.config["steps"]),
            output_chunk_shift=self.config.get("output_chunk_shift", 0),  # Default: 0
            kernel_size=self.config.get("kernel_size", 3),  # Default: 3
            num_filters=self.config.get("num_filters", 64),  # Default: 64
            dilation_base=self.config.get("dilation_base", 2),  # Default: 2
            # weight_norm=True, #BUG!
            dropout=self.config.get("dropout", 0.2),  # Default: 0.2
            force_reset=self.config.get(
                "force_reset", True
            ),  # Reset the model if it already exists
            save_checkpoints=True,
            batch_size=self.config.get("batch_size", 64),
            model_name=self.config.get("name", "TCNModel"),
            random_state=self.config.get("random_state", 42),
            n_epochs=self.config.get("n_epochs", 2),
            loss_fn=self.loss_fn,
            use_reversible_instance_norm=self.config.get(
                "use_reversible_instance_norm", True
            ),  # https://openreview.net/forum?id=cGDAkQo1C0p - good for non-stationary conflict data
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "logger": WandbLogger(log_model="all"),
                "gradient_clip_val": self.config.get("gradient_clip_val", 0.8),
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config.get("early_stopping_patience", 5),
                        min_delta=self.config.get("early_stopping_min_delta", 0.001),
                        mode="min",
                    ),
                ],
            },
            optimizer_kwargs={
                "lr": self.config.get("lr", 3e-4),
                "weight_decay": self.config.get("weight_decay", 1e-3),
            },
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    def _get_rnn_model(self):
        torch.serialization.add_safe_globals([BlockRNNModel, LossSelector])
        return BlockRNNModel(
            input_chunk_length=self.config.get("input_chunk_length", 12 * 6),
            output_chunk_length=len(self.config["steps"]),
            output_chunk_shift=self.config.get("output_chunk_shift", 0),  # Default: 0
            model=self.config.get(
                "rnn_type", "LSTM"
            ),  # Choose between 'LSTM', 'GRU', or 'RNN'
            hidden_dim=self.config.get("hidden_dim", 5),  # Size of the hidden layers
            activation=self.config.get("activation", "ReLU"),
            n_rnn_layers=self.config.get("n_rnn_layers", 2),  # Number of RNN layers
            dropout=self.config.get("dropout", 0.4),  # Dropout rate for regularization
            batch_size=self.config.get("batch_size", 256),  # Batch size for training
            n_epochs=self.config.get("n_epochs", 7),  # Number of training epochs
            loss_fn=self.loss_fn,
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "logger": WandbLogger(log_model="all"),
                "gradient_clip_val": self.config.get("gradient_clip_val", 0.8),
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config.get("early_stopping_patience", 5),
                        min_delta=self.config.get("early_stopping_min_delta", 0.001),
                        mode="min",
                    ),
                ],
                "enable_progress_bar": True,
            },
            optimizer_kwargs={
                "lr": self.config.get("lr", 1e-4),  # Learning rate
                "weight_decay": self.config.get(
                    "weight_decay", 1e-4
                ),  # L2 regularization
            },
            use_reversible_instance_norm=self.config.get(
                "use_reversible_instance_norm", True
            ),  # Config-based, default True for non-stationary conflict data
            model_name=self.config.get("name", "BlockRNNModel"),  # Model name
            random_state=self.config.get(
                "random_state", 42
            ),  # Random seed for reproducibility
            force_reset=True,  # Reset the model if it already exists
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    def _get_transformer_model(self):
        torch.serialization.add_safe_globals([TransformerModel, LossSelector])
        return TransformerModel(
            input_chunk_length=self.config.get(
                "input_chunk_length", 12 * 6
            ),  # Default: 72
            output_chunk_length=len(
                self.config["steps"]
            ),  # Output chunk length based on steps
            output_chunk_shift=self.config.get("output_chunk_shift", 0),  # Default: 0
            d_model=self.config.get("d_model", 64),  # Default: 64
            nhead=self.config.get(
                "num_attention_heads", 4
            ),  # Default: 4 attention heads
            num_encoder_layers=self.config.get(
                "num_encoder_layers", 3
            ),  # Default: 3 encoder layers
            num_decoder_layers=self.config.get(
                "num_decoder_layers", 3
            ),  # Default: 3 decoder layers
            dim_feedforward=self.config.get("dim_feedforward", 512),  # Default: 512
            dropout=self.config.get("dropout", 0.1),  # Default: 0.1
            activation=self.config.get("activation", "ReLU"),  # Default: 'ReLU'
            norm_type=self.config.get("norm_type", None),  # Default: None
            batch_size=self.config.get("batch_size", 256),  # Default: 32
            n_epochs=self.config.get("n_epochs", 2),  # Default: 100
            loss_fn=self.loss_fn,
            model_name=self.config.get("name", "TransformerModel"),  # Model name
            random_state=self.config.get(
                "random_state", 42
            ),  # Random seed for reproducibility
            force_reset=True,  # Reset the model if it already exists
            use_reversible_instance_norm=self.config.get(
                "use_reversible_instance_norm", True
            ),  # Good for non-stationary conflict data
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "gradient_clip_val": self.config.get("gradient_clip_val", 0.8),
                "logger": WandbLogger(log_model="all"),
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config.get("early_stopping_patience", 5),
                        min_delta=self.config.get("early_stopping_min_delta", 0.001),
                        mode="min",
                    ),
                    LearningRateMonitor(log_momentum=True),
                ],
                "enable_progress_bar": True,
            },
            optimizer_kwargs={
                "lr": self.config.get("lr", 3e-4),  # Default learning rate
                "weight_decay": self.config.get(
                    "weight_decay", 1e-3
                ),  # Default L2 regularization
            },
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    def _get_nlinear_model(self):
        torch.serialization.add_safe_globals([NLinearModel, LossSelector])
        return NLinearModel(
            input_chunk_length=self.config.get(
                "input_chunk_length", 12 * 6
            ),  # Default: 72
            output_chunk_length=len(
                self.config["steps"]
            ),  # Output chunk length based on steps
            output_chunk_shift=self.config.get("output_chunk_shift", 0),  # Default: 0
            shared_weights=self.config.get("shared_weights", False),  # Default: False
            const_init=self.config.get("const_init", True),  # Default: True
            normalize=self.config.get("normalize", False),  # Default: False
            use_static_covariates=self.config.get(
                "use_static_covariates", True
            ),  # Default: True
            batch_size=self.config.get("batch_size", 64),  # Default: 64
            n_epochs=self.config.get("n_epochs", 2),  # Default: 2
            loss_fn=self.loss_fn,  # Use configured loss function from LossSelector
            model_name=self.config.get("name", "NLinearModel"),  # Model name
            random_state=self.config.get(
                "random_state", 42
            ),  # Random seed for reproducibility
            force_reset=True,  # Reset the model if it already exists
            use_reversible_instance_norm=self.config.get(
                "use_reversible_instance_norm", True
            ),  # Good for non-stationary conflict data
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "logger": WandbLogger(log_model="all"),
                "gradient_clip_val": self.config.get("gradient_clip_val", 0.8),
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config.get("early_stopping_patience", 5),
                        min_delta=self.config.get("early_stopping_min_delta", 0.001),
                        mode="min",
                    ),
                    LearningRateMonitor(log_momentum=True),
                ],
                "enable_progress_bar": True,
            },
            optimizer_kwargs={
                "lr": self.config.get("lr", 3e-4),  # Default learning rate
                "weight_decay": self.config.get(
                    "weight_decay", 1e-3
                ),  # Default L2 regularization
            },
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    def _get_dlinear_model(self):
        torch.serialization.add_safe_globals([DLinearModel, LossSelector])
        return DLinearModel(
            input_chunk_length=self.config.get(
                "input_chunk_length", 12 * 6
            ),  # Default: 72
            output_chunk_length=len(
                self.config["steps"]
            ),  # Output chunk length based on steps
            output_chunk_shift=self.config.get("output_chunk_shift", 0),  # Default: 0
            shared_weights=self.config.get("shared_weights", False),  # Default: False
            kernel_size=self.config.get("kernel_size", 25),  # Default: 25
            const_init=self.config.get("const_init", True),  # Default: True
            use_static_covariates=self.config.get(
                "use_static_covariates", True
            ),  # Default: True
            batch_size=self.config.get("batch_size", 64),  # Default: 64
            n_epochs=self.config.get("n_epochs", 2),  # Default: 2
            loss_fn=self.loss_fn,  # Use configured loss function from LossSelector
            model_name=self.config.get("name", "DLinearModel"),  # Model name
            random_state=self.config.get(
                "random_state", 42
            ),  # Random seed for reproducibility
            force_reset=True,  # Reset the model if it already exists
            use_reversible_instance_norm=self.config.get(
                "use_reversible_instance_norm", True
            ),  # Good for non-stationary conflict data
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "logger": WandbLogger(log_model="all"),
                "gradient_clip_val": self.config.get("gradient_clip_val", 0.8),
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config.get("early_stopping_patience", 5),
                        min_delta=self.config.get("early_stopping_min_delta", 0.001),
                        mode="min",
                    ),
                    LearningRateMonitor(log_momentum=True),
                ],
                "enable_progress_bar": True,
            },
            optimizer_kwargs={
                "lr": self.config.get("lr", 3e-4),  # Default learning rate
                "weight_decay": self.config.get(
                    "weight_decay", 1e-3
                ),  # Default L2 regularization
            },
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    def _get_tide_model(self):
        torch.serialization.add_safe_globals([TiDEModel, LossSelector])
        batch_size = 64
        return TiDEModel(
            input_chunk_length=self.config.get(
                "input_chunk_length", 12 * 4
            ),  # Default: 72
            output_chunk_length=len(
                self.config["steps"]
            ),  # Output chunk length based on steps
            output_chunk_shift=self.config.get("output_chunk_shift", 0),  # Default: 0
            num_encoder_layers=self.config.get(
                "num_encoder_layers", 1
            ),  # Default: 1 encoder layer
            num_decoder_layers=self.config.get(
                "num_decoder_layers", 1
            ),  # Default: 1 decoder layer
            decoder_output_dim=self.config.get("decoder_output_dim", 16),  # Default: 16
            hidden_size=self.config.get("hidden_size", 128),  # Default: 128
            temporal_width_past=self.config.get("temporal_width_past", 4),  # Default: 4
            temporal_width_future=self.config.get(
                "temporal_width_future", 4
            ),  # Default: 4
            temporal_decoder_hidden=self.config.get(
                "temporal_decoder_hidden", 32
            ),  # Default: 32
            use_layer_norm=self.config.get("use_layer_norm", False),  # Default: False
            dropout=self.config.get("dropout", 0.4),  # Default: 0.1
            use_static_covariates=self.config.get(
                "use_static_covariates", True
            ),  # Default: True
            batch_size=self.config.get("batch_size", batch_size),  # Default: 64
            n_epochs=self.config.get("n_epochs", 2),  # Default: 2
            loss_fn=self.loss_fn,
            model_name=self.config.get("name", "TiDEModel"),  # Model name
            random_state=self.config.get(
                "random_state", 42
            ),  # Random seed for reproducibility
            force_reset=True,  # Reset the model if it already exists
            use_reversible_instance_norm=self.config.get(
                "use_reversible_instance_norm", True
            ),  # Good for non-stationary conflict data
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "logger": WandbLogger(log_model="all"),
                "gradient_clip_val": self.config.get("gradient_clip_val", 0.8),
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config.get("early_stopping_patience", 3),
                        min_delta=self.config.get("early_stopping_min_delta", 0.001),
                        mode="min",
                    ),
                    LearningRateMonitor(log_momentum=True),
                ],
                "enable_progress_bar": True,
            },
            optimizer_kwargs={
                "lr": self.config.get("lr", 2e-3),
                "weight_decay": self.config.get(
                    "weight_decay", 1e-5
                ),  # Default L2 regularization
            },
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )