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
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import WandbLogger
import torch
import logging

from views_r2darts2.model.darts_forecaster import DartsForecaster
from views_r2darts2.utils.loss.loss_catalog import LossCatalog
from views_r2darts2.utils.optimizer_catalog import OptimizerCatalog
from views_r2darts2.utils.reproducibility_gate import ReproducibilityGate
from views_r2darts2.utils.callbacks import GradientHealthCallback


logger = logging.getLogger(__name__)


class ModelCatalog:
    """
    Central repository for translating DNA manifests into concrete Darts Model instances.

    Intent Contract:
        - Purpose: Act as a factory for Darts forecasting models, ensuring they are initialized with
          the exact hyperparameters and loss functions declared in the upstream DNA manifest.
        - Non-Goals: Does not handle data loading or model execution.
        - Guarantees:
            - Ensures every model is audited for architectural compatibility with the forecast horizon.
            - Ensures standard loss functions and custom Fortress losses are correctly instantiated.
            - Guarantees that PyTorch Lightning trainers are configured with mandatory Fortress callbacks (NaNDetection, GradientHealth).
        - Failure Behavior: Raises ValueError if a requested model or loss is unknown, and KeyError if
          mandatory hyperparameters for a specific architecture are missing.
    """

    def __init__(self, config: dict):
        """
        Initializes the model catalog with configuration parameters.
        """
        # GENOMIC FIREWALL: Audit the manifest immediately upon entry
        # This protects against direct instantiation that bypasses the Manager
        ReproducibilityGate.Config.audit_manifest(config)

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

        # DELEGATION: Specialized catalogs handle genomic translation
        self.loss_fn = LossCatalog(self.config).get_loss()
        self.opt_catalog = OptimizerCatalog(self.config)

        self.lr_scheduler_args = {
            "mode": "min",
            "factor": self.config.get("lr_scheduler_factor"),
            "patience": self.config.get("lr_scheduler_patience"),
            "min_lr": self.config.get("lr_scheduler_min_lr"),
            "monitor": "train_loss",
        }

    def _get_common_pl_trainer_kwargs(self, extra_callbacks=None):
        callbacks = [
            EarlyStopping(
                monitor="train_loss",
                patience=self.config.get("early_stopping_patience"),
                min_delta=self.config.get("early_stopping_min_delta"),
                mode="min",
            ),
            LearningRateMonitor(log_momentum=True),
            GradientHealthCallback(),
        ]
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        return {
            "accelerator": "gpu",
            "logger": WandbLogger(log_model="all"),
            "gradient_clip_val": self.config.get("gradient_clip_val"),
            "callbacks": callbacks,
            "enable_progress_bar": True,
        }

    def _get_common_model_args(self):
        """
        Extracts common hyperparameters and training controls used by most Darts models.
        
        Returns:
            dict: Mapping of parameter names to DNA values.
        """
        return {
            "input_chunk_length": self.config.get("input_chunk_length"),
            "output_chunk_length": self.config.get("output_chunk_length"),
            "output_chunk_shift": self.config.get("output_chunk_shift"),
            "batch_size": self.config.get("batch_size"),
            "n_epochs": self.config.get("n_epochs"),
            "loss_fn": self.loss_fn,
            "model_name": self.config.get("name"),
            "random_state": self.config.get("random_state"),
            "force_reset": True,
            "pl_trainer_kwargs": self._get_common_pl_trainer_kwargs(),
            "optimizer_cls": self.opt_catalog.get_optimizer_cls(),
            "optimizer_kwargs": self.opt_catalog.get_optimizer_kwargs(),
            "lr_scheduler_cls": ReduceLROnPlateau,
            "lr_scheduler_kwargs": self.lr_scheduler_args,
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
        torch.serialization.add_safe_globals([TSMixerModel, LossCatalog])
        ReproducibilityGate.Config.audit_architecture(self.config)
        return TSMixerModel(
            **self._get_common_model_args(),
            num_blocks=self.config.get("num_blocks"),
            ff_size=self.config.get("ff_size"),
            hidden_size=self.config.get("hidden_size"),
            activation=self.config.get("activation"),
            dropout=self.config.get("dropout"),
            norm_type=self.config.get("norm_type"),
            normalize_before=self.config.get("normalize_before"),
            use_static_covariates=self.config.get("use_static_covariates"),
            use_reversible_instance_norm=self.config.get("use_reversible_instance_norm"),
        )

    def _get_tft_model(self):
        torch.serialization.add_safe_globals([TFTModel, LossCatalog])
        ReproducibilityGate.Config.audit_architecture(self.config)

        return TFTModel(
            **self._get_common_model_args(),
            feed_forward=self.config.get("feed_forward"),
            add_relative_index=self.config.get("add_relative_index"),
            use_static_covariates=self.config.get("use_static_covariates"),
            full_attention=self.config.get("full_attention"),
            lstm_layers=self.config.get("lstm_layers"),
            num_attention_heads=self.config.get("num_attention_heads"),
            hidden_size=self.config.get("hidden_size"),
            dropout=self.config.get("dropout"),
            norm_type=self.config.get("norm_type"),
            use_reversible_instance_norm=self.config.get("use_reversible_instance_norm"),
            skip_interpolation=self.config.get("skip_interpolation"),
            hidden_continuous_size=self.config.get("hidden_continuous_size"),
        )

    def _get_nbeats(self):
        torch.serialization.add_safe_globals([NBEATSModel, LossCatalog])
        ReproducibilityGate.Config.audit_architecture(self.config)
        return NBEATSModel(
            **self._get_common_model_args(),
            generic_architecture=self.config.get("generic_architecture"),
            num_stacks=self.config.get("num_stacks"),
            num_blocks=self.config.get("num_blocks"),
            num_layers=self.config.get("num_layers"),
            layer_widths=self.config.get("layer_widths"),
            activation=self.config.get("activation"),
            dropout=self.config.get("dropout"),
        )

    def _get_nhits(self):
        """N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting."""
        torch.serialization.add_safe_globals([NHiTSModel, LossCatalog])
        ReproducibilityGate.Config.audit_architecture(self.config)
        return NHiTSModel(
            **self._get_common_model_args(),
            num_stacks=self.config.get("num_stacks"),
            num_blocks=self.config.get("num_blocks"),
            num_layers=self.config.get("num_layers"),
            layer_widths=self.config.get("layer_widths"),
            pooling_kernel_sizes=self.config.get("pooling_kernel_sizes"),
            n_freq_downsample=self.config.get("n_freq_downsample"),
            activation=self.config.get("activation"),
            MaxPool1d=self.config.get("max_pool_1d"),
            dropout=self.config.get("dropout"),
            use_reversible_instance_norm=self.config.get("use_reversible_instance_norm"),
        )

    def _get_tcn_model(self):
        torch.serialization.add_safe_globals([TCNModel, LossCatalog])
        ReproducibilityGate.Config.audit_architecture(self.config)
        return TCNModel(
            **self._get_common_model_args(),
            kernel_size=self.config.get("kernel_size"),
            num_filters=self.config.get("num_filters"),
            dilation_base=self.config.get("dilation_base"),
            dropout=self.config.get("dropout"),
            use_reversible_instance_norm=self.config.get("use_reversible_instance_norm"),
        )

    def _get_rnn_model(self):
        torch.serialization.add_safe_globals([BlockRNNModel, LossCatalog])
        ReproducibilityGate.Config.audit_architecture(self.config)
        return BlockRNNModel(
            **self._get_common_model_args(),
            model=self.config.get("rnn_type"),
            hidden_dim=self.config.get("hidden_dim"),
            activation=self.config.get("activation"),
            n_rnn_layers=self.config.get("n_rnn_layers"),
            dropout=self.config.get("dropout"),
            use_reversible_instance_norm=self.config.get("use_reversible_instance_norm"),
        )

    def _get_transformer_model(self):
        torch.serialization.add_safe_globals([TransformerModel, LossCatalog])
        ReproducibilityGate.Config.audit_architecture(self.config)

        return TransformerModel(
            **self._get_common_model_args(),
            d_model=self.config.get("d_model"),
            nhead=self.config.get("nhead"),
            num_encoder_layers=self.config.get("num_encoder_layers"),
            num_decoder_layers=self.config.get("num_decoder_layers"),
            dim_feedforward=self.config.get("dim_feedforward"),
            dropout=self.config.get("dropout"),
            activation=self.config.get("activation"),
            norm_type=self.config.get("norm_type"),
            use_reversible_instance_norm=self.config.get("use_reversible_instance_norm"),
        )

    def _get_nlinear_model(self):
        torch.serialization.add_safe_globals([NLinearModel, LossCatalog])
        ReproducibilityGate.Config.audit_architecture(self.config)
        return NLinearModel(
            **self._get_common_model_args(),
            shared_weights=self.config.get("shared_weights"),
            const_init=self.config.get("const_init"),
            normalize=self.config.get("normalize"),
            use_static_covariates=self.config.get("use_static_covariates"),
            use_reversible_instance_norm=self.config.get("use_reversible_instance_norm"),
        )

    def _get_dlinear_model(self):
        torch.serialization.add_safe_globals([DLinearModel, LossCatalog])
        ReproducibilityGate.Config.audit_architecture(self.config)
        return DLinearModel(
            **self._get_common_model_args(),
            shared_weights=self.config.get("shared_weights"),
            kernel_size=self.config.get("kernel_size"),
            const_init=self.config.get("const_init"),
            use_static_covariates=self.config.get("use_static_covariates"),
            use_reversible_instance_norm=self.config.get("use_reversible_instance_norm"),
        )

    def _get_tide_model(self):
        torch.serialization.add_safe_globals([TiDEModel, LossCatalog])
        ReproducibilityGate.Config.audit_architecture(self.config)

        # ---- 1. Model construction (STRICT access only) ----
        return TiDEModel(
            **self._get_common_model_args(),
            num_encoder_layers=self.config.get("num_encoder_layers"),
            num_decoder_layers=self.config.get("num_decoder_layers"),
            decoder_output_dim=self.config.get("decoder_output_dim"),
            hidden_size=self.config.get("hidden_size"),
            temporal_width_past=self.config.get("temporal_width_past"),
            temporal_width_future=self.config.get("temporal_width_future"),
            temporal_decoder_hidden=self.config.get("temporal_decoder_hidden"),
            use_layer_norm=self.config.get("use_layer_norm"),
            dropout=self.config.get("dropout"),
            use_static_covariates=self.config.get("use_static_covariates"),
            use_reversible_instance_norm=self.config.get("use_reversible_instance_norm"),
        )
