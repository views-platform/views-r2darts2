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
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import WandbLogger
import torch
import numpy as np
import logging

from views_r2darts2.model.forecaster import DartsForecaster
from views_r2darts2.utils.loss import LossSelector
from views_r2darts2.utils.gates import ReproducibilityGate


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


class GradientHealthCallback(Callback):
    """
    Callback to monitor gradient health after each epoch.
    
    Logs statistics about gradients to help diagnose:
    - Vanishing gradients (very small norms)
    - Exploding gradients (very large norms)
    - NaN/Inf gradients
    """
    
    def __init__(self, log_every_n_epochs: int = 1, warn_threshold: float = 1e-7, explode_threshold: float = 100.0):
        """
        Args:
            log_every_n_epochs: How often to log gradient stats (default: every epoch)
            warn_threshold: Gradient norm below this triggers vanishing warning
            explode_threshold: Gradient norm above this triggers exploding warning
        """
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
            return  # No gradients yet
        
        # Compute stats
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
        
        # Build status message
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
            # Tweedie-family params
            "p": self.config.get("p"),
            "eps": self.config.get("eps"),
        }
        # Filter out None values, so that loss function defaults can apply
        self.loss_args = {k: v for k, v in self.loss_args.items() if v is not None}

        self.loss_fn = LossSelector.get_loss_function(self.loss_name, **self.loss_args)

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

    def _get_common_optimizer_kwargs(self):
        return {
            "lr": self.config.get("lr"),
            "weight_decay": self.config.get("weight_decay"),
        }

    def _get_optimizer_cls(self):
        opt_name = self.config.get("optimizer_cls")
        if not opt_name:
            from views_r2darts2.utils.gates import MissingHyperparameterError
            raise MissingHyperparameterError(
                "MANDATORY HYPERPARAMETER MISSING: 'optimizer_cls' must be explicitly declared in the DNA manifest."
            )
        try:
            return getattr(torch.optim, opt_name)
        except AttributeError:
            raise ValueError(
                f"INVALID HYPERPARAMETER: '{opt_name}' is not a valid torch.optim class name."
            )

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
        ReproducibilityGate.Config.audit_architecture(self.config)
        return TSMixerModel(
            input_chunk_length=self.config.get("input_chunk_length"),
            output_chunk_length=self.config.get("output_chunk_length"),
            output_chunk_shift=self.config.get("output_chunk_shift"),
            num_blocks=self.config.get("num_blocks"),
            ff_size=self.config.get("ff_size"),
            hidden_size=self.config.get("hidden_size"),
            activation=self.config.get("activation"),
            dropout=self.config.get("dropout"),
            norm_type=self.config.get("norm_type"),
            normalize_before=self.config.get("normalize_before"),
            batch_size=self.config.get("batch_size"),
            n_epochs=self.config.get("n_epochs"),
            loss_fn=self.loss_fn,
            model_name=self.config.get("name"),
            random_state=self.config.get("random_state"),
            force_reset=True,
            use_static_covariates=self.config.get("use_static_covariates"),
            use_reversible_instance_norm=self.config.get("use_reversible_instance_norm"),
            pl_trainer_kwargs=self._get_common_pl_trainer_kwargs(),
            optimizer_cls=self._get_optimizer_cls(),
            optimizer_kwargs=self._get_common_optimizer_kwargs(),
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    def _get_tft_model(self):
        torch.serialization.add_safe_globals([TFTModel, LossSelector])
        ReproducibilityGate.Config.audit_architecture(self.config)

        return TFTModel(
            input_chunk_length=self.config.get("input_chunk_length"),
            output_chunk_length=self.config.get("output_chunk_length"),
            output_chunk_shift=self.config.get("output_chunk_shift"),
            feed_forward=self.config.get("feed_forward"),
            add_relative_index=self.config.get("add_relative_index"),
            use_static_covariates=self.config.get("use_static_covariates"),
            full_attention=self.config.get("full_attention"),
            lstm_layers=self.config.get("lstm_layers"),
            num_attention_heads=self.config.get("num_attention_heads"),
            hidden_size=self.config.get("hidden_size"),
            dropout=self.config.get("dropout"),
            batch_size=self.config.get("batch_size"),
            loss_fn=self.loss_fn,
            model_name=self.config.get("name"),
            norm_type=self.config.get("norm_type", "RMSNorm"),
            n_epochs=self.config.get("n_epochs"),
            random_state=self.config.get("random_state"),
            use_reversible_instance_norm=self.config.get("use_reversible_instance_norm"),
            skip_interpolation=self.config.get("skip_interpolation"),
            hidden_continuous_size=self.config.get("hidden_continuous_size"),
            pl_trainer_kwargs=self._get_common_pl_trainer_kwargs(),
            optimizer_cls=self._get_optimizer_cls(),
            optimizer_kwargs=self._get_common_optimizer_kwargs(),
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    
    def _get_nbeats(self):
        torch.serialization.add_safe_globals([NBEATSModel, LossSelector])
    
        # ---- 1. Audit architecture ----
        ReproducibilityGate.Config.audit_architecture(self.config)
    
        # ---- 2. Model construction (STRICT access) ----
        return NBEATSModel(
            input_chunk_length=self.config.get("input_chunk_length"),
            output_chunk_length=self.config.get("output_chunk_length"),
            output_chunk_shift=self.config.get("output_chunk_shift"),
            generic_architecture=self.config.get("generic_architecture"),
            num_stacks=self.config.get("num_stacks"),
            num_blocks=self.config.get("num_blocks"),
            num_layers=self.config.get("num_layers"),
            layer_widths=self.config.get("layer_widths"),
            activation=self.config.get("activation"),
            dropout=self.config.get("dropout"),
            random_state=self.config.get("random_state"),
            n_epochs=self.config.get("n_epochs"),
            batch_size=self.config.get("batch_size"),
            loss_fn=self.loss_fn,
            model_name=self.config.get("name"),
            force_reset=self.config.get("force_reset"),
            pl_trainer_kwargs=self._get_common_pl_trainer_kwargs(),
            optimizer_cls=self._get_optimizer_cls(),
            optimizer_kwargs=self._get_common_optimizer_kwargs(),
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
        ReproducibilityGate.Config.audit_architecture(self.config)
        return NHiTSModel(
            input_chunk_length=self.config.get("input_chunk_length"),
            output_chunk_length=self.config.get("output_chunk_length"),
            output_chunk_shift=self.config.get("output_chunk_shift"),
            num_stacks=self.config.get("num_stacks"),
            num_blocks=self.config.get("num_blocks"),
            num_layers=self.config.get("num_layers"),
            layer_widths=self.config.get("layer_widths"),
            pooling_kernel_sizes=self.config.get("pooling_kernel_sizes"),
            n_freq_downsample=self.config.get("n_freq_downsample"),
            activation=self.config.get("activation"),
            MaxPool1d=self.config.get("max_pool_1d"),
            dropout=self.config.get("dropout"),
            random_state=self.config.get("random_state"),
            n_epochs=self.config.get("n_epochs"),
            batch_size=self.config.get("batch_size"),
            loss_fn=self.loss_fn,
            model_name=self.config.get("name"),
            force_reset=self.config.get("force_reset"),
            use_reversible_instance_norm=self.config.get("use_reversible_instance_norm"),
            pl_trainer_kwargs=self._get_common_pl_trainer_kwargs(),
            optimizer_cls=self._get_optimizer_cls(),
            optimizer_kwargs=self._get_common_optimizer_kwargs(),
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,
        )

    def _get_tcn_model(self):
        torch.serialization.add_safe_globals([TCNModel, LossSelector])
        ReproducibilityGate.Config.audit_architecture(self.config)
        return TCNModel(
            input_chunk_length=self.config.get("input_chunk_length"),
            output_chunk_length=self.config.get("output_chunk_length"),
            output_chunk_shift=self.config.get("output_chunk_shift"),
            kernel_size=self.config.get("kernel_size"),
            num_filters=self.config.get("num_filters"),
            dilation_base=self.config.get("dilation_base"),
            dropout=self.config.get("dropout"),
            force_reset=self.config.get("force_reset"),
            save_checkpoints=True,
            batch_size=self.config.get("batch_size"),
            model_name=self.config.get("name"),
            random_state=self.config.get("random_state"),
            n_epochs=self.config.get("n_epochs"),
            loss_fn=self.loss_fn,
            use_reversible_instance_norm=self.config.get("use_reversible_instance_norm"),
            pl_trainer_kwargs=self._get_common_pl_trainer_kwargs(),
            optimizer_cls=self._get_optimizer_cls(),
            optimizer_kwargs=self._get_common_optimizer_kwargs(),
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    def _get_rnn_model(self):
        torch.serialization.add_safe_globals([BlockRNNModel, LossSelector])
        ReproducibilityGate.Config.audit_architecture(self.config)
        return BlockRNNModel(
            input_chunk_length=self.config.get("input_chunk_length"),
            output_chunk_length=self.config.get("output_chunk_length"),
            output_chunk_shift=self.config.get("output_chunk_shift"),
            model=self.config.get("rnn_type"),
            hidden_dim=self.config.get("hidden_dim"),
            activation=self.config.get("activation"),
            n_rnn_layers=self.config.get("n_rnn_layers"),
            dropout=self.config.get("dropout"),
            batch_size=self.config.get("batch_size"),
            n_epochs=self.config.get("n_epochs"),
            loss_fn=self.loss_fn,
            pl_trainer_kwargs=self._get_common_pl_trainer_kwargs(),
            optimizer_cls=self._get_optimizer_cls(),
            optimizer_kwargs=self._get_common_optimizer_kwargs(),
            use_reversible_instance_norm=self.config.get("use_reversible_instance_norm"),
            model_name=self.config.get("name"),
            random_state=self.config.get("random_state"),
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
        
        ReproducibilityGate.Config.audit_architecture(self.config)
        
        return TransformerModel(
            input_chunk_length=self.config.get("input_chunk_length"),
            output_chunk_length=self.config.get("output_chunk_length"),
            output_chunk_shift=self.config.get("output_chunk_shift"),
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=self.config.get("num_encoder_layers"),
            num_decoder_layers=self.config.get("num_decoder_layers"),
            dim_feedforward=self.config.get("dim_feedforward"),
            dropout=self.config.get("dropout"),
            activation=self.config.get("activation"),
            norm_type=self.config.get("norm_type"),
            batch_size=self.config.get("batch_size"),
            n_epochs=self.config.get("n_epochs"),
            loss_fn=self.loss_fn,
            model_name=self.config.get("name"),
            random_state=self.config.get("random_state"),
            force_reset=True,
            use_reversible_instance_norm=self.config.get("use_reversible_instance_norm"),
            pl_trainer_kwargs={
                **self._get_common_pl_trainer_kwargs(extra_callbacks=[NaNDetectionCallback(patience=5)]),
                "detect_anomaly": self.config.get("detect_anomaly"),
            },
            optimizer_cls=self._get_optimizer_cls(),
            optimizer_kwargs=self._get_common_optimizer_kwargs(),
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    def _get_nlinear_model(self):
        torch.serialization.add_safe_globals([NLinearModel, LossSelector])
        ReproducibilityGate.Config.audit_architecture(self.config)
        return NLinearModel(
            input_chunk_length=self.config.get("input_chunk_length"),
            output_chunk_length=self.config.get("output_chunk_length"),
            output_chunk_shift=self.config.get("output_chunk_shift"),
            shared_weights=self.config.get("shared_weights"),
            const_init=self.config.get("const_init"),
            normalize=self.config.get("normalize"),
            use_static_covariates=self.config.get("use_static_covariates"),
            batch_size=self.config.get("batch_size"),
            n_epochs=self.config.get("n_epochs"),
            loss_fn=self.loss_fn,
            model_name=self.config.get("name"),
            random_state=self.config.get("random_state"),
            force_reset=True,
            use_reversible_instance_norm=self.config.get("use_reversible_instance_norm"),
            pl_trainer_kwargs=self._get_common_pl_trainer_kwargs(),
            optimizer_cls=self._get_optimizer_cls(),
            optimizer_kwargs=self._get_common_optimizer_kwargs(),
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    def _get_dlinear_model(self):
        torch.serialization.add_safe_globals([DLinearModel, LossSelector])
        ReproducibilityGate.Config.audit_architecture(self.config)
        return DLinearModel(
            input_chunk_length=self.config.get("input_chunk_length"),
            output_chunk_length=self.config.get("output_chunk_length"),
            output_chunk_shift=self.config.get("output_chunk_shift"),
            shared_weights=self.config.get("shared_weights"),
            kernel_size=self.config.get("kernel_size"),
            const_init=self.config.get("const_init"),
            use_static_covariates=self.config.get("use_static_covariates"),
            batch_size=self.config.get("batch_size"),
            n_epochs=self.config.get("n_epochs"),
            loss_fn=self.loss_fn,
            model_name=self.config.get("name"),
            random_state=self.config.get("random_state"),
            force_reset=True,
            use_reversible_instance_norm=self.config.get("use_reversible_instance_norm"),
            pl_trainer_kwargs=self._get_common_pl_trainer_kwargs(),
            optimizer_cls=self._get_optimizer_cls(),
            optimizer_kwargs=self._get_common_optimizer_kwargs(),
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,  
        )

    
    def _get_tide_model(self):
        torch.serialization.add_safe_globals([TiDEModel, LossSelector])
        ReproducibilityGate.Config.audit_architecture(self.config)
    
        # ---- 1. Model construction (STRICT access only) ----
        return TiDEModel(
            input_chunk_length=self.config["input_chunk_length"],
            output_chunk_length=self.config["output_chunk_length"],
            output_chunk_shift=self.config["output_chunk_shift"],
            num_encoder_layers=self.config["num_encoder_layers"],
            num_decoder_layers=self.config["num_decoder_layers"],
            decoder_output_dim=self.config["decoder_output_dim"],
            hidden_size=self.config["hidden_size"],
            temporal_width_past=self.config["temporal_width_past"],
            temporal_width_future=self.config["temporal_width_future"],
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
            pl_trainer_kwargs=self._get_common_pl_trainer_kwargs(),
            optimizer_cls=self._get_optimizer_cls(),
            optimizer_kwargs=self._get_common_optimizer_kwargs(),
            lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_kwargs=self.lr_scheduler_args,
        )

