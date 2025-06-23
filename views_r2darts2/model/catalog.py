from token import NL
from darts.models.forecasting.nbeats import NBEATSModel
from darts.models.forecasting.tft_model import TFTModel
from darts.models.forecasting.tcn_model import TCNModel
from darts.models.forecasting.block_rnn_model import BlockRNNModel
from darts.models.forecasting.transformer_model import TransformerModel
from darts.models.forecasting.nlinear import NLinearModel
from darts.models.forecasting.tide_model import TiDEModel
from darts.models.forecasting.dlinear import DLinearModel
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
import torch.optim.adam
from views_r2darts2.utils.loss import WeightedSmoothL1Loss, WeightedHuberLoss, TimeAwareWeightedHuberLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import OneCycleLR
import torch
import numpy as np
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from views_r2darts2.model.forecaster import DartsForecaster
class ModelCatalog:
    """
    Catalog class to manage and provide access to various forecasting models.
    """

    def __init__(self, config: dict):
        self.models = {
            "NBEATSModel": self._get_nbeats,
            "TFTModel": self._get_tft_model,
            "TCNModel": self._get_tcn_model,
            "BlockRNNModel": self._get_rnn_model,
            "TransformerModel": self._get_transformer_model,
            "NLinearModel": self._get_nlinear_model,
            "TiDEModel": self._get_tide_model,
            "DLinearModel": self._get_dlinear_model,
        }
        self.config = config
        self.device = DartsForecaster.get_device()

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
    

    # def _get_tft_model(self):
    #     torch.serialization.add_safe_globals([TFTModel])
    #     return TFTModel(
    #         input_chunk_length=self.config.get('input_chunk_length', 12 * 6),
    #         output_chunk_length=len(self.config['steps']),
    #         output_chunk_shift=self.config.get('output_chunk_shift', 0),  # Default: 0
    #         add_relative_index=self.config.get('add_relative_index', True),  # Use relative index for temporal features
    #         full_attention=self.config.get('full_attention', False),  # Use full attention mechanism
    #         lstm_layers=self.config.get('lstm_layers', 1),  # Number of LSTM layers
    #         num_attention_heads=self.config.get('num_attention_heads', 2),  # Number of attention heads
    #         hidden_size=self.config.get('hidden_size', 64),  # Size of the hidden layers
    #         dropout=self.config.get('dropout', 0.2),  # Dropout rate
    #         batch_size=self.config.get('batch_size', 512),  # Batch size
    #         loss_fn=torch.nn.HuberLoss(delta=0.1),
    #         model_name=self.config.get('name', 'TFTModel'),  # Model name
    #         norm_type=self.config.get('norm_type', "RMSNorm"),  # Normalization type
    #         force_reset=True,
    #         random_state=self.config.get('random_state', 42),
    #         pl_trainer_kwargs={
    #             "accelerator": "gpu",
    #             "gradient_clip_val": 1.0,  # Prevent exploding gradients
    #             "callbacks": [
    #                 EarlyStopping(monitor="train_loss", patience=self.config.get('early_stopping_patience', 5), mode="min"),
    #                 StochasticWeightAveraging(swa_lrs=self.config.get('stochastic_weight_averaging_swa_lrs', 1e-4)),
    #             ]
    #         },
    #         optimizer_kwargs={
    #             "lr": 3e-4,  # initial learning rate
    #             "weight_decay": 1e-3  # L2 regularization
    #         },
    #         n_epochs=self.config.get('n_epochs', 2),
    #         # lr_scheduler_cls=OneCycleLR,
    #         lr_scheduler_kwargs={
    #             "max_lr": self.config.get('max_lr', 1e-3),
    #             "epochs": self.config.get('n_epochs', 2),
    #             "steps_per_epoch": 118,
    #             "pct_start": self.config.get('pct_start', 0.3),
    #             "anneal_strategy": self.config.get('anneal_strategy', "cos"),
    #         },
    #     )
    
    def _get_tft_model(self):
        torch.serialization.add_safe_globals([TFTModel, WeightedHuberLoss])
        
        # Revised training parameters
        batch_size = 256  # Reduced from 512 for better gradient variety
        training_samples = 180000
        steps_per_epoch = int(np.ceil(training_samples / batch_size))
        
        return TFTModel(
            # Keep temporal configuration unchanged
            input_chunk_length=self.config.get('input_chunk_length', 12*4),
            output_chunk_length=len(self.config['steps']),
            output_chunk_shift=0,
            
            add_relative_index=True,
            full_attention=False,
            lstm_layers=4,
            num_attention_heads=4, 
            hidden_size=512, 
            dropout=0.4,
            
            # Critical training modifications
            batch_size=batch_size,
            loss_fn=WeightedHuberLoss(
                zero_threshold=0.01,  # Adjusted for scaled data
                delta=0.05,  # Tighter error sensitivity
                non_zero_weight=8.0  # Stronger non-zero emphasis
            ),
            model_name='TFTModel',
            norm_type="RMSNorm",  # Better for scaled outputs
            n_epochs=2,  # Essential for rare event learning
            
            # Learning rate schedule
            lr_scheduler_cls=LinearWarmupCosineAnnealingLR,
            lr_scheduler_kwargs={
                "warmup_epochs": 50,
                "max_epochs": 300,
                "warmup_start_lr": 1e-6,
                "eta_min": 1e-7
            },
            
            # Training controls
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config.get('early_stopping_patience', 3),
                        min_delta=self.config.get('early_stopping_min_delta', 0.001),
                        mode="min"
                    ),
                    LearningRateMonitor(log_momentum=True),
                ],
                "enable_progress_bar": True,
                "logger": True,
            },
            optimizer_kwargs={
                "lr": self.config.get('lr', 3e-4),  # Default learning rate
                # "weight_decay": self.config.get('weight_decay', 1e-3),  # Default L2 regularization
            },
        )
    
    def _get_nbeats(self):
        torch.serialization.add_safe_globals([NBEATSModel])
        return NBEATSModel(
            input_chunk_length=self.config.get('input_chunk_length', 12 * 2),
            output_chunk_length=len(self.config['steps']),
            output_chunk_shift=self.config.get('output_chunk_shift', 0),  # Default: 0
            generic_architecture=self.config.get('generic_architecture', True),
            num_stacks=self.config.get('num_stacks', 4),
            num_blocks=self.config.get('num_blocks', 2),
            num_layers=self.config.get('num_layers', 2),
            layer_widths=self.config.get('layer_width', 128),
            activation=self.config.get('activation', 'ReLU'),
            dropout=self.config.get('dropout', 0.2),
            random_state=self.config.get('random_state', 42),
            n_epochs=self.config.get('n_epochs', 2),
            batch_size=self.config.get('batch_size', 128),
            loss_fn=torch.nn.HuberLoss(delta=0.5),
            model_name=self.config.get('name', 'NBEATSModel'),
            force_reset=True,
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "gradient_clip_val": self.config.get('gradient_clip_val', 0.8),
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config.get('early_stopping_patience', 5),
                        min_delta=self.config.get('early_stopping_min_delta', 0.001),
                        mode="min"
                    ),
                    LearningRateMonitor(log_momentum=True, log_weight_decay=True),
                ],
                "enable_progress_bar": True,
                "logger": True,
            },
            optimizer_kwargs={
                "lr": self.config.get('lr', 3e-4),
                "weight_decay": self.config.get('weight_decay', 1e-3)
            },
            lr_scheduler_cls=OneCycleLR,
            lr_scheduler_kwargs={
                "max_lr": self.config.get('max_lr', 1e-3),
                "epochs": self.config.get('n_epochs', 2),
                "steps_per_epoch": 208,
                "pct_start": self.config.get('pct_start', 0.3),
                "anneal_strategy": self.config.get('anneal_strategy', "cos"),
            }
        )

    def _get_tcn_model(self):
        torch.serialization.add_safe_globals([TCNModel])
        return TCNModel(
            input_chunk_length=self.config.get('input_chunk_length', 12 * 6),
            output_chunk_length=len(self.config['steps']),
            output_chunk_shift=self.config.get('output_chunk_shift', 0),  # Default: 0
            kernel_size=2,
            num_filters=1,
            # dilation_base=2,
            # weight_norm=True, #BUG!
            dropout=0.25,
            force_reset=True,
            save_checkpoints=True,
            batch_size=self.config.get('batch_size', 64),
            model_name=self.config.get('name', 'TCNModel'),
            random_state=self.config.get('random_state', 42),
            n_epochs=self.config.get('n_epochs', 2),
            loss_fn=torch.nn.HuberLoss(delta=0.45),
            use_reversible_instance_norm=True, # https://openreview.net/forum?id=cGDAkQo1C0p
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "gradient_clip_val": self.config.get('gradient_clip_val', 0.8),
                # "callbacks": [
                #     EarlyStopping(
                #         monitor="train_loss",
                #         patience=self.config.get('early_stopping_patience', 5),
                #         min_delta=self.config.get('early_stopping_min_delta', 0.001),
                #         mode="min"
                #     ),
                #     LearningRateMonitor(log_momentum=True, log_weight_decay=True),
                # ],
                # "enable_progress_bar": True,
                # "logger": True,
            },
            optimizer_kwargs={
                "lr": self.config.get('lr', 3e-4),
                "weight_decay": self.config.get('weight_decay', 1e-3)
            },
            lr_scheduler_cls=OneCycleLR,
            lr_scheduler_kwargs={
                "max_lr": self.config.get('max_lr', 1e-3),
                "epochs": self.config.get('n_epochs', 2),
                "steps_per_epoch": 833,
                "pct_start": self.config.get('pct_start', 0.3),
                "anneal_strategy": self.config.get('anneal_strategy', "cos"),
            },
        )
    
    def _get_rnn_model(self):
        torch.serialization.add_safe_globals([BlockRNNModel])
        return BlockRNNModel(
            input_chunk_length=self.config.get('input_chunk_length', 12 * 4),
            output_chunk_length=len(self.config['steps']),
            output_chunk_shift=self.config.get('output_chunk_shift', 0),  # Default: 0
            model=self.config.get('rnn_type', 'LSTM'),  # Choose between 'LSTM', 'GRU', or 'RNN'
            hidden_dim=self.config.get('hidden_dim', 25),  # Size of the hidden layers
            activation=self.config.get('activation', 'ReLU'),
            n_rnn_layers = 1,
            dropout=self.config.get('dropout', 0.2),  # Dropout rate for regularization
            batch_size=self.config.get('batch_size', 256),  # Batch size for training
            n_epochs=self.config.get('n_epochs', 7),  # Number of training epochs
            loss_fn=torch.nn.HuberLoss(delta=0.5),
            pl_trainer_kwargs={
                "accelerator": "gpu",
                # "gradient_clip_val": self.config.get('gradient_clip_val', 0.8),
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config.get('early_stopping_patience', 5),
                        min_delta=self.config.get('early_stopping_min_delta', 0.001),
                        mode="min"
                    ),
                    LearningRateMonitor(log_momentum=True, log_weight_decay=True),
                ],
                "enable_progress_bar": True,
                "logger": True,
            },
            optimizer_kwargs={
                "lr": self.config.get('lr', 1e-4),  # Learning rate
                "weight_decay": self.config.get('weight_decay', 1e-4),  # L2 regularization
            },
            lr_scheduler_cls=OneCycleLR,
            lr_scheduler_kwargs={
                "max_lr": self.config.get('max_lr', 1e-3),
                "epochs": self.config.get('n_epochs', 2),
                "steps_per_epoch": 208,
                "pct_start": self.config.get('pct_start', 0.3),
                "anneal_strategy": self.config.get('anneal_strategy', "cos"),
            },
            use_reversible_instance_norm=True,
            model_name=self.config.get('name', 'BlockRNNModel'),  # Model name
            random_state=self.config.get('random_state', 42),  # Random seed for reproducibility
            force_reset=True,  # Reset the model if it already exists
        )

    def _get_transformer_model(self):
        torch.serialization.add_safe_globals([TransformerModel])
        return TransformerModel(
            input_chunk_length=self.config.get('input_chunk_length', 12 * 6),  # Default: 72
            output_chunk_length=len(self.config['steps']),  # Output chunk length based on steps
            output_chunk_shift=self.config.get('output_chunk_shift', 0),  # Default: 0
            d_model=self.config.get('d_model', 64),  # Default: 64
            nhead=self.config.get('num_attention_heads', 4),  # Default: 4 attention heads
            num_encoder_layers=self.config.get('num_encoder_layers', 3),  # Default: 3 encoder layers
            num_decoder_layers=self.config.get('num_decoder_layers', 3),  # Default: 3 decoder layers
            dim_feedforward=self.config.get('dim_feedforward', 512),  # Default: 512
            dropout=self.config.get('dropout', 0.1),  # Default: 0.1
            activation=self.config.get('activation', 'ReLU'),  # Default: 'ReLU'
            norm_type=self.config.get('norm_type', None),  # Default: None
            batch_size=self.config.get('batch_size', 256),  # Default: 32
            n_epochs=self.config.get('n_epochs', 2),  # Default: 100
            loss_fn=torch.nn.HuberLoss(delta=0.5),
            model_name=self.config.get('name', 'TransformerModel'),  # Model name
            random_state=self.config.get('random_state', 42),  # Random seed for reproducibility
            force_reset=True,  # Reset the model if it already exists
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config.get('early_stopping_patience', 5),
                        min_delta=self.config.get('early_stopping_min_delta', 0.001),
                        mode="min"
                    ),
                    LearningRateMonitor(log_momentum=True, log_weight_decay=True),
                ],
                "enable_progress_bar": True,
                "logger": True,
            },
            optimizer_kwargs={
                "lr": self.config.get('lr', 3e-4),  # Default learning rate
                "weight_decay": self.config.get('weight_decay', 1e-3),  # Default L2 regularization
            },
            lr_scheduler_cls=OneCycleLR,
            lr_scheduler_kwargs={
                "max_lr": self.config.get('max_lr', 1e-3),  # Maximum learning rate
                "epochs": self.config.get('n_epochs', 2),  # Number of epochs
                "steps_per_epoch": self.config.get('steps_per_epoch', 208),  # Steps per epoch
                "pct_start": self.config.get('pct_start', 0.3),  # Percentage of steps for increasing LR
                "anneal_strategy": self.config.get('anneal_strategy', "cos"),  # Cosine annealing
            },
        )
    
    def _get_nlinear_model(self):
        torch.serialization.add_safe_globals([NLinearModel])
        return NLinearModel(
            input_chunk_length=self.config.get('input_chunk_length', 12 * 6),  # Default: 72
            output_chunk_length=len(self.config['steps']),  # Output chunk length based on steps
            output_chunk_shift=self.config.get('output_chunk_shift', 0),  # Default: 0
            shared_weights=self.config.get('shared_weights', False),  # Default: False
            const_init=self.config.get('const_init', True),  # Default: True
            normalize=self.config.get('normalize', False),  # Default: False
            use_static_covariates=self.config.get('use_static_covariates', True),  # Default: True
            batch_size=self.config.get('batch_size', 64),  # Default: 64
            n_epochs=self.config.get('n_epochs', 2),  # Default: 2
            loss_fn=torch.nn.HuberLoss(delta=0.5),  # Default loss function
            model_name=self.config.get('name', 'NLinearModel'),  # Model name
            random_state=self.config.get('random_state', 42),  # Random seed for reproducibility
            force_reset=True,  # Reset the model if it already exists
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config.get('early_stopping_patience', 5),
                        min_delta=self.config.get('early_stopping_min_delta', 0.001),
                        mode="min"
                    ),
                    LearningRateMonitor(log_momentum=True, log_weight_decay=True),
                ],
                "enable_progress_bar": True,
                "logger": True,
            },
            optimizer_kwargs={
                "lr": self.config.get('lr', 3e-4),  # Default learning rate
                "weight_decay": self.config.get('weight_decay', 1e-3),  # Default L2 regularization
            },
            lr_scheduler_cls=OneCycleLR,
            lr_scheduler_kwargs={
                "max_lr": self.config.get('max_lr', 1e-3),  # Maximum learning rate
                "epochs": self.config.get('n_epochs', 2),  # Number of epochs
                "steps_per_epoch": self.config.get('steps_per_epoch', 208),  # Steps per epoch
                "pct_start": self.config.get('pct_start', 0.3),  # Percentage of steps for increasing LR
                "anneal_strategy": self.config.get('anneal_strategy', "cos"),  # Cosine annealing
            },
        )
    
    def _get_dlinear_model(self):
        torch.serialization.add_safe_globals([DLinearModel])
        return DLinearModel(
            input_chunk_length=self.config.get('input_chunk_length', 12 * 6),  # Default: 72
            output_chunk_length=len(self.config['steps']),  # Output chunk length based on steps
            output_chunk_shift=self.config.get('output_chunk_shift', 0),  # Default: 0
            shared_weights=self.config.get('shared_weights', False),  # Default: False
            kernel_size=self.config.get('kernel_size', 25),  # Default: 25
            const_init=self.config.get('const_init', True),  # Default: True
            use_static_covariates=self.config.get('use_static_covariates', True),  # Default: True
            batch_size=self.config.get('batch_size', 64),  # Default: 64
            n_epochs=self.config.get('n_epochs', 2),  # Default: 2
            loss_fn=torch.nn.HuberLoss(delta=0.5),  # Default loss function
            model_name=self.config.get('name', 'DLinearModel'),  # Model name
            random_state=self.config.get('random_state', 42),  # Random seed for reproducibility
            force_reset=True,  # Reset the model if it already exists
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config.get('early_stopping_patience', 5),
                        min_delta=self.config.get('early_stopping_min_delta', 0.001),
                        mode="min"
                    ),
                    LearningRateMonitor(log_momentum=True, log_weight_decay=True),
                ],
                "enable_progress_bar": True,
                "logger": True,
            },
            optimizer_kwargs={
                "lr": self.config.get('lr', 3e-4),  # Default learning rate
                "weight_decay": self.config.get('weight_decay', 1e-3),  # Default L2 regularization
            },
            lr_scheduler_cls=OneCycleLR,
            lr_scheduler_kwargs={
                "max_lr": self.config.get('max_lr', 1e-3),  # Maximum learning rate
                "epochs": self.config.get('n_epochs', 2),  # Number of epochs
                "steps_per_epoch": self.config.get('steps_per_epoch', 208),  # Steps per epoch
                "pct_start": self.config.get('pct_start', 0.3),  # Percentage of steps for increasing LR
                "anneal_strategy": self.config.get('anneal_strategy', "cos"),  # Cosine annealing
            },
        )
        
    def _get_tide_model(self):
        torch.serialization.add_safe_globals([TiDEModel, WeightedHuberLoss, TimeAwareWeightedHuberLoss])
        batch_size = 64  # Reduced from 512 for better gradient variety
        training_samples = 180000
        steps_per_epoch = int(np.ceil(training_samples / batch_size))
        return TiDEModel(
            input_chunk_length=self.config.get('input_chunk_length', 12 * 4),  # Default: 72
            output_chunk_length=len(self.config['steps']),  # Output chunk length based on steps
            output_chunk_shift=self.config.get('output_chunk_shift', 0),  # Default: 0
            num_encoder_layers=self.config.get('num_encoder_layers', 1),  # Default: 1 encoder layer
            num_decoder_layers=self.config.get('num_decoder_layers', 1),  # Default: 1 decoder layer
            decoder_output_dim=self.config.get('decoder_output_dim', 16),  # Default: 16
            hidden_size=self.config.get('hidden_size', 128),  # Default: 128
            temporal_width_past=self.config.get('temporal_width_past', 4),  # Default: 4
            temporal_width_future=self.config.get('temporal_width_future', 4),  # Default: 4
            temporal_hidden_size_past=self.config.get('temporal_hidden_size_past', None),  # Default: None
            temporal_hidden_size_future=self.config.get('temporal_hidden_size_future', None),  # Default: None
            temporal_decoder_hidden=self.config.get('temporal_decoder_hidden', 32),  # Default: 32
            use_layer_norm=self.config.get('use_layer_norm', False),  # Default: False
            dropout=self.config.get('dropout', 0.4),  # Default: 0.1
            use_static_covariates=self.config.get('use_static_covariates', True),  # Default: True
            batch_size=self.config.get('batch_size', batch_size),  # Default: 64
            n_epochs=self.config.get('n_epochs', 2),  # Default: 2
            loss_fn=WeightedHuberLoss(
                zero_threshold=0.01,
                delta=0.05, #0.05
                non_zero_weight=6.0,
            ),
            # loss_fn = TimeAwareWeightedHuberLoss(non_zero_weight=5.0, zero_weight=1.0, decay_factor=0.1, delta=0.1),
            model_name=self.config.get('name', 'TiDEModel'),  # Model name
            random_state=self.config.get('random_state', 42),  # Random seed for reproducibility
            force_reset=True,  # Reset the model if it already exists
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "callbacks": [
                    EarlyStopping(
                        monitor="train_loss",
                        patience=self.config.get('early_stopping_patience', 3),
                        min_delta=self.config.get('early_stopping_min_delta', 0.001),
                        mode="min"
                    ),
                    LearningRateMonitor(log_momentum=True),
                ],
                "enable_progress_bar": True,
                "logger": True,
            },
            optimizer_kwargs={
                "lr": self.config.get('lr', 2e-3),  # Default learning rate was 3e-4
                "weight_decay": self.config.get('weight_decay', 1e-5),  # Default L2 regularization
            },
            # lr_scheduler_cls=OneCycleLR,
            # lr_scheduler_kwargs={
            #     "max_lr": self.config.get('max_lr', 1e-3),  # Maximum learning rate
            #     "epochs": self.config.get('n_epochs', 2),  # Number of epochs
            #     "steps_per_epoch": self.config.get('steps_per_epoch', steps_per_epoch),  # Steps per epoch
            #     "pct_start": self.config.get('pct_start', 0.3),  # Percentage of steps for increasing LR
            #     "anneal_strategy": self.config.get('anneal_strategy', "cos"),  # Cosine annealing
            # },
            lr_scheduler_cls=LinearWarmupCosineAnnealingLR,
            lr_scheduler_kwargs={
                "warmup_epochs": 50,
                "max_epochs": 300,
                "warmup_start_lr": 1e-6,
                "eta_min": 1e-7
            },

        )