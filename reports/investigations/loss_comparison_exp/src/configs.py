"""
Centralized configuration for the loss function comparison experiment.
"""

# 1. Data Generation Configuration
DATA_CONFIG = {
    "n_samples": 300,  # Each timeline is 300 steps long
    "n_timelines": 200,  # Number of distinct timelines (e.g., countries)
    # Parameters for generating country-specific Markov Chain transition probabilities
    "P_PC_BETA_PARAMS": {"alpha": 2, "beta": 20},  # Low conflict propensity on average
    "P_CP_BETA_PARAMS": {"alpha": 5, "beta": 1},  # Short conflict spells on average
    # Parameters for generating country-specific Log-normal magnitudes
    "LOGNORMAL_MEAN_NORMAL_PARAMS": {
        "loc": 0.5,
        "scale": 0.3,
    },  # Mean of log-magnitudes
    "LOGNORMAL_SIGMA_LOGNORMAL_PARAMS": {
        "mean": 0.0,
        "sigma": 0.5,
    },  # Variability of log-magnitudes
}

# 2. Experiment Configuration
EXPERIMENT_CONFIG = {
    "test_split_ratio": 0.2,
    "random_seed": 42,
    "output_dir": "reports/investigations/loss_comparison_exp",
}

# 3. Baseline Model Configuration
BASELINE_CONFIG = {
    # window_size is no longer relevant as the baseline uses the mean of the entire training set.
}

# 4. N-BEATS Model Hyperparameters
# These parameters directly define the N-BEATS model architecture and core behavior.
# Kept consistent for both N-BEATS models to isolate the loss function's effect.
NBEATS_HPS = {
    "input_chunk_length": 24,  # Updated from new parameters
    "output_chunk_length": 36,  # Updated from new parameters
    "num_stacks": 2,  # Updated from new parameters
    "num_blocks": 4,  # Updated from new parameters
    "dropout": 0.3,  # New parameter
    "layer_widths": 16,  # Updated from new parameters
    "num_layers": 2,  # Updated from new parameters
    "activation": "LeakyReLU",  # New parameter
    "generic_architecture": True,  # New parameter
    "batch_size": 8,  # New parameter
    "output_chunk_shift": 0,  # New parameter
    # "target_scaler": "MinMaxScaler",    # Removed: Not a direct NBEATSModel constructor arg
    # "feature_scaler": "MinMaxScaler",   # Removed: Not a direct NBEATSModel constructor arg
    # "log_targets": True,                # Removed: Not a direct NBEATSModel constructor arg
    # "log_features": None,               # Removed: Not a direct NBEATSModel constructor arg
    # "mc_dropout": True,                 # Removed: Not a direct NBEATSModel constructor arg
    "random_state": 2,  # Updated from new parameters
    "force_reset": True,  # Existing, keep as is
    "n_epochs": 2,  # Keep at 2 for quick runs (default from user input was 300)
}

# 5. Trainer Configuration (for PyTorch Lightning Trainer)
TRAINER_CONFIG = {
    "accelerator": "cpu",  # Keep as cpu for default runs, can be overridden
    "enable_progress_bar": True,  # Keep as True
    "gradient_clip_val": 0.64,  # New parameter
}

# 6. Optimizer Configuration
OPTIMIZER_CONFIG = {
    "lr": 0.0003,  # New parameter
    "optimizer_cls": "Adam",  # New parameter (string representation)
    "weight_decay": 0.0003,  # New parameter
}

# 7. Learning Rate Scheduler Configuration
LR_SCHEDULER_CONFIG = {
    "lr_scheduler_cls": "ReduceLROnPlateau",  # New parameter (string representation)
    "lr_scheduler_patience": 7,  # New parameter
    "lr_scheduler_factor": 0.46,  # New parameter
    "lr_scheduler_min_lr": 0.00001,  # New parameter
}

# 8. Early Stopping Callback Configuration
EARLY_STOPPING_CONFIG = {
    "early_stopping_patience": 20,  # Re-enabled
    "early_stopping_min_delta": 0.01,  # New parameter
}


# 9. Model Configurations (specific to each loss function)
MODEL_CONFIGS = {
    "huber_model": {
        "model_type": "NBEATS",
        "loss_fn_class": "WeightedPenaltyHuberLoss",
        "loss_fn_params": {
            "zero_threshold": 0.01,
            "delta": 0.025,
            "non_zero_weight": 5.0,
            "false_positive_weight": 1.0,
            "false_negative_weight": 10.0,
        },
    },
    "quantile_model": {
        "model_type": "NBEATS",
        "loss_fn_class": "AsymmetricQuantileLoss",
        "loss_fn_params": {
            "tau": 0.98,
            "non_zero_weight": 5.0,
            "zero_threshold": 0.01,
        },
    },
}
