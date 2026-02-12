"""
W&B Sweep configuration for AsymmetricQuantileLoss.

This file defines a set of hyperparameters to be used in a Weights & Biases
hyperparameter sweep experiment for the AsymmetricQuantileLoss. This sweep is
designed to work with log1p(y) transformed data.
"""


def get_sweep_config():
    """
    Generates the sweep configuration for AsymmetricQuantileLoss.

    This configuration is designed for hyperparameter sweeps using Weights & Biases,
    and includes parameters for N-BEATS architecture, trainer/optimizer,
    data handling, and AsymmetricQuantileLoss-specific parameters.
    """

    sweep_config = {
        "method": "grid",
        "name": "preliminary_directives_xx",
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # --- N-BEATS Architecture ---
        "num_blocks": {"values": [4]},
        "num_stacks": {"values": [2]},
        "dropout": {"values": [0.3]},
        "layer_widths": {"values": [64]},
        "num_layers": {"values": [3]},
        "activation": {"values": ["LeakyReLU"]},
        "generic_architecture": {"values": [True]},
        # --- Loss Function ---
        "loss_function": {"values": ["AsymmetricQuantileLoss"]},
        "tau": {"values": [0.4]},  # Optimized tau
        "non_zero_weight": {
            "values": [5.0]
        },  # Keeping default non_zero_weight from AsymmetricQuantileLoss spec
        "zero_threshold": {"values": [0.01]},  # Keeping default zero_threshold
        # --- Trainer & Optimizer ---
        "n_epochs": {"values": [300]},
        "lr": {"values": [0.0003]},
        "optimizer_cls": {"values": ["Adam"]},
        "weight_decay": {"values": [0.0003]},
        "gradient_clip_val": {"values": [1]},
        "lr_scheduler_cls": {"values": ["ReduceLROnPlateau"]},
        "lr_scheduler_patience": {"values": [7]},
        "lr_scheduler_factor": {"values": [0.46]},
        "lr_scheduler_min_lr": {"values": [0.00001]},
        "early_stopping_patience": {"values": [20]},
        "early_stopping_min_delta": {"values": [0.01]},
        # --- Data Handling & Input/Output ---
        "input_chunk_length": {"values": [24]},
        "output_chunk_length": {"values": [36]},
        "output_chunk_shift": {"values": [0]},
        "batch_size": {"values": [8]},
        "target_scaler": {"values": ["MinMaxScaler"]},
        "feature_scaler": {"values": ["MinMaxScaler"]},
        "log_targets": {"values": [True]},
        "log_features": {"values": [None]},
        # --- Other ---
        "steps": {"values": [[*range(1, 37)]]},
        "mc_dropout": {"values": [True]},
        "force_reset": {"values": [True]},
        "random_state": {"values": [1, 2, 3, 4, 5]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config
