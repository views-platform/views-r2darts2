"""
Template for a sweep configuration file, aligned with src/configs.py structure.

This file defines a set of hyperparameters to be used in a hyperparameter sweep
experiment. Each parameter is defined as a dictionary with a 'values' key
specifying a list of options to explore.

The structure of this SWEEP_CONFIG mirrors the configuration dictionaries
defined in `reports/investigations/loss_comparison_exp/src/configs.py`
(e.g., NBEATS_HPS, TRAINER_CONFIG, OPTIMIZER_CONFIG, etc.) to ensure clarity
and direct mapping to the model instantiation process.

Refer to the experiment's documentation for details on required parameters
and their expected types.
"""

SWEEP_CONFIG = {
    # --- Loss Function Specification (top-level as it defines the model type) ---
    "loss_fn_class": {
        "values": [
            "WeightedPenaltyHuberLoss",
            "AsymmetricQuantileLoss",
            "SpikeFocalLoss",
        ]
    },
    "loss_fn_params": {
        "values": [
            # Example for WeightedPenaltyHuberLoss
            {
                "zero_threshold": 0.01,
                "delta": 0.025,
                "non_zero_weight": 5.0,
                "false_positive_weight": 1.0,
                "false_negative_weight": 10.0,
            },
            # Example for AsymmetricQuantileLoss
            {"tau": 0.98, "non_zero_weight": 5.0, "zero_threshold": 0.01},
            # Example for SpikeFocalLoss
            {"alpha": 0.9, "gamma": 1.5, "spike_threshold": 0.75},
        ]
    },
    # --- N-BEATS Model Hyperparameters (maps to NBEATS_HPS in src/configs.py) ---
    "NBEATS_HPS": {
        "input_chunk_length": {"values": [24, 36]},
        "output_chunk_length": {"values": [12, 24, 36]},
        "num_stacks": {"values": [2, 4]},
        "num_blocks": {"values": [2, 4]},
        "dropout": {"values": [0.1, 0.3, 0.5]},
        "layer_widths": {"values": [16, 32, 64]},
        "num_layers": {"values": [2, 3]},
        "activation": {"values": ["ReLU", "LeakyReLU"]},
        "generic_architecture": {"values": [True]},
        "batch_size": {"values": [8, 16]},
        "output_chunk_shift": {"values": [0]},
        # Scalers and log transforms (if implemented in DartsNBEATS wrapper)
        "target_scaler": {"values": ["MinMaxScaler", "StandardScaler"]},  # Example
        "log_targets": {"values": [True, False]},  # Example
        # Other NBEATS_HPS parameters that are usually fixed for a sweep:
        "random_state": {"values": [42]},
        "force_reset": {"values": [True]},
        "n_epochs": {
            "values": [100]
        },  # Max epochs, actual could be less with EarlyStopping
    },
    # --- Trainer Configuration (maps to TRAINER_CONFIG in src/configs.py) ---
    "TRAINER_CONFIG": {
        "accelerator": {"values": ["cpu"]},  # Set to "gpu" if GPU is available
        "enable_progress_bar": {"values": [True]},
        "gradient_clip_val": {"values": [0.5, 1.0]},
    },
    # --- Optimizer Configuration (maps to OPTIMIZER_CONFIG in src/configs.py) ---
    "OPTIMIZER_CONFIG": {
        "lr": {"values": [0.001, 0.0001]},
        "optimizer_cls": {"values": ["Adam", "SGD"]},
        "weight_decay": {"values": [0.0, 0.0001, 0.0003]},
    },
    # --- Learning Rate Scheduler Configuration (maps to LR_SCHEDULER_CONFIG in src/configs.py) ---
    "LR_SCHEDULER_CONFIG": {
        "lr_scheduler_cls": {"values": ["ReduceLROnPlateau"]},
        "lr_scheduler_patience": {"values": [5, 7]},
        "lr_scheduler_factor": {"values": [0.3, 0.5]},
        "lr_scheduler_min_lr": {"values": [1e-6]},
    },
    # --- Early Stopping Callback Configuration (maps to EARLY_STOPPING_CONFIG in src/configs.py) ---
    "EARLY_STOPPING_CONFIG": {
        "early_stopping_patience": {"values": [10, 15, 20]},
        "early_stopping_min_delta": {"values": [0.001, 0.01]},
    },
    # --- Other Experiment-specific Parameters (if any) ---
    "steps": {
        "values": [[*range(1, 37)]]
    },  # Example for 'steps' parameter often used in evaluation
}

# The MODEL_SPECIFIC_OVERRIDES section from the old template is removed as
# the new nested structure of SWEEP_CONFIG makes it redundant for parameter
# overrides within a single sweep definition. If per-model overrides are needed,
# it should be handled by the sweep runner or a more advanced config management system.
