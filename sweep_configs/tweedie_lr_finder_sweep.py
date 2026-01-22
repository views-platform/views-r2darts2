# file: sweep_configs/tweedie_lr_finder_sweep.py

def get_sweep_config():
    """
    This sweep is designed for Experiment 1 of the TweedieLoss diagnostic plan.
    It aims to find a stable learning rate for the TweedieLoss by sweeping
    over a wide range of learning rates on a logarithmic scale.

    - p is fixed to 1.5.
    - lr uses a log-uniform distribution between 1e-6 and 1e-2.
    """

    sweep_config = {
        "method": "random",  # Use random search for a log-uniform distribution
        "name": "tweedie_LR_finder_experiment",
        "metric": {"name": "val_loss", "goal": "minimize"},
    }

    parameters = {
        # --- N-BEATS Architecture (kept simple and fixed) ---
        "num_blocks": {"value": 4},
        "num_stacks": {"value": 2},
        "dropout": {"value": 0.3},
        "layer_widths": {"value": 16},
        "num_layers": {"value": 2},
        "activation": {"value": "LeakyReLU"},
        "generic_architecture": {"value": True},
        # --- Loss Function (p is fixed) ---
        "loss_function": {"value": "TweedieLoss"},
        "p": {"value": 1.5},
        # --- Trainer & Optimizer (lr is swept) ---
        "n_epochs": {"value": 100}, # Shorter epochs for a faster search
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-2,
        },
        "optimizer_cls": {"value": "Adam"},
        "weight_decay": {"value": 0.0003},
        "gradient_clip_val": {"value": 0.64},
        # --- Data Handling & Input/Output (using Pipeline A as a baseline) ---
        "input_chunk_length": {"value": 24},
        "output_chunk_length": {"value": 36},
        "batch_size": {"value": 8},
        "target_scaler": {"value": "MinMaxScaler"},
        "feature_scaler": {"value": "MinMaxScaler"},
        "log_targets": {"value": True},
        "log_features": {"value": None},
        # --- Other ---
        "steps": {"value": [*range(1, 37)]},
        "mc_dropout": {"value": True},
        "force_reset": {"value": True},
        "random_state": {"value": 42}, # Use a single random state for consistency
    }

    sweep_config["parameters"] = parameters
    return sweep_config
