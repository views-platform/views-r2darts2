# file: sweep_configs/tweedie_focused_grid_sweep.py

def get_sweep_config():
    """
    This sweep is for Experiment 2 of the TweedieLoss diagnostic plan.
    It performs a focused grid search on both the learning rate 'lr' and the
    Tweedie power parameter 'p'.

    !!! ACTION REQUIRED !!!
    Before running, you must replace the placeholder learning rates in the
    'lr' parameter with the stable range you identified from the
    'tweedie_lr_finder_experiment' (Experiment 1).
    """

    sweep_config = {
        "method": "grid",
        "name": "tweedie_focused_grid_experiment",
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
        # --- Loss Function (p is swept) ---
        "loss_function": {"value": "TweedieLoss"},
        "p": {"values": [1.2, 1.4, 1.6, 1.8]}, # Finer grid for p
        # --- Trainer & Optimizer (lr is swept in a focused range) ---
        "n_epochs": {"value": 300},
        "lr": {
            # !!! REPLACE THESE VALUES !!!
            # Use the stable learning rate range found in Experiment 1.
            # For example: "values": [1e-5, 5e-5, 1e-4]
            "values": ["!!!_REPLACE_ME_!!!", "!!!_REPLACE_ME_!!!"]
        },
        "optimizer_cls": {"value": "Adam"},
        "weight_decay": {"value": 0.0003},
        "gradient_clip_val": {"value": 0.64},
        "lr_scheduler_cls": {"value": "ReduceLROnPlateau"},
        "lr_scheduler_patience": {"value": [7]},
        "lr_scheduler_factor": {"value": [0.46]},
        "lr_scheduler_min_lr": {"value": [0.00001]},
        "early_stopping_patience": {"value": [20]},
        "early_stopping_min_delta": {"value": [0.001]},
        # --- Data Handling & Input/Output (using Pipeline A as a baseline) ---
        "input_chunk_length": {"value": 24},
        "output_chunk_length": {"value": 36},
        "batch_size": {"value": [8]},
        "target_scaler": {"value": "MinMaxScaler"},
        "feature_scaler": {"value": "MinMaxScaler"},
        "log_targets": {"value": True},
        "log_features": {"value": None},
        # --- Other ---
        "steps": {"value": [[*range(1, 37)]]},
        "mc_dropout": {"value": True},
        "force_reset": {"value": True},
        "random_state": {"value": 42},
    }

    sweep_config["parameters"] = parameters
    return sweep_config
