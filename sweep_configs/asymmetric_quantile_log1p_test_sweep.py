# file: sweep_configs/asymmetric_quantile_log1p_test_sweep.py

def get_sweep_config():
    """
    Configuration for testing AsymmetricQuantileLoss with log1p transformation
    and optimized tau and learning rate on the N-BEATS model.
    """

    sweep_config = {
        "method": "grid",
        "name": "asymmetric_quantile_log1p_test",
        "metric": {"name": "time_series_wise_msle_mean_sb", "goal": "minimize"},
    }

    parameters = {
        # --- N-BEATS Architecture (keeping defaults from existing sweeps) ---
        "num_blocks": {"values": [4]},
        "num_stacks": {"values": [2]},
        "dropout": {"values": [0.3]},
        "layer_widths": {"values": [16]},
        "num_layers": {"values": [2]},
        "activation": {"values": ["LeakyReLU"]},
        "generic_architecture": {"values": [True]},
        # --- Loss Function ---
        "loss_function": {"values": ["AsymmetricQuantileLoss"]},
        "tau": {"values": [0.95]},  # Optimized tau
        "non_zero_weight": {"values": [5.0]}, # Keeping default non_zero_weight from AsymmetricQuantileLoss spec
        "zero_threshold": {"values": [0.01]}, # Keeping default zero_threshold
        # --- Trainer & Optimizer ---
        "n_epochs": {"values": [300]},
        "lr": {"values": [0.01]}, # Optimized learning rate
        "optimizer_cls": {"values": ["Adam"]},
        "weight_decay": {"values": [0.0003]},
        "gradient_clip_val": {"values": [0.64]},
        "early_stopping_patience": {"values": [20]},
        "early_stopping_min_delta": {"values": [0.001]},
        # --- Data Handling & Input/Output (Pipeline A) ---
        "input_chunk_length": {"values": [24]},
        "output_chunk_length": {"values": [36]},
        "output_chunk_shift": {"values": [0]},
        "batch_size": {"values": [8]},
        "target_scaler": {"values": ["MinMaxScaler"]},
        "feature_scaler": {"values": ["MinMaxScaler"]},
        "log_targets": {"values": [True]}, # Corresponds to log1p transformation
        "log_features": {"values": [None]},
        # --- Other ---
        "steps": {"values": [[*range(1, 37)]]},
        "force_reset": {"values": [True]},
        "random_state": {"values": [42]}, # Fixed random state for reproducibility
    }

    sweep_config["parameters"] = parameters
    return sweep_config