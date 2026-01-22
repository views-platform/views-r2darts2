# file: sweep_configs/tweedie_nlinear_test_sweep.py

def get_sweep_config():
    """
    This sweep is for Experiment 2a of the TweedieLoss diagnostic plan.
    It tests the TweedieLoss with a simple NLinearModel to check for
    baseline convergence and rule out issues with the N-BEATS architecture.
    """

    sweep_config = {
        "method": "grid",
        "name": "tweedie_NLinear_sanity_check",
        "metric": {"name": "val_loss", "goal": "minimize"},
    }

    parameters = {
        # --- Model Architecture (Simplified to NLinear) ---
        "model_cls_name": {"value": "NLinearModel"},

        # --- Loss Function ---
        "loss_function": {"value": "TweedieLoss"},
        "p": {"values": [1.2, 1.5, 1.8]},
        
        # --- Trainer & Optimizer ---
        "n_epochs": {"value": 100},
        "lr": {"values": [1e-3, 1e-4, 1e-5]}, # Search over a reasonable lr range
        "optimizer_cls": {"value": "Adam"},
        "gradient_clip_val": {"value": 1.0},
        
        # --- Data Handling & Input/Output (Pipeline A) ---
        "input_chunk_length": {"value": 24},
        "output_chunk_length": {"value": 36},
        "batch_size": {"value": 32},
        "target_scaler": {"value": "MinMaxScaler"},
        "feature_scaler": {"value": "MinMaxScaler"},
        "log_targets": {"value": True},
        "log_features": {"value": None},
        
        # --- Other ---
        "steps": {"value": [[*range(1, 37)]]},
        "mc_dropout": {"value": False}, # NLinear doesn't use dropout
        "force_reset": {"value": True},
        "random_state": {"value": 42},
    }

    sweep_config["parameters"] = parameters
    return sweep_config
