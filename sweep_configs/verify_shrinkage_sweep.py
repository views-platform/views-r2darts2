# file: sweep_configs/verify_shrinkage_sweep.py

def get_sweep_config():
    """
    Verification sweep for ShrinkageLoss.
    Tests that a non-default 'a' is correctly passed.
    """

    sweep_config = {
        "method": "grid",
        "name": "VERIFY_ShrinkageLoss",
        "metric": {"name": "val_loss", "goal": "minimize"},
    }

    parameters = {
        # --- Verification Target ---
        "loss_function": {"values": ["ShrinkageLoss"]},
        "a": {"values": [50.0]}, # Unique, non-default value

        # --- Fixed dummy parameters for a single, fast run ---
        "n_epochs": {"values": [1]},
        "lr": {"values": [0.001]},
        "log_targets": {"values": [True]},
        "target_scaler": {"values": ["MinMaxScaler"]},
        
        # --- Minimal N-BEATS architecture ---
        "num_blocks": {"values": [1]},
        "num_stacks": {"values": [1]},
        "layer_widths": {"values": [16]},
        "num_layers": {"values": [1]},
        "input_chunk_length": {"values": [24]},
        "output_chunk_length": {"values": [36]},
        "random_state": {"values": [42]},
    }

    sweep_config["parameters"] = parameters
    return sweep_config
