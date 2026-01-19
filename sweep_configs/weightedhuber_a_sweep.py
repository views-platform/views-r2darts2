# file: sweep_configs/weightedhuber_a_sweep.py

def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This sweep tests the WeightedHuberLoss for Pipeline A: log1p + minmax(0,1).
    """

    sweep_config = {
        'method': 'grid',
        'name': 'weighted_huber_A_pipeline_test',
        'metric': {
            'name': 'time_series_wise_msle_mean_sb',
            'goal': 'minimize'
        },
    }

    parameters = {
        # --- N-BEATS Architecture ---
        'num_blocks': {'values': [4]},
        'num_stacks': {'values': [2]},
        'dropout': {'values': [0.3]},
        'layer_widths': {'values': [16]},
        'num_layers': {'values': [2]},
        'activation': {'values': ['LeakyReLU']},
        'generic_architecture': {'values': [True]},

        # --- Loss Function ---
        'loss_function': {'values': ['WeightedHuberLoss']},
        'delta': {'values': [0.05, 0.1, 0.25]},
        'non_zero_weight': {'values': [2.0, 5.0, 10.0]},
        'zero_threshold': {'values': [0.01]},

        # --- Trainer & Optimizer ---
        'n_epochs': {'values': [300]},
        'lr': {'values': [0.0006]},
        'optimizer_cls': {'values': ['Adam']},
        'weight_decay': {'values': [0.0003]},
        'gradient_clip_val': {'values': [0.64]},
        'lr_scheduler_cls': {'values': ['ReduceLROnPlateau']},
        'lr_scheduler_patience': {'values': [7]},
        'lr_scheduler_factor': {'values': [0.46]},
        'lr_scheduler_min_lr': {'values': [0.00001]},
        'early_stopping_patience': {'values': [20]},
        'early_stopping_min_delta': {'values': [0.001]},
        
        # --- Data Handling & Input/Output ---
        'input_chunk_length': {'values': [24]},
        'output_chunk_length': {'values': [36]},
        'output_chunk_shift': {'values': [0]},
        'batch_size': {'values': [8]},
        'target_scaler': {'values': ['MinMaxScaler']},
        'feature_scaler': {'values': ['MinMaxScaler']},
        'log_targets': {'values': [True]},
        'log_features': {'values': [None]},
        
        # --- Other ---
        'steps': {'values': [[*range(1, 37)]]},
        'mc_dropout': {'values': [True]},
        'force_reset': {'values': [True]},
        'random_state': {'values': [1, 2]},
    }

    sweep_config['parameters'] = parameters
    return sweep_config