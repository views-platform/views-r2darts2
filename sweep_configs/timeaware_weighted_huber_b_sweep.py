# file: sweep_configs/timeaware_weighted_huber_b_sweep.py

def get_sweep_config():
    """
    Contains the configuration for hyperparameter sweeps using WandB.
    This sweep tests the TimeAwareWeightedHuberLoss for Pipeline B: raw_counts.
    """

    sweep_config = {
        'method': 'grid',
        'name': 'time_aware_weighted_huber_B_pipeline_test',
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
        'loss_function': {'values': ['TimeAwareWeightedHuberLoss']},
        'delta': {'values': [50.0, 100.0, 200.0]},
        'decay_factor': {'values': [0.9, 0.95]},
        'zero_weight': {'values': [1.0]},
        'non_zero_weight': {'values': [5.0, 10.0, 20.0]},

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
        'target_scaler': {'values': [None]},
        'feature_scaler': {'values': ['MinMaxScaler']},
        'log_targets': {'values': [False]},
        'log_features': {'values': [None]},
        
        # --- Other ---
        'steps': {'values': [[*range(1, 37)]]},
        'mc_dropout': {'values': [True]},
        'force_reset': {'values': [True]},
        'random_state': {'values': [1, 2]},
    }

    sweep_config['parameters'] = parameters
    return sweep_config