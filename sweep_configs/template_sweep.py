"""
Template for a sweep configuration file.

This file defines a set of hyperparameters to be used in a hyperparameter sweep
experiment. Each parameter is defined as a dictionary with a 'values' key
specifying a list of options to explore.

Refer to the experiment's documentation for details on required parameters
and their expected types.
"""

SWEEP_CONFIG = {
    # --- N-BEATS Architecture ---
    'num_blocks': {'values': [2, 4]},
    'num_stacks': {'values': [2, 4]},
    'dropout': {'values': [0.1, 0.3, 0.5]},
    'layer_widths': {'values': [16, 32, 64]},
    'num_layers': {'values': [2, 3]},
    'activation': {'values': ['ReLU', 'LeakyReLU']},
    'generic_architecture': {'values': [True]}, # Usually fixed for a sweep type

    # --- Trainer & Optimizer ---
    'n_epochs': {'values': [100]}, # Max epochs, actual could be less with EarlyStopping
    'lr': {'values': [0.001, 0.0001]},
    'optimizer_cls': {'values': ['Adam']},
    'weight_decay': {'values': [0.0, 0.0001, 0.0003]},
    'gradient_clip_val': {'values': [0.5, 1.0]},
    'lr_scheduler_cls': {'values': ['ReduceLROnPlateau']},
    'lr_scheduler_patience': {'values': [5, 7]},
    'lr_scheduler_factor': {'values': [0.3, 0.5]},
    'lr_scheduler_min_lr': {'values': [1e-6]},
    'early_stopping_patience': {'values': [10, 15, 20]},
    'early_stopping_min_delta': {'values': [0.001, 0.01]},
    
    # --- Data Handling & Input/Output ---
    'input_chunk_length': {'values': [24, 36]},
    'output_chunk_length': {'values': [12, 24, 36]},
    'batch_size': {'values': [8, 16]},
    'log_targets': {'values': [True]}, # Whether to log1p transform targets
    
    # --- Other ---
    'random_state': {'values': [42]},
    'force_reset': {'values': [True]},
    'loss_fn_class': {'values': ['WeightedPenaltyHuberLoss', 'AsymmetricQuantileLoss']},
    'loss_fn_params': {'values': [
        # Example for WeightedPenaltyHuberLoss
        {'zero_threshold': 0.01, 'delta': 0.025, 'non_zero_weight': 5.0, 'false_positive_weight': 1.0, 'false_negative_weight': 10.0},
        # Example for AsymmetricQuantileLoss
        {'tau': 0.98, 'non_zero_weight': 5.0, 'zero_threshold': 0.01}
    ]}
}

# Optional: Add specific model configuration overrides if needed
# MODEL_SPECIFIC_OVERRIDES = {
#     "huber_model": {
#         "loss_fn_params": {
#             "false_negative_weight": 12.0
#         }
#     }
# }
