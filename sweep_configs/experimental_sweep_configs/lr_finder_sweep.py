# sweep_configs/lr_finder_sweep.py

# Wandb sweep configuration for finding a stable learning rate
# for TweedieLoss and ShrinkageLoss on synthetic data.

sweep_config = {
    "program": "simple_training_run.py",
    "method": "bayes",
    "name": "lr_finder_tweedie_shrinkage",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "run_cap": 24,  # Limit the total number of runs for this sweep
    "parameters": {
        "loss_function": {"values": ["TweedieLoss", "ShrinkageLoss"]},
        "transformation": {"values": ["raw", "log1p", "asinh"]},
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-1,
        },
    },
}
