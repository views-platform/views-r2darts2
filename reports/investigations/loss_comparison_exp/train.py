import os
import argparse
import pandas as pd
from darts import TimeSeries  # Keep TimeSeries import for standalone execution
import sys  # Keep sys import for error handling

from src.configs import (
    EXPERIMENT_CONFIG,
    NBEATS_HPS,
    MODEL_CONFIGS,
    TRAINER_CONFIG,
    OPTIMIZER_CONFIG,
    LR_SCHEDULER_CONFIG,
    EARLY_STOPPING_CONFIG,
)
from src.model_definitions import DartsNBEATS


def main(
    model_name: str,
    train_series: list[TimeSeries],  # Now expects list of TimeSeries
    val_series: list[TimeSeries],  # Now expects list of TimeSeries
    nbeats_hps: dict,
    model_configs: dict,
    trainer_config: dict,
    optimizer_config: dict,
    lr_scheduler_config: dict,
    early_stopping_config: dict,
    experiment_config: dict,
):
    """Main function to run the training script, accepting parameters directly."""
    print(f"--- Starting training for model: {model_name} ---")

    # --- 2. Configuration and Directory Setup ---
    output_dir = experiment_config["output_dir"]
    model_config = model_configs[model_name]

    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # --- 5. Instantiate Model ---
    print("Instantiating DartsNBEATS model...")
    model = DartsNBEATS(
        nbeats_hps=nbeats_hps,
        loss_fn_class=model_config["loss_fn_class"],
        loss_fn_params=model_config["loss_fn_params"],
        trainer_config=trainer_config,
        optimizer_config=optimizer_config,
        lr_scheduler_config=lr_scheduler_config,
        early_stopping_config=early_stopping_config,
    )

    # --- 6. Train Model ---
    print("Training model...")
    model.fit(train_series, val_series)  # Pass lists of TimeSeries
    print("Training complete.")

    # --- 7. Save Model ---
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    model.save(model_path)
    print(f"Model saved successfully to {model_path}")
    print(f"--- Finished training for model: {model_name} ---")


if __name__ == "__main__":
    # This block will handle standalone execution and data loading/splitting
    parser = argparse.ArgumentParser(description="Train a single N-BEATS model.")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        choices=MODEL_CONFIGS.keys(),
        help="The name of the model to train, as defined in configs.py",
    )
    args = parser.parse_args()
    model_name = args.model_name

    # --- Data loading and splitting (for standalone execution) ---
    output_dir = EXPERIMENT_CONFIG["output_dir"]
    data_path = os.path.join(output_dir, "data", "log1p_transformed.csv")
    try:
        data_df = pd.read_csv(data_path, parse_dates=["time"])
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}.")
        print("Please run data_generator.py first.")
        sys.exit(1)  # Use sys.exit here as it's a standalone script

    # Convert DataFrame to list of Darts TimeSeries
    full_series_log1p = TimeSeries.from_dataframe(
        data_df,
        "time",
        "target",
        group_cols="country_id",
        fill_missing_dates=True,
        freq="M",
    )

    # Data Splitting: Apply split to each TimeSeries in the list
    train_series_standalone = []
    val_series_standalone = []
    for ts_log1p in full_series_log1p:
        split_point = int(len(ts_log1p) * (1 - EXPERIMENT_CONFIG["test_split_ratio"]))
        train_series_standalone.append(ts_log1p[:split_point])
        val_series_standalone.append(ts_log1p[split_point:])  # Validation set

    # Call main function with loaded and split data
    main(
        model_name=model_name,
        train_series=train_series_standalone,
        val_series=val_series_standalone,
        nbeats_hps=NBEATS_HPS,
        model_configs=MODEL_CONFIGS,
        trainer_config=TRAINER_CONFIG,
        optimizer_config=OPTIMIZER_CONFIG,
        lr_scheduler_config=LR_SCHEDULER_CONFIG,
        early_stopping_config=EARLY_STOPPING_CONFIG,
        experiment_config=EXPERIMENT_CONFIG,
    )
