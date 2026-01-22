import os
import argparse
import pandas as pd
import numpy as np
import json
import sys
from darts import TimeSeries
from sklearn.metrics import mean_squared_log_error

from src.configs import (
    EXPERIMENT_CONFIG,
    NBEATS_HPS,
    MODEL_CONFIGS,
    TRAINER_CONFIG,
    OPTIMIZER_CONFIG,
    LR_SCHEDULER_CONFIG,
    EARLY_STOPPING_CONFIG
)
from src.model_definitions import DartsNBEATS
from src.utils.metrics import calculate_mean_pred_true_ratio

def main(model_name: str,
         val_series_raw: list[TimeSeries],     # New: expects list of TimeSeries for raw data
         val_series_log1p: list[TimeSeries],   # New: expects list of TimeSeries for log1p data
         nbeats_hps: dict,
         model_configs: dict,
         trainer_config: dict,
         optimizer_config: dict,
         lr_scheduler_config: dict,
         early_stopping_config: dict,
         experiment_config: dict):

    """Main function to run the evaluation script, accepting parameters directly."""
    print(f"--- Starting evaluation for model: {model_name} ---")

    # --- 2. Configuration and Directory Setup ---
    output_dir = experiment_config["output_dir"]
    model_config = model_configs[model_name]
    
    models_dir = os.path.join(output_dir, "models")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # --- 5. Load Trained Model ---
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    print(f"Loading model from {model_path}...")
    try:
        model = DartsNBEATS.load(
            path=model_path,
            nbeats_hps=nbeats_hps,
            loss_fn_class=model_config["loss_fn_class"],
            loss_fn_params=model_config["loss_fn_params"],
            trainer_config=trainer_config,
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
            early_stopping_config=early_stopping_config
        )
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        print(f"Please run train.py for model '{model_name}' first.")
        return
    print("Model loaded successfully.")

    # --- 6. Generate Predictions ---
    print(f"Generating predictions for {len(val_series_log1p)} series...")
    # Get number of steps to predict from the first series (assuming all same length)
    n_predict = len(val_series_log1p[0]) 
    predictions_log1p_list = model.predict(n=n_predict, series=val_series_log1p) # Pass list of series

    # --- 7. Inverse Transform and Align Data ---
    all_y_true = []
    all_y_pred = []

    for i in range(len(val_series_log1p)):
        predictions_raw = predictions_log1p_list[i].map(np.expm1)
        
        y_true_country = val_series_raw[i].values().flatten()
        y_pred_country = predictions_raw.values().flatten()

        # Ensure predictions are non-negative
        y_pred_country[y_pred_country < 0] = 0
        # Clip predictions to avoid overflow errors in metrics for very large values
        y_pred_country = np.clip(y_pred_country, a_min=None, a_max=1e10)
        
        all_y_true.append(y_true_country)
        all_y_pred.append(y_pred_country)

    # Aggregate all predictions and true values across countries
    aggregated_y_true = np.concatenate(all_y_true)
    aggregated_y_pred = np.concatenate(all_y_pred)

    # --- 8. Calculate Metrics ---
    print("Calculating metrics...")
    msle = mean_squared_log_error(aggregated_y_true, aggregated_y_pred)
    ratio = calculate_mean_pred_true_ratio(aggregated_y_true, aggregated_y_pred)
    
    metrics = {
        "msle": msle,
        "mean_pred_true_ratio": ratio,
    }
    print(f"Metrics: {metrics}")

    # --- 9. Save Results ---
    result_path = os.path.join(results_dir, f"{model_name}_results.json")
    with open(result_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Results saved successfully to {result_path}")
    print(f"--- Finished evaluation for model: {model_name} ---")

if __name__ == "__main__":
    # This block will handle standalone execution and data loading/splitting
    parser = argparse.ArgumentParser(description="Evaluate a single N-BEATS model.")
    parser.add_argument("--model-name", type=str, required=True,
                        choices=MODEL_CONFIGS.keys(),
                        help="The name of the model to evaluate, as defined in configs.py")
    args = parser.parse_args()
    model_name = args.model_name

    # --- Data loading and splitting (for standalone execution) ---
    output_dir = EXPERIMENT_CONFIG["output_dir"]
    log1p_path = os.path.join(output_dir, "data", "log1p_transformed.csv")
    raw_path = os.path.join(output_dir, "data", "raw_counts.csv")
    try:
        log1p_df = pd.read_csv(log1p_path, parse_dates=['time'])
        raw_df = pd.read_csv(raw_path, parse_dates=['time'])
    except FileNotFoundError:
        print(f"Error: Data files not found in {output_dir}/data.")
        print("Please run data_generator.py first.")
        sys.exit(1)

    # Convert DataFrame to list of Darts TimeSeries
    full_series_log1p = TimeSeries.from_dataframe(log1p_df, 'time', 'target', group_cols='country_id', fill_missing_dates=True, freq='M')
    full_series_raw = TimeSeries.from_dataframe(raw_df, 'time', 'target', group_cols='country_id', fill_missing_dates=True, freq='M')

    # Data Splitting: Apply split to each TimeSeries in the list
    val_series_raw_standalone = []
    val_series_log1p_standalone = []
    for ts_log1p, ts_raw in zip(full_series_log1p, full_series_raw):
        split_point = int(len(ts_log1p) * (1 - EXPERIMENT_CONFIG["test_split_ratio"]))
        val_series_log1p_standalone.append(ts_log1p[split_point:])
        val_series_raw_standalone.append(ts_raw[split_point:])

    # Call main function with loaded and split data
    main(model_name=model_name,
         val_series_raw=val_series_raw_standalone,
         val_series_log1p=val_series_log1p_standalone,
         nbeats_hps=NBEATS_HPS,
         model_configs=MODEL_CONFIGS,
         trainer_config=TRAINER_CONFIG,
         optimizer_config=OPTIMIZER_CONFIG,
         lr_scheduler_config=LR_SCHEDULER_CONFIG,
         early_stopping_config=EARLY_STOPPING_CONFIG,
         experiment_config=EXPERIMENT_CONFIG)
