"""
Main orchestration script for the loss function comparison experiment.
This script sets up the experiment, runs data generation, trains models,
evaluates them, and generates a final report.
"""

import os
import json
import pandas as pd
import numpy as np
from darts import TimeSeries
from sklearn.metrics import mean_squared_log_error

# Import modules from the src directory
from src.configs import (
    EXPERIMENT_CONFIG,
    NBEATS_HPS,  # Now imported here for direct passing
    MODEL_CONFIGS,
    TRAINER_CONFIG,
    OPTIMIZER_CONFIG,
    LR_SCHEDULER_CONFIG,
    EARLY_STOPPING_CONFIG,  # Keep this here for direct use
)
from src.data_generator import main as data_generator_main
from src.model_definitions import HeuristicBaseline

# Import top-level scripts as functions
import train
import evaluate
from src.utils.metrics import calculate_mean_pred_true_ratio


def main():
    """Orchestrates the entire experiment pipeline."""
    output_base_dir = EXPERIMENT_CONFIG["output_dir"]
    data_dir = os.path.join(output_base_dir, "data")
    models_dir = os.path.join(output_base_dir, "models")
    results_dir = os.path.join(output_base_dir, "results")

    # --- 1. Setup: Create directories ---
    print("\n--- 1. Setting up directories ---")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # --- 2. Data Generation ---
    print("\n--- 2. Generating synthetic data ---")
    data_generator_main()

    # --- Load Data for Baseline and Splits ---
    log1p_path = os.path.join(data_dir, "log1p_transformed.csv")
    raw_path = os.path.join(data_dir, "raw_counts.csv")

    log1p_df = pd.read_csv(log1p_path, parse_dates=["time"])
    raw_df = pd.read_csv(raw_path, parse_dates=["time"])

    # Convert DataFrames to list of Darts TimeSeries
    # Each TimeSeries in the list represents a single country (timeline)
    full_series_log1p = TimeSeries.from_group_dataframe(
        df=log1p_df,
        time_col="time",
        value_cols="target",
        group_cols="country_id",
        fill_missing_dates=True,
        freq="M",
    )
    full_series_raw = TimeSeries.from_group_dataframe(
        df=raw_df,
        time_col="time",
        value_cols="target",
        group_cols="country_id",
        fill_missing_dates=True,
        freq="M",
    )

    # Data Splitting: Apply split to each TimeSeries in the list
    train_series_log1p = []
    val_series_raw = []
    val_series_log1p = []  # New: Validation series for log1p
    for i in range(len(full_series_log1p)):
        ts_log1p = full_series_log1p[i]
        ts_raw = full_series_raw[i]

        split_point = int(len(ts_log1p) * (1 - EXPERIMENT_CONFIG["test_split_ratio"]))
        train_series_log1p.append(ts_log1p[:split_point])
        val_series_log1p.append(ts_log1p[split_point:])  # For model evaluation
        val_series_raw.append(ts_raw[split_point:])  # For baseline and model evaluation

    print(
        f"Loaded {len(train_series_log1p)} training series and {len(val_series_raw)} validation series."
    )

    # --- 3. Baseline Evaluation ---
    print("\n--- 3. Evaluating Heuristic Baseline ---")
    all_baseline_y_true = []
    all_baseline_predictions_raw = []

    for i in range(len(train_series_log1p)):
        train_ts_country = train_series_log1p[i]
        val_ts_country_raw = val_series_raw[i]

        baseline_country = HeuristicBaseline()  # No window_size parameter
        baseline_country.fit(train_ts_country)

        n_predict_baseline_country = len(val_ts_country_raw)
        baseline_predictions_country_raw = (
            baseline_country.predict(n=n_predict_baseline_country).values().flatten()
        )
        baseline_y_true_country = val_ts_country_raw.values().flatten()

        baseline_predictions_country_raw[baseline_predictions_country_raw < 0] = 0

        all_baseline_y_true.append(baseline_y_true_country)
        all_baseline_predictions_raw.append(baseline_predictions_country_raw)

    aggregated_baseline_y_true = np.concatenate(all_baseline_y_true)
    aggregated_baseline_predictions_raw = np.concatenate(all_baseline_predictions_raw)

    baseline_msle = mean_squared_log_error(
        aggregated_baseline_y_true, aggregated_baseline_predictions_raw
    )
    baseline_ratio = calculate_mean_pred_true_ratio(
        aggregated_baseline_y_true, aggregated_baseline_predictions_raw
    )

    baseline_metrics = {
        "msle": baseline_msle,
        "mean_pred_true_ratio": baseline_ratio,
    }
    baseline_result_path = os.path.join(results_dir, "baseline_results.json")
    with open(baseline_result_path, "w") as f:
        json.dump(baseline_metrics, f, indent=4)
    print(f"Baseline metrics saved to {baseline_result_path}")

    # --- 4. Training Models ---
    print("\n--- 4. Training N-BEATS models ---")
    for model_name in MODEL_CONFIGS.keys():
        print(f"--- Training {model_name} ---")
        train.main(
            model_name=model_name,
            train_series=train_series_log1p,  # Pass list of TimeSeries
            val_series=val_series_log1p,  # Pass list of TimeSeries
            nbeats_hps=NBEATS_HPS,
            model_configs=MODEL_CONFIGS,
            trainer_config=TRAINER_CONFIG,
            optimizer_config=OPTIMIZER_CONFIG,
            lr_scheduler_config=LR_SCHEDULER_CONFIG,
            early_stopping_config=EARLY_STOPPING_CONFIG,
            experiment_config=EXPERIMENT_CONFIG,
        )

    # --- 5. Evaluation Models ---
    print("\n--- 5. Evaluating N-BEATS models ---")
    for model_name in MODEL_CONFIGS.keys():
        print(f"--- Evaluating {model_name} ---")
        evaluate.main(
            model_name=model_name,
            val_series_raw=val_series_raw,  # Pass list of TimeSeries
            val_series_log1p=val_series_log1p,  # Pass list of TimeSeries
            nbeats_hps=NBEATS_HPS,
            model_configs=MODEL_CONFIGS,
            trainer_config=TRAINER_CONFIG,
            optimizer_config=OPTIMIZER_CONFIG,
            lr_scheduler_config=LR_SCHEDULER_CONFIG,
            early_stopping_config=EARLY_STOPPING_CONFIG,
            experiment_config=EXPERIMENT_CONFIG,
        )

    # --- 6. Reporting ---
    print("\n--- 6. Generating Final Report ---")
    all_results = {}

    # Load baseline results
    with open(baseline_result_path, "r") as f:
        all_results["Baseline"] = json.load(f)

    # Load N-BEATS model results
    for model_name in MODEL_CONFIGS.keys():
        result_path = os.path.join(results_dir, f"{model_name}_results.json")
        with open(result_path, "r") as f:
            all_results[model_name] = json.load(f)

    print("\n--- Experiment Results Summary ---")
    report_df = pd.DataFrame.from_dict(all_results, orient="index")
    print(report_df.to_markdown())
    print("----------------------------------")


if __name__ == "__main__":
    main()
