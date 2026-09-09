"""Script to verify the refined HeuristicBaseline implementation.
Generates data, applies the baseline, plots predictions against actuals for a subset of timelines,
and provides performance statistics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from darts import TimeSeries
from sklearn.metrics import mean_squared_log_error

from src.configs import EXPERIMENT_CONFIG
from src.data_generator import (
    main as data_generator_main,
)  # Import the data generator's main function
from src.model_definitions import (
    HeuristicBaseline,
)  # Import the updated HeuristicBaseline
from src.utils.metrics import (
    calculate_mean_pred_true_ratio,
)  # Import the metric function


def main():
    """Orchestrates the baseline verification process."""
    output_base_dir = EXPERIMENT_CONFIG["output_dir"]
    data_dir = os.path.join(output_base_dir, "data")
    plots_dir = os.path.join(data_dir, "plots")
    stats_dir = os.path.join(data_dir, "stats")

    # --- 1. Setup: Create directories ---
    print("\n--- 1. Setting up directories ---")
    os.makedirs(plots_dir, exist_ok=True)  # Ensure plots_dir exists
    os.makedirs(stats_dir, exist_ok=True)  # Ensure stats_dir exists

    # --- 2. Generate Data ---
    print("\n--- 2. Generating synthetic data via src.data_generator.main() ---")
    data_generator_main()  # This ensures raw_counts.csv and log1p_transformed.csv are up-to-date

    # --- 3. Load Generated Data ---
    print("\n--- 3. Loading generated raw data for analysis ---")
    raw_path = os.path.join(data_dir, "raw_counts.csv")
    log1p_path = os.path.join(
        data_dir, "log1p_transformed.csv"
    )  # Need log1p for N-BEATS, but raw for baseline fit.
    try:
        raw_df = pd.read_csv(raw_path, parse_dates=["time"])
        log1p_df = pd.read_csv(log1p_path, parse_dates=["time"])
    except FileNotFoundError:
        print("Error: Data files not found.")
        print("Ensure data_generator.py has run successfully.")
        return

    # Convert DataFrames to list of Darts TimeSeries
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

    # --- 4. Data Splitting ---
    train_series_log1p = []
    val_series_raw = []
    # No need for val_series_log1p or train_series_raw in this script's main logic
    # but train_series_log1p will be passed to baseline.fit as the only argument
    # val_series_raw is needed for actual values in verification

    for i in range(len(full_series_log1p)):
        ts_log1p = full_series_log1p[i]
        ts_raw = full_series_raw[i]

        split_point = int(len(ts_log1p) * (1 - EXPERIMENT_CONFIG["test_split_ratio"]))
        train_series_log1p.append(ts_log1p[:split_point])
        val_series_raw.append(ts_raw[split_point:])

    print(
        f"Loaded {len(train_series_log1p)} training series and {len(val_series_raw)} validation series."
    )

    # --- 5. Select Random Subset of Timelines for Verification ---
    print("\n--- 5. Selecting random subset of timelines for verification ---")
    num_timelines_to_plot = 10
    random.seed(
        EXPERIMENT_CONFIG["random_seed"]
    )  # For reproducibility of random selection

    selected_indices = random.sample(
        range(len(full_series_log1p)),
        min(num_timelines_to_plot, len(full_series_log1p)),
    )

    selected_train_series = [train_series_log1p[i] for i in selected_indices]
    selected_val_series_raw = [val_series_raw[i] for i in selected_indices]

    # --- 6. Apply Heuristic Baseline & Gather Predictions for Selected Timelines ---
    print("\n--- 6. Applying Heuristic Baseline and gathering predictions ---")
    all_plot_y_true = []
    all_plot_predictions_raw = []
    per_country_metrics = []

    fig, axes = plt.subplots(
        len(selected_indices), 1, figsize=(15, 4 * len(selected_indices)), sharex=True
    )
    if len(selected_indices) == 1:
        axes = [axes]  # Ensure axes is iterable even for a single plot

    for idx, (train_ts_country, val_ts_country_raw) in enumerate(
        zip(selected_train_series, selected_val_series_raw)
    ):
        country_id = full_series_raw[selected_indices[idx]].components[
            0
        ]  # Get original country ID

        baseline_country = HeuristicBaseline()  # Instantiate new baseline
        baseline_country.fit(train_ts_country)  # Fit on country's training data

        n_predict_baseline_country = len(val_ts_country_raw)
        baseline_predictions_country_raw = baseline_country.predict(
            n=n_predict_baseline_country
        )  # Get TimeSeries
        # Predictions are now directly in raw scale from HeuristicBaseline.predict()

        y_true_flat = val_ts_country_raw.values().flatten()
        y_pred_flat = (
            baseline_predictions_country_raw.values().flatten()
        )  # Flatten directly for metrics

        # Calculate metrics for this country
        msle = mean_squared_log_error(y_true_flat, y_pred_flat)
        ratio = calculate_mean_pred_true_ratio(y_true_flat, y_pred_flat)

        per_country_metrics.append(
            {
                "country_id": country_id,
                "mean_of_training_set": baseline_country.prediction_value,
                "msle": msle,
                "mean_pred_true_ratio": ratio,
            }
        )

        all_plot_y_true.append(y_true_flat)
        all_plot_predictions_raw.append(y_pred_flat)

        # Plotting for this country
        ax = axes[idx]

        # Plot training data (log1p scale for context)
        train_ts_country.plot(
            ax=ax, label="Training Data (Log1p)", color="blue", alpha=0.7
        )

        # Plot validation data (raw scale)
        val_ts_country_raw.plot(
            ax=ax, label="Validation Data (Raw)", color="green", alpha=0.7
        )

        # Plot baseline predictions (raw scale), ensuring it's a flat line over the validation period
        # The baseline prediction is constant, so it should be a horizontal line in the validation period.
        baseline_predictions_country_raw.plot(
            ax=ax, label="Baseline Prediction", color="red", linestyle="--", linewidth=2
        )

        # Add a vertical line for the split point
        split_date = train_ts_country.end_time()
        ax.axvline(
            split_date,
            color="grey",
            linestyle=":",
            alpha=0.8,
            label="Train/Validation Split",
        )

        ax.set_title(f"Country: {country_id} | MSLE: {msle:.2f} | Ratio: {ratio:.2f}")
        ax.legend()  # Legend per subplot for clarity
        ax.set_ylabel("Conflict Count")
        ax.grid(True)

    plt.xlabel("Time")
    plt.tight_layout()
    plots_output_path = os.path.join(plots_dir, "baseline_verification_plots.png")
    plt.savefig(plots_output_path)
    print(f"\nBaseline verification plots saved to {plots_output_path}")

    # --- 7. Report Statistics ---
    print("\n--- 7. Baseline Performance per Selected Country ---")
    per_country_metrics_df = pd.DataFrame(per_country_metrics)
    print(per_country_metrics_df.to_markdown(index=False))

    overall_msle = mean_squared_log_error(
        np.concatenate(all_plot_y_true), np.concatenate(all_plot_predictions_raw)
    )
    overall_ratio = calculate_mean_pred_true_ratio(
        np.concatenate(all_plot_y_true), np.concatenate(all_plot_predictions_raw)
    )
    print("\n--- Overall Baseline Performance (Selected Subset) ---")
    print(f"Overall MSLE: {overall_msle:.4f}")
    print(f"Overall Mean Pred/True Ratio: {overall_ratio:.4f}")

    print("\n--- Baseline Verification Complete ---")


if __name__ == "__main__":
    main()
