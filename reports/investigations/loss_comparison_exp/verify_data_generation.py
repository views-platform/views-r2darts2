"""
Script to verify the data generation process by generating data,
calculating light statistics, and plotting a random subset of timelines.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from src.configs import EXPERIMENT_CONFIG
from src.data_generator import (
    main as data_generator_main,
)  # Import the data generator's main function


def main():
    """Orchestrates the data generation verification process."""
    output_base_dir = EXPERIMENT_CONFIG["output_dir"]
    data_dir = os.path.join(output_base_dir, "data")
    plots_dir = os.path.join(data_dir, "plots")
    stats_dir = os.path.join(data_dir, "stats")

    # --- 1. Setup: Create directories ---
    print("\n--- 1. Setting up directories ---")
    os.makedirs(
        data_dir, exist_ok=True
    )  # Ensure data_dir exists (created by data_generator but for robustness)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    # --- 2. Generate Data ---
    print("\n--- 2. Generating synthetic data via src.data_generator.main() ---")
    data_generator_main()

    # --- 3. Load Generated Data ---
    print("\n--- 3. Loading generated raw data for analysis ---")
    raw_path = os.path.join(data_dir, "raw_counts.csv")
    try:
        raw_df = pd.read_csv(raw_path, parse_dates=["time"])
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_path}.")
        print("Ensure data_generator.py has run successfully.")
        return

    # --- 4. Perform Light Statistics ---
    print("\n--- 4. Calculating light statistics per timeline ---")
    stats_list = []

    for country_id, group_df in raw_df.groupby("country_id"):
        target_values = group_df["target"].values
        num_zeros = np.sum(target_values == 0)

        # Calculate mean of non-zero values
        non_zero_values = target_values[target_values > 0]
        mean_non_zero = np.mean(non_zero_values) if len(non_zero_values) > 0 else 0.0

        stats_list.append(
            {
                "country_id": country_id,
                "mean_overall": np.mean(target_values),
                "std_overall": np.std(target_values),
                "min_overall": np.min(target_values),
                "max_overall": np.max(target_values),
                "num_zeros": num_zeros,
                "prop_zeros": num_zeros / len(target_values),
                "mean_non_zero": mean_non_zero,
            }
        )

    timeline_stats_df = pd.DataFrame(stats_list)

    print("\n--- Per-Timeline Statistics Summary (first 10) ---")
    print(timeline_stats_df.head(10).to_markdown(index=False))

    print("\n--- Aggregated Statistics Across All Timelines ---")
    print(timeline_stats_df.describe().to_markdown())

    # Save aggregated statistics
    stats_output_path = os.path.join(stats_dir, "timeline_statistics.csv")
    timeline_stats_df.to_csv(stats_output_path, index=False)
    print(f"\nDetailed timeline statistics saved to {stats_output_path}")

    # --- 5. Generate Random Plots ---
    print("\n--- 5. Generating random plots of timelines ---")
    unique_countries = raw_df["country_id"].unique()
    num_plots = min(10, len(unique_countries))  # Plot up to 10 random timelines

    # Ensure reproducibility for random selection
    random.seed(EXPERIMENT_CONFIG["random_seed"])
    random_countries = random.sample(list(unique_countries), num_plots)

    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), sharex=True)
    if num_plots == 1:  # Handle case of single plot
        axes = [axes]
    else:
        axes = axes.flatten()  # Flatten in case num_plots > 1 for consistent indexing

    for i, country_id in enumerate(random_countries):
        country_df = raw_df[raw_df["country_id"] == country_id]
        ax = axes[i]
        ax.plot(country_df["time"], country_df["target"])

        # Add a vertical line for the split point
        split_point_idx = int(
            len(country_df) * (1 - EXPERIMENT_CONFIG["test_split_ratio"])
        )
        split_date = country_df["time"].iloc[split_point_idx]
        ax.axvline(
            split_date,
            color="r",
            linestyle="--",
            alpha=0.7,
            label="Train/Validation Split",
        )

        prop_zeros = timeline_stats_df[timeline_stats_df["country_id"] == country_id][
            "prop_zeros"
        ].iloc[0]
        ax.set_title(f"Timeline: {country_id} (Prop. Zeros: {prop_zeros:.2f})")
        ax.set_ylabel("Conflict Count")
        ax.grid(True)
        if i == 0:  # Add legend only once
            ax.legend()

    plt.xlabel("Time")
    plt.tight_layout()
    plots_output_path = os.path.join(plots_dir, "random_timeline_plots.png")
    plt.savefig(plots_output_path)
    print(f"Random timeline plots saved to {plots_output_path}")

    print("\n--- Data Generator Verification Complete ---")


if __name__ == "__main__":
    main()
