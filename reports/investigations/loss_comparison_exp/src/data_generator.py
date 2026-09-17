"""
Generates and saves synthetic time series data for the experiment.
"""

import os
import numpy as np
import pandas as pd

# Import config from the same directory
from .configs import DATA_CONFIG, EXPERIMENT_CONFIG


def generate_clustered_synthetic_data(
    n_samples,
    p_peace_to_conflict,
    p_conflict_to_peace,
    lognormal_mean,
    lognormal_sigma,
    random_seed,
):
    """
    Generates a single synthetic time series with temporally clustered events using a Markov Chain
    and log-normally distributed conflict magnitudes.
    """
    # Use a local random state for reproducibility per country, but allow global randomness if needed
    rng = np.random.default_rng(seed=random_seed)

    states = np.zeros(n_samples, dtype=int)
    raw_counts = np.zeros(n_samples, dtype=np.float32)

    p_peace_to_peace = 1 - p_peace_to_conflict
    p_conflict_to_conflict = 1 - p_conflict_to_peace

    # Initial state (peace)
    states[0] = 0
    for i in range(1, n_samples):
        if states[i - 1] == 0:
            states[i] = rng.choice([0, 1], p=[p_peace_to_peace, p_peace_to_conflict])
        else:
            states[i] = rng.choice(
                [0, 1], p=[p_conflict_to_peace, p_conflict_to_conflict]
            )

    n_conflict_events = np.sum(states)

    # Only draw magnitudes for conflict events
    if n_conflict_events > 0:
        conflict_magnitudes = rng.lognormal(
            mean=lognormal_mean, sigma=lognormal_sigma, size=n_conflict_events
        )
        raw_counts[states == 1] = conflict_magnitudes

    return raw_counts


def generate_multiple_clustered_synthetic_data(
    data_config: dict, experiment_config: dict
):
    """
    Generates multiple synthetic time series (timelines/countries) with varying
    stochastic parameters, then combines them into a single multivariate DataFrame.
    """
    n_samples = data_config["n_samples"]
    n_timelines = data_config["n_timelines"]
    global_random_seed = experiment_config["random_seed"]

    all_raw_data = []

    # Set numpy random seed once for reproducibility of meta-parameters
    np.random.seed(global_random_seed)

    for i in range(n_timelines):
        # Sample country-specific parameters from global distributions
        # Use a local random seed for each timeline, derived from the global seed + iteration
        country_random_seed = global_random_seed + i

        # Beta distribution for transition probabilities (ensures values between 0 and 1)
        p_peace_to_conflict = np.random.beta(
            data_config["P_PC_BETA_PARAMS"]["alpha"],
            data_config["P_PC_BETA_PARAMS"]["beta"],
        )
        p_conflict_to_peace = np.random.beta(
            data_config["P_CP_BETA_PARAMS"]["alpha"],
            data_config["P_CP_BETA_PARAMS"]["beta"],
        )

        # Normal distribution for lognormal_mean (truncated to avoid excessively low values if desired)
        lognormal_mean = np.random.normal(
            data_config["LOGNORMAL_MEAN_NORMAL_PARAMS"]["loc"],
            data_config["LOGNORMAL_MEAN_NORMAL_PARAMS"]["scale"],
        )
        lognormal_mean = max(0.01, lognormal_mean)  # Ensure mean is not too low

        # Log-normal distribution for lognormal_sigma (ensures positive sigma)
        lognormal_sigma = np.random.lognormal(
            data_config["LOGNORMAL_SIGMA_LOGNORMAL_PARAMS"]["mean"],
            data_config["LOGNORMAL_SIGMA_LOGNORMAL_PARAMS"]["sigma"],
        )
        lognormal_sigma = max(0.1, lognormal_sigma)  # Ensure sigma is not too low

        # Generate data for this country
        raw_counts_country = generate_clustered_synthetic_data(
            n_samples=n_samples,
            p_peace_to_conflict=p_peace_to_conflict,
            p_conflict_to_peace=p_conflict_to_peace,
            lognormal_mean=lognormal_mean,
            lognormal_sigma=lognormal_sigma,
            random_seed=country_random_seed,
        )

        # Create DataFrame for this country
        time_index = pd.date_range(start="2000-01-01", periods=n_samples, freq="M")
        country_df = pd.DataFrame(
            {
                "time": time_index,
                "country_id": f"country_{i:03d}",
                "target": raw_counts_country,
            }
        )
        all_raw_data.append(country_df)

    # Concatenate all country data into a single DataFrame
    full_raw_df = pd.concat(all_raw_data, ignore_index=True)

    return full_raw_df


def main():
    """Main function to generate and save the datasets."""
    output_dir = os.path.join(EXPERIMENT_CONFIG["output_dir"], "data")
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"Generating {DATA_CONFIG['n_timelines']} timelines of {DATA_CONFIG['n_samples']} steps each..."
    )
    # Generate data for multiple timelines
    full_raw_df = generate_multiple_clustered_synthetic_data(
        DATA_CONFIG, EXPERIMENT_CONFIG
    )

    # Create log1p transformed DataFrame
    full_log1p_df = full_raw_df.copy()
    full_log1p_df["target"] = np.log1p(full_raw_df["target"])

    # Save to CSV
    raw_path = os.path.join(output_dir, "raw_counts.csv")
    log1p_path = os.path.join(output_dir, "log1p_transformed.csv")

    full_raw_df.to_csv(raw_path, index=False)
    full_log1p_df.to_csv(log1p_path, index=False)

    print(
        f"Successfully saved raw data for {DATA_CONFIG['n_timelines']} timelines to {raw_path}"
    )
    print(
        f"Successfully saved log1p transformed data for {DATA_CONFIG['n_timelines']} timelines to {log1p_path}"
    )


if __name__ == "__main__":
    main()
