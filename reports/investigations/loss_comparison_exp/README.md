# Plan: Modular Experiment for Loss Function Comparison

This document outlines the structure and plan for a modular, verifiable experiment to compare the performance of different loss functions for conflict forecasting on synthetic data.

## 1. Core Objective

To create a controlled environment to rigorously test if `WeightedPenaltyHuberLoss` outperforms `AsymmetricQuantileLoss` on synthetic, zero-inflated, heavy-tailed, and temporally clustered time series data. The primary evaluation criteria are Mean Squared Log Error (MSLE) and the Mean Prediction/True Value Ratio.

## 2. Directory Structure (Best Practice)

The experiment is contained within the `loss_comparison_exp/` directory with the following state-of-the-art structure:

```
loss_comparison_exp/
├── data/                  # Output directory for generated datasets (raw_counts.csv, log1p_transformed.csv)
├── models/                # Output directory for trained model artifacts (huber_model.pkl, quantile_model.pkl)
├── results/               # Output directory for evaluation metric outputs (JSONs)
├── src/                   # Source code for the experiment's Python modules
│   ├── __init__.py        # Makes 'src' a package
│   ├── configs.py         # Centralized configuration for the entire experiment
│   ├── data_generator.py  # Logic to generate and save the synthetic dataset
│   ├── model_definitions.py # Python module defining model wrappers and the baseline
│   └── utils/             # General utilities for the experiment
│       ├── __init__.py    # Makes 'utils' a package
│       └── metrics.py     # Contains utility functions like `calculate_mean_pred_true_ratio`
├── main.py                # Main orchestrator script to run the full pipeline (formerly run_experiment.py)
├── train.py               # Top-level script to execute a single training run
└── evaluate.py            # Top-level script to execute a single evaluation run
```

## 3. Component Breakdown

### Top-Level Scripts:

*   **`main.py`**
    *   **Purpose:** The primary entry point and orchestrator for the entire experiment pipeline.
    *   **Process:** Sets up directories, calls `src.data_generator.main()`, evaluates the `HeuristicBaseline`, iterates through `MODEL_CONFIGS` to call `train.main()` and `evaluate.main()` for each model, and generates a final consolidated report.

*   **`train.py`**
    *   **Purpose:** To perform a single, isolated model training run.
    *   **Process:** Takes a `--model-name` argument, loads configuration and data from `src/`, instantiates and trains the model, and saves the trained model artifact to `models/{model_name}.pkl`.

*   **`evaluate.py`**
    *   **Purpose:** To perform a single, isolated model evaluation run.
    *   **Process:** Takes a `--model-name` argument, loads the trained model and data, generates predictions, calculates metrics (MSLE and Mean Pred/True Ratio) on the raw-scale data, and saves the metrics to a JSON file in the `results/` directory.
    *   **Note on `mean_pred_true_ratio`:** A value of 0 for this metric, particularly for the `HeuristicBaseline`, can occur if the baseline consistently predicts zero for all validation points where the true value is non-zero. This highlights the baseline's inability to forecast certain events and provides context for model comparison.

### `src/` Modules:

*   **`src/configs.py`**
    *   **Purpose:** A single source of truth for all experimental parameters.
    *   **Contents:** Dictionaries for data generation (`DATA_CONFIG`), model architecture (`NBEATS_HPS`), and specific loss function configurations for each experimental run (`MODEL_CONFIGS`).

*   **`src/data_generator.py`**
    *   **Purpose:** Contains the logic to generate a reproducible synthetic dataset based on the `src.configs` file.
    *   **Process:** Uses a 2-state Markov Chain to create temporal clustering and a log-normal distribution for heavy tails. Saves two datasets to the `data/` output directory.

*   **`src/model_definitions.py`**
    *   **Purpose:** To define the structure of all models used in the experiment for consistency.
    *   **Contents:** `HeuristicBaseline` (rolling average) and `DartsNBEATSModel` (a wrapper for Darts N-BEATS models).

*   **`src/utils/metrics.py`**
    *   **Purpose:** Contains general utility functions related to metric calculation.
    *   **Contents:** `calculate_mean_pred_true_ratio` function.