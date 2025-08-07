# views-r2darts2: Advanced Deep Learning Conflict Forecasting Suite 

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Darts Version](https://img.shields.io/badge/darts-0.35.0%2B-green.svg)](https://unit8co.github.io/darts/)

---

**views-r2darts2** is a time series forecasting package designed for conflict prediction within the VIEWS (Violence and Impacts Early-Warning System) ecosystem. It employs state-of-the-art deep learning models from the Darts library to produce forecasts and conflict-related metrics.

---

## Key Features

- 🚀 **Production-Ready Integration**: Seamlessly integrates with the VIEWS pipeline ecosystem
- � **Zero-Inflated Data Handling**: Specialized scalers and loss functions for conflict data
- 🧠 **Multiple Model Architectures**: Supports 8+ cutting-edge forecasting models
- 📈 **Probabilistic Forecasting**: Quantifies uncertainty through multiple samples
- ⚙️ **Hyperparameter Management**: Centralized configuration for all models
- 🧮 **GPU Acceleration**: Optimized for GPU training and inference

---

## 📦 Installation

```bash
git clone https://github.com/views-platform/views-r2darts2.git
cd views-r2darts2
pip install -e .
```
For GPU support, install the correct PyTorch version for your CUDA setup. See [PyTorch installation guide](https://pytorch.org/get-started/locally/).

---

## 🧠 Supported Models

| Model | Description | Key Strengths | Ideal For |
|-------|-------------|---------------|-----------|
| **TFT** (Temporal Fusion Transformer) | Hybrid architecture combining LSTM encoders, variable selection networks, and multi-head attention. Integrates static, known, and observed covariates for interpretable, context-aware forecasting. | Captures complex temporal and feature relationships, interpretable feature importances, handles missing data and static covariates. | Multivariate forecasting, interpretability, heterogeneous data sources. |
| **N-BEATS** | Deep stack of fully-connected blocks with backward and forward residual links. Each block models trend and seasonality using interpretable basis expansions. | Highly interpretable, flexible for both univariate and multivariate series, excels at trend/seasonality extraction. | Univariate/multivariate forecasting, interpretable decomposition, long-term prediction. |
| **TiDE** | MLP-based architecture that models temporal dependencies using time and feature mixing blocks. Avoids recurrence and attention for efficient training and inference. | Fast training, scalable to large datasets, effective for long input sequences. | Large-scale forecasting, resource-constrained environments, long input horizons. |
| **TCN** (Temporal Convolutional Network) | Deep convolutional network with dilated causal convolutions and residual connections. Captures long-range dependencies without recurrence. | Highly parallelizable, robust to vanishing gradients, effective for high-frequency and long-range temporal patterns. | High-frequency time series, long-range dependencies, parallel training. |
| **Block RNN** | Sequential model using stacked RNN, LSTM, or GRU layers. Supports autoregressive and direct multi-step forecasting. | Good for capturing sequential and temporal dependencies, flexible for various RNN types. | Sequential dependencies, classic time series modeling, autoregressive tasks. |
| **Transformer** | Self-attention-based model with positional encoding, multi-head attention, and feed-forward layers. Models long-term dependencies and complex temporal relationships. | Scalable to large datasets, handles multivariate and long-horizon forecasting, interpretable attention weights. | Multivariate, long-term forecasting, complex temporal patterns. |
| **NLinear** | Simple linear model optionally decomposing input into trend and seasonality components. Uses linear layers for fast, interpretable predictions. | Extremely fast inference, interpretable, suitable for baseline and trend/seasonality separation. | Baseline modeling, trend/seasonality separation, rapid prototyping. |
| **DLinear** | Decomposition-Linear model that splits input into trend and seasonal components, each modeled by separate linear layers. | Efficient for large datasets, interpretable decomposition, improved performance over simple linear models. | Trend/seasonality separation, fast inference, large-scale forecasting. |
| **TSMixer** | MLP-based model that alternates mixing along time and feature dimensions using dedicated mixing blocks. Captures temporal and cross-feature interactions without convolutions or attention. | Lightweight, highly parallelizable, effective for large-scale and multivariate time series, fast training. | Large-scale time series, fast training, scalable architectures, cross-feature interaction modeling. |

---

## ⚡ Loss Functions

- **WeightedHuberLoss**: A custom loss function that combines Huber loss with sample weighting. Non-zero targets are given higher importance, helping the model focus on significant events in zero-inflated data.
- **TimeAwareWeightedHuberLoss**: Extends WeightedHuberLoss by applying temporal decay and event-based weighting, allowing the model to emphasize recent or important events more strongly.
- **SpikeFocalLoss**: Combines mean squared error with a focal mechanism to emphasize errors on "spike" targets, useful for rare but impactful events.
- **WeightedPenaltyHuberLoss**: Further extends Huber loss by applying different penalties for false positives and false negatives, making it ideal for imbalanced datasets where missing or misclassifying events has different costs.

---

## 🛠️ How It Works

1. **Data Preparation**:
   - Handles VIEWS-formatted spatiotemporal data, supporting both CSV and DataFrame inputs.
   - Integrates with the `_ViewsDatasetDarts` class, which converts raw data into Darts-compatible `TimeSeries` objects, manages static and dynamic covariates, and ensures correct alignment of time and entity indices.
   - Supports broadcasting of features and targets, enabling flexible handling of multi-entity and multi-feature datasets.

2. **Model Training**:
   - Models are instantiated via the centralized `ModelCatalog`, which selects and configures the appropriate Darts model based on the provided configuration dictionary.
   - Loss functions are selected and parameterized using the `LossSelector` utility, allowing for advanced weighting and penalty schemes tailored to zero-inflated and imbalanced data.
   - Training leverages PyTorch Lightning under the hood, enabling features such as:
     - Learning rate scheduling (OneCycle, Cosine Annealing, etc.)
     - Early stopping based on monitored metrics (e.g., train loss)
     - Gradient clipping to stabilize training
     - Callbacks for logging, monitoring, and checkpointing
   - Models are trained on GPU (if available), with automatic device selection (`cuda`, `mps`, or `cpu`).

3. **Forecasting**:
   - The `DartsForecaster` class manages the workflow for generating forecasts, including preprocessing, scaling, and device management.
   - Supports probabilistic forecasting by generating multiple samples per prediction (Monte Carlo dropout, ensemble methods).
   - Handles variable-length input sequences and flexible output horizons, allowing for both rolling and fixed-window predictions.
   - Maintains temporal alignment of predictions, ensuring that forecasted values are correctly indexed by time and entity.
   - Predictions are post-processed to ensure non-negativity and proper formatting, with support for inverse scaling.

4. **Evaluation**:
   - Evaluation routines automatically select the appropriate model artifact and partition the data for assessment.
   - Outputs predictions as DataFrames indexed by time and entity, with support for multiple samples and uncertainty quantification.
   - Includes automatic model versioning and artifact management, saving trained models and scalers for reproducibility and future inference.

5. **Model Management & Saving**:
   - The `DartsForecastingModelManager` class orchestrates the full lifecycle of model training, evaluation, and artifact management.
   - Models and associated scalers are saved and loaded together, ensuring consistent preprocessing during inference.
   - Supports partitioned dataset handling, parallel prediction jobs, and Monte Carlo inference for uncertainty estimation.

---

## 🧩 Troubleshooting & FAQ

- **CUDA Errors:** Ensure correct PyTorch and CUDA versions.
- **Data Shape Mismatches:** Check input dimensions and required columns.
- **Scaling Issues:** Verify scalers are fitted before prediction.
- **Getting Help:** Open an issue on GitHub or contact maintainers.

