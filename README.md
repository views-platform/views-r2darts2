# views-r2darts2: Advanced Deep Learning Conflict Forecasting Suite 

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Darts Version](https://img.shields.io/badge/darts-0.35.0%2B-green.svg)](https://unit8co.github.io/darts/)

**views-r2darts2** is a time series forecasting package designed for conflict prediction within the VIEWS (Violence and Impacts Early-Warning System) ecosystem. It leverages state-of-the-art deep learning models from the Darts library to produce forecasts and conflict-related metrics.

## Key Features

- 🚀 **Production-Ready Integration**: Seamlessly integrates with the VIEWS pipeline ecosystem
- � **Zero-Inflated Data Handling**: Specialized scalers and loss functions for conflict data
- 🧠 **Multiple Model Architectures**: Supports 8+ cutting-edge forecasting models
- 📈 **Probabilistic Forecasting**: Quantifies uncertainty through multiple samples
- ⚙️ **Hyperparameter Management**: Centralized configuration for all models
- 🧮 **GPU Acceleration**: Optimized for GPU training and inference

## Supported Models

| Model | Key Strengths | Ideal For |
|-------|---------------|-----------|
| **TFT** (Temporal Fusion Transformer) | Attention mechanisms, feature importance | Complex patterns, covariate relationships |
| **N-BEATS** | Interpretable, stackable blocks | Univariate forecasting, model interpretability |
| **TiDE** | Efficient temporal modeling | Large datasets, resource-constrained environments |
| **TCN** (Temporal Convolutional Network) | Parallel processing, long-range dependencies | High-frequency data, efficiency |
| **Block RNN** | Sequential pattern recognition | Sequential dependencies, classic RNN/LSTM/GRU |
| **Transformer** | Long-term dependencies | Complex temporal relationships |
| **NLinear** | Simple linear architecture | Baseline modeling, fast inference |
| **DLinear** | Decomposition + linear | Trend/seasonality separation |

## How It Works

### Core Components

1. **Data Preparation**:
   - Handles VIEWS-formatted spatiotemporal data
   - Applies specialized scalers for zero-inflated conflict data
   - Fully integrated with `_ViewsDataset`

2. **Model Training**:
   - Configures models through centralized `ModelCatalog`
   - Uses weighted loss functions (`WeightedHuberLoss`) to handle data imbalance
   - Implements advanced training techniques:
     - Learning rate scheduling (OneCycle, Cosine Annealing)
     - Early stopping
     - Gradient clipping

3. **Forecasting**:
   - Generates probabilistic forecasts with uncertainty quantification
   - Handles variable-length input sequences
   - Maintains temporal alignment of predictions

4. **Evaluation**:
   - Supports multiple evaluation types (standard, long-term, live)
   - Automatic model versioning and artifact management
