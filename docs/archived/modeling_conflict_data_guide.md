# Guide: Modeling for Zero-Inflated, Heavy-Tailed Conflict Data

**Version:** 1.0
**Purpose:** This guide outlines the core structural properties of conflict event data and, most importantly, the practical implications for data science and machine learning workflows. It is intended to steer practitioners away from naive defaults toward robust and defensible modeling choices.

---

### 1. The Challenge of Zero-Inflation and Rarity

*   **Description:** The vast majority of location-time observations record zero conflict events. Non-zero events are rare "minority cases."
*   **Implications for Modeling:**
    *   **Architecture:** Standard regression models are ill-suited. The process should be modeled in two parts:
        1.  A **classification** model to estimate the probability of an event occurring at all (`P(y > 0)`).
        2.  A **regression** model to estimate the magnitude of the event, conditional on it occurring (`E[y | y > 0]`).
        These are often combined in "Hurdle" or "Two-Part" models. Alternatively, use distributions that explicitly model zero-inflation (e.g., Zero-Inflated Poisson, Zero-Inflated Negative Binomial).
    *   **Evaluation:** Standard classification accuracy is a dangerously misleading metric, as a model can achieve >99% accuracy by trivially predicting zero everywhere. Focus on metrics that are sensitive to the minority case, such as **Precision, Recall, F1-Score, and the Precision-Recall Curve (PRC)**.

### 2. The Challenge of Heavy-Tails and Extreme Events

*   **Description:** The distribution of conflict severity (e.g., fatalities) is highly skewed. A tiny fraction of "extreme" events accounts for a majority of total harm, rendering statistical moments like mean and variance unstable and uninformative.
*   **Implications for Modeling:**
    *   **Data Transformation:** Raw target values are unsuitable for most models.
        *   Reject `StandardScaler`, which is distorted by extremes.
        *   Prefer transformations that compress the tail and handle zeros gracefully, such as the **inverse hyperbolic sine (`asinh(x)`)**, over the less robust `log(x+1)`.
        *   For feature scaling, use `RobustScaler` (which uses medians and quartiles) to prevent extreme outliers in one feature from dominating the dataset.
    *   **Loss Function:** Mean Squared Error (MSE) is the wrong choice. It is pathologically sensitive to outliers and will cause the model to fixate on predicting a few extreme events at the expense of performing well on all others.
        *   Prefer **Mean Absolute Error (MAE / L1 Loss)** or **Huber Loss**, which are less sensitive to extreme errors.
        *   For a probabilistic approach, use **Quantile Loss** to explicitly model different parts of the predictive distribution (e.g., the 90th percentile) rather than just the central tendency.
    *   **Evaluation:** Do not use Root Mean Squared Error (RMSE). Use **MAE** or **Mean Squared Log Error (MSLE)** for more stable and interpretable error measurement.

### 3. The Challenge of Asymmetric Costs

*   **Description:** In humanitarian and security contexts, the cost of different errors is not equal. Failing to predict a conflict that occurs (a **False Negative**) is vastly more costly than predicting a conflict that does not (a **False Positive**).
*   **Implications for Modeling:**
    *   **Loss Function:** This is a hard constraint on the objective function. A standard, symmetric loss is guaranteed to produce an operationally useless model.
        *   The loss function **must be custom and asymmetric**.
        *   Implement a **weighted loss** that applies a significantly higher penalty multiplier to false negatives. The exact weight is a key parameter that encodes strategic priorities. For example, a `Weighted Penalty Huber Loss` that heavily penalizes missed events is a valid approach.
    *   **Evaluation:** Beyond metrics, the final evaluation must involve plotting the trade-off between false negatives and false positives (e.g., on a PRC) and selecting a threshold that aligns with the organization's risk tolerance.

### 4. The Challenge of Latent, Filtered Processes

*   **Description:** Observed event data is not ground truth. It is an incomplete, filtered signal of a deeper, unobservable latent risk process. The signal is filtered by **exposure** (a population must be present for violence to occur) and **observability** (an event must be recorded by an outside source to be in the data).
*   **Implications for Modeling:**
    *   **Conceptual Framing:** The modeling goal is not to predict the next raw count perfectly. The goal is to **estimate the underlying risk**. Model outputs should be interpreted as risk scores or probabilities, not as literal forecasts of fatality counts.
    *   **Feature Engineering:** This implies that features related to observability are critical. Include variables that proxy for observability, such as **press freedom indices, distance to cities, number of active news sources, and cell phone coverage.** This helps the model learn to distinguish between a true zero (low risk) and a zero caused by a lack of reporting.

### 5. The Challenge of Spatiotemporal Dependencies

*   **Description:** Events are not independent. Conflict in one location is correlated with conflict in neighboring locations (spatial dependency) and in the recent past (temporal dependency).
*   **Implications for Modeling:**
    *   **Architecture:** Reject simple tabular models (MLPs) that assume feature independence.
        *   To handle **temporal dependencies**, use sequence models like **Recurrent Neural Networks (LSTMs, GRUs)** or **Temporal Convolutional Networks (TCNs)**.
        *   To handle **spatial dependencies**, use **Graph Neural Networks (GNNs)** where locations are nodes and adjacency defines the edges, or **Spatiotemporal CNNs** that treat the data as a "video" of evolving risk.
    *   **Feature Engineering:** Explicitly create lagged and spatially-lagged features. For example, for a given location, include features like "average fatalities in neighboring cells last month."

### 6. The Challenge of Non-Stationarity

*   **Description:** The underlying drivers of conflict change over time. Relationships that held in the past may not hold in the future due to regime changes, new technologies, or other structural breaks.
*   **Implications for Modeling:**
    *   **Training Strategy:** Do not assume all historical data is equally relevant. The model must be able to adapt to new patterns.
        *   Consider **shorter look-back windows** for training or place a higher weight on more recent data.
        *   Implement a **continuous monitoring and retraining pipeline (MLOps)** to detect performance degradation ("drift") and automatically retrain the model on new data.
    *   **Architecture:** Prefer architectures that can explicitly model regime changes, such as **Temporal Fusion Transformers (TFTs)** or models with attention mechanisms that can learn to change focus over time.

### 7. The Challenge of Imprecise Ground Truth

*   **Description:** Fatality counts are often noisy estimates, and event locations/dates can be off by several kilometers or days. The data is fuzzy.
*   **Implications for Modeling:**
    *   **Target Variable:** Treat the target variable as a probabilistic distribution, not a crisp integer. The goal is to predict a **credible range or distribution** of likely outcomes.
    *   **Evaluation:** Evaluate the model's **probabilistic forecasts**, not just its point predictions. Use **proper scoring rules** like the **Continuous Ranked Probability Score (CRPS)** which reward models for producing well-calibrated distributions.
    *   **Loss Function:** The loss function should encourage robustness to label noise. This further strengthens the case for using MAE or Huber Loss over MSE.
