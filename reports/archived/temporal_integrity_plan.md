# Temporal Integrity Implementation Plan: The "Glass House" Fortress

This plan outlines the implementation of 5 critical safety checks to ensure 100000% certainty regarding data leakage, temporal continuity, and forecast horizons.

## 1. The Checks (The "What")

### 1.1 The Continuity Guardian ($t+1$)
*   **Requirement**: The test set must start exactly one month after the training set ends.
*   **Logic**: `assert partition['test'].start == partition['train'].end + 1`.
*   **Failure Mode**: Immediate `ValueError`. No silent gaps or overlaps allowed.

### 1.2 The Horizon Siren (36-Month Standard)
*   **Requirement**: Highlight any deviation from the production standard of 36 months.
*   **Logic**: `if len(config['steps']) != 36: trigger_warning()`.
*   **Failure Mode**: Visual warning block in logs; run continues but developer is "painfully aware."

### 1.3 The Leakage Firewall (Set Intersection)
*   **Requirement**: Zero shared time IDs between training and testing data.
*   **Logic**: `set(train_time_ids) & set(test_time_ids) == empty_set`.
*   **Failure Mode**: `RuntimeError` before weights are initialized.

### 1.4 The Sequence Auditor (No Holes)
*   **Requirement**: The training set must be a continuous, contiguous range of month IDs.
*   **Logic**: `sorted(train_time_ids) == list(range(min, max + 1))`.
*   **Failure Mode**: `ValueError` alerting to missing data in the historical series.

### 1.5 Boundary Verification (Bit-Level)
*   **Requirement**: The final `TimeSeries` objects used by Darts must not exceed the `train_end` month ID.
*   **Logic**: `assert targets.max_time_index <= train_end`.
*   **Failure Mode**: `RuntimeError` to prevent "peeking" into the future during training.

---

## 2. Implementation Roadmap (The "How")

### Phase 1: Permanent Test Suite (Target: `tests/`)
Before modifying any library code, I will implement a "Verification Battery" in the test suite.
1.  **Mock Dataset Factory**: Create a utility to generate DataFrames with intentional "holes" or "overlaps" to test our auditors.
2.  **Infrastructure Tests**:
    *   `test_partition_continuity_enforcement`: Proves the manager catches gaps.
    *   `test_hole_detection`: Proves the system catches missing months in training data.
    *   `test_leakage_detection`: Proves the system catches test IDs inside training tensors.

### Phase 2: Runtime Gates (Target: `views_r2darts2/`)
Once tests are passing (proving we can detect the issues), I will implement the gates in the live code.
1.  **`ModelManager._resolve_active_partition_dict`**: Implement Gates 1.1 and 1.2.
2.  **`DartsForecaster._preprocess_timeseries`**: Implement Gate 1.5.
3.  **`_ViewsDatasetDarts`**: Implement Gates 1.3 and 1.4 (Auditing the raw data before it hits the model).

---

## 3. Reflection on Reproducibility
By moving these checks into the **Dataset** and **Pre-processing** layers, we ensure they apply equally to **Sweeps** and **Single Runs**. Because both paths now use the `active_config` snapshot, these gates provide a unified temporal barrier that cannot be bypassed by "clever" configuration merges.
