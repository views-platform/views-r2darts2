import pytest
import pandas as pd
import numpy as np
from darts.models.forecasting.tcn_model import TCNModel

from views_r2darts2.data.handlers import _ViewsDatasetDarts
from views_r2darts2.model.forecaster import DartsForecaster


@pytest.fixture
def dummy_df() -> pd.DataFrame:
    """Creates a simple DataFrame for testing feature transformations."""
    data = {
        "time": range(20),
        "entity_id": [1] * 20,
        "target1": np.random.rand(20),
        "feature_A": [10.0] * 20,  # Constant value for easy testing
        "feature_B": np.random.rand(20),
    }
    df = pd.DataFrame(data)
    df.set_index(["time", "entity_id"], inplace=True)
    return df


@pytest.fixture
def dummy_dataset(dummy_df: pd.DataFrame) -> _ViewsDatasetDarts:
    """Creates a _ViewsDatasetDarts instance from the dummy DataFrame."""
    dataset = _ViewsDatasetDarts(
        source=dummy_df,
        targets=["target1"],
        broadcast_features=True,
    )
    # Manually set other attributes that are accessed by the forecaster
    # but are not part of this specific constructor's known signature.
    dataset.features = ["feature_A", "feature_B"]
    dataset._time_id = "time"
    dataset._entity_id = "entity_id"
    return dataset


@pytest.fixture
def dummy_model() -> TCNModel:
    """Creates a minimal TCNModel for testing purposes."""
    return TCNModel(
        input_chunk_length=12,
        output_chunk_length=6,
        n_epochs=1,
        random_state=42,
    )


@pytest.fixture
def partition_dict() -> dict:
    """Creates a simple partition dictionary for training."""
    return {"train": (0, 18), "test": (18, 20)}


class TestDartsForecasterLogFeatures:
    """
    Test suite to verify the behavior of the `log_features` parameter
    in the DartsForecaster.
    """

    @pytest.mark.parametrize(
        "log_features_config, expected_value_A",
        [
            # Positive Control: feature_A is specified and should be transformed.
            (["feature_A"], np.log1p(10.0)),
            # Scenario: log_features is an empty list, no transformation should occur.
            ([], 10.0),
            # Scenario: log_features is None, simulating an absent config key.
            # No transformation should occur.
            (None, 10.0),
        ],
    )
    def test_log_features_behavior(
        self,
        dummy_dataset,
        dummy_model,
        partition_dict,
        log_features_config,
        expected_value_A,
    ):
        """
        Tests that `log_features` correctly applies or skips log transformation.

        Args:
            log_features_config: The configuration to be passed for `log_features`.
            expected_value_A: The expected value of feature_A after preprocessing.
        """
        # 1. Initialize the Forecaster with the specific test configuration
        forecaster = DartsForecaster(
            dataset=dummy_dataset,
            model=dummy_model,
            partition_dict=partition_dict,
            log_features=log_features_config,
        )

        # 2. Get the raw TimeSeries list from the dataset
        timeseries_list = dummy_dataset.as_darts_timeseries()

        # 3. Trigger the preprocessing step directly
        _, past_covariates = forecaster._preprocess_timeseries(
            timeseries=timeseries_list,
            start=partition_dict["train"][0],
            end=partition_dict["train"][1],
            train_mode=True,
        )

        # 4. Extract the processed values for verification
        # We have one entity, so we inspect the first TimeSeries in the list.
        processed_values = past_covariates[0].all_values(copy=False)

        # Find the column index for 'feature_A'
        feature_a_index = past_covariates[0].components.get_loc("feature_A")

        # Get all values for feature_A
        feature_a_values = processed_values[:, feature_a_index]

        # 5. Assert that all values for feature_A match the expected value
        np.testing.assert_allclose(
            feature_a_values,
            expected_value_A,
            err_msg=f"Failed for log_features config: {log_features_config}",
        )

        # 6. Sanity check: Assert that feature_B was NOT transformed
        feature_b_index = past_covariates[0].components.get_loc("feature_B")
        feature_b_values = processed_values[:, feature_b_index]
        # Important: covariates were sliced during train_mode=True
        start_idx = partition_dict["train"][0]
        end_idx = partition_dict["train"][1]
        original_b_values = timeseries_list[0][start_idx : end_idx + 1]["feature_B"].all_values(copy=False)

        np.testing.assert_allclose(
            feature_b_values.flatten(),
            original_b_values.flatten(),
            err_msg="feature_B should not have been transformed.",
        )
