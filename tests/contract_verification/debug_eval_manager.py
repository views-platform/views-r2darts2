import pandas as pd
import numpy as np
from views_evaluation.evaluation.evaluation_manager import EvaluationManager

def generate_mock_data_from_guide():
    """Creates mock actuals and point predictions in the required schema."""
    target_name = "lr_ged_sb_best"
    pred_col_name = f"pred_{target_name}"
    loc_id_name = "country_id"

    # 1. Actuals DataFrame
    actuals_index = pd.MultiIndex.from_product(
        [range(500, 506), [10, 20]],
        names=['month_id', loc_id_name]
    )
    actuals = pd.DataFrame(
        {target_name: np.random.randint(0, 50, size=len(actuals_index))},
        index=actuals_index
    )
    # Add metadata attributes, a common pattern in views-pipeline-core
    actuals.attrs['target_name'] = target_name
    actuals.attrs['entity_id'] = loc_id_name
    actuals.attrs['time_id'] = 'month_id'

    # 2. Predictions List (2 rolling sequences of 3 steps each)
    predictions_list = []
    # First sequence
    preds_1_index = pd.MultiIndex.from_product(
        [range(500, 503), [10, 20]],
        names=['month_id', loc_id_name]
    )
    preds_1 = pd.DataFrame(
        # Note the required list format for point predictions
        {pred_col_name: [[val] for val in np.random.rand(len(preds_1_index)) * 50]},
        index=preds_1_index
    )
    predictions_list.append(preds_1)

    # Second sequence
    preds_2_index = pd.MultiIndex.from_product(
        [range(501, 504), [10, 20]],
        names=['month_id', loc_id_name]
    )
    preds_2 = pd.DataFrame(
        {pred_col_name: [[val] for val in np.random.rand(len(preds_2_index)) * 50]},
        index=preds_2_index
    )
    predictions_list.append(preds_2)

    return actuals, predictions_list, target_name

if __name__ == "__main__":
    print("1. Generating mock data adhering to the required schema...")
    actuals_data, predictions_data, target = generate_mock_data_from_guide()

    print(f"   Actuals DataFrame shape: {actuals_data.shape}")
    print(f"   Number of prediction sequences: {len(predictions_data)}")
    print(f"   Shape of first prediction sequence: {predictions_data[0].shape}")

    print("\n2. Initializing EvaluationManager...")
    # Define which metrics to run
    metrics = ['RMSLE', 'CRPS', 'Pearson']
    manager = EvaluationManager(metrics_list=metrics)
    print(f"   Metrics to compute: {metrics}")

    print("\n3. Running evaluation...")
    # Define the configuration
    eval_config = {'steps': [1, 2, 3]}
    results_dict = manager.evaluate(
        actual=actuals_data,
        predictions=predictions_data,
        target=target,
        config=eval_config
    )
    print("   Evaluation complete.")

    print("\n4. Processing results...")

    # Access and display the step-wise results DataFrame
    step_wise_results_df = results_dict['step'][1]
    print("\n--- Step-Wise Evaluation Results ---")
    print(step_wise_results_df)

    # Access and display the time-series-wise results DataFrame
    ts_wise_results_df = results_dict['time_series'][1]
    print("\n--- Time-Series-Wise Evaluation Results ---")
    print(ts_wise_results_df)

    # Access and display the month-wise results DataFrame
    month_wise_results_df = results_dict['month'][1]
    print("\n--- Month-Wise Evaluation Results ---")
    print(month_wise_results_df.head()) # Print head for brevity
