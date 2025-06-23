from views_pipeline_core.data.handlers import _ViewsDataset
from typing import Optional, List, Dict, Union
import pandas as pd
from darts import TimeSeries


class _ViewsDatasetDarts(_ViewsDataset):
    """
    This class is a subclass of _ViewsDataset and is used to handle data for the Darts library.
    It provides methods to load and preprocess data specifically for Darts models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def as_darts_timeseries(
    #     self,
    #     time_ids: Optional[List[int]] = None,
    #     entity_ids: Optional[List[int]] = None,
    #     static_covariates: Optional[Dict[str, Union[pd.Series, pd.DataFrame]]] = None,
    #     group_entities: bool = True,
    # ) -> Union["TimeSeries", List["TimeSeries"]]:
    #     """
    #     Converts the dataset into one or more darts.TimeSeries objects for time series forecasting.

    #     Parameters:
    #     time_ids (Optional[List[int]]): Subset of time IDs to include.
    #     entity_ids (Optional[List[int]]): Subset of entity IDs to include.
    #     static_covariates (Optional[Dict[str, Union[pd.Series, pd.DataFrame]]]): Static covariates per entity.
    #     group_entities (bool): Whether to group by entity to create multiple TimeSeries. Defaults to True.

    #     Returns:
    #     Union[TimeSeries, List[TimeSeries]]: A single TimeSeries if group_entities=False and one entity exists,
    #     otherwise a list of TimeSeries.

    #     Raises:
    #     ImportError: If the darts library is not installed.
    #     ValueError: For invalid dataset structure, empty data after filtering, or incorrect grouping.
    #     """
    #     try:
    #         from darts import TimeSeries
    #     except ImportError as e:
    #         raise ImportError(
    #             "The darts library is required to use this method. Please install darts."
    #         ) from e

    #     # Validate dataset structure
    #     if not isinstance(self.dataframe.index, pd.MultiIndex) or len(self.dataframe.index.names) != 2:
    #         raise ValueError("Dataset must have a MultiIndex with exactly two levels (time and entity).")

    #     entity_col = self._entity_id
    #     time_col = self._time_id

    #     # Filter the dataset
    #     subset_df = self.get_subset_dataframe(time_ids=time_ids, entity_ids=entity_ids)
    #     if subset_df.empty:
    #         raise ValueError("No data available after applying time and entity filters.")

    #     # Get full time index range for the filtered subset
    #     time_min = subset_df.index.get_level_values(time_col).min()
    #     time_max = subset_df.index.get_level_values(time_col).max()
    #     full_time_index = pd.RangeIndex(
    #         start=time_min, stop=time_max + 1, step=1, name=time_col
    #     )

    #     if group_entities:
    #         # Group by entity and create a TimeSeries per group
    #         grouped = subset_df.groupby(level=entity_col, group_keys=False)
    #         series_list = []

    #         for entity_id, group in grouped:
    #             group_df = group.droplevel(entity_col)

    #             # Reindex to include all time steps in the filtered range and fill missing values with 0
    #             group_df = group_df.reindex(full_time_index, fill_value=0)
    #             group_df = group_df.sort_index()

    #             try:
    #                 # Create TimeSeries with integer-based frequency
    #                 ts = TimeSeries.from_dataframe(group_df, freq=1)
    #             except Exception as e:
    #                 raise ValueError(f"Failed to create TimeSeries for entity {entity_id}: {e}") from e

    #             # Add static covariates
    #             if static_covariates:
    #                 static_data = {}
    #                 for covar_name, covar in static_covariates.items():
    #                     if isinstance(covar, pd.Series):
    #                         if entity_id in covar.index:
    #                             static_data[covar_name] = covar.loc[entity_id]
    #                     elif isinstance(covar, pd.DataFrame):
    #                         if entity_id in covar.index:
    #                             entity_covariates = covar.loc[entity_id]
    #                             if isinstance(entity_covariates, pd.Series):
    #                                 static_data.update(entity_covariates.to_dict())
    #                             else:
    #                                 raise ValueError(f"Static covariate {covar_name} has multiple rows for entity {entity_id}.")
    #                     else:
    #                         raise TypeError(f"Static covariate {covar_name} must be a pandas Series or DataFrame.")
    #                 if static_data:
    #                     ts = ts.with_static_covariates(pd.DataFrame([static_data]))

    #             series_list.append(ts)

    #         return series_list
    #     else:
    #         # Check for single entity
    #         entities = subset_df.index.get_level_values(entity_col).unique()
    #         if len(entities) > 1:
    #             raise ValueError(
    #                 "Multiple entities found when group_entities=False. "
    #                 "Set group_entities=True or filter to a single entity."
    #             )

    #         # Create single TimeSeries
    #         single_df = subset_df.droplevel(entity_col)

    #         # Reindex to full time range and fill missing values
    #         single_df = single_df.reindex(full_time_index, fill_value=0)
    #         single_df = single_df.sort_index()

    #         try:
    #             ts = TimeSeries.from_dataframe(single_df, freq=1)
    #         except Exception as e:
    #             raise ValueError(f"Failed to create TimeSeries: {e}") from e

    #         # Add static covariates
    #         if static_covariates and len(entities) == 1:
    #             entity_id = entities[0]
    #             static_data = {}
    #             for covar_name, covar in static_covariates.items():
    #                 if isinstance(covar, pd.Series):
    #                     if entity_id in covar.index:
    #                         static_data[covar_name] = covar.loc[entity_id]
    #                 elif isinstance(covar, pd.DataFrame):
    #                     if entity_id in covar.index:
    #                         entity_covariates = covar.loc[entity_id]
    #                         static_data.update(entity_covariates.to_dict())
    #                 else:
    #                     raise TypeError(f"Static covariate {covar_name} must be a pandas Series or DataFrame.")
    #             if static_data:
    #                 ts = ts.with_static_covariates(pd.DataFrame([static_data]))

    #         return ts

    def as_darts_timeseries(self, time_ids: Optional[Union[int, List[int]]] = None, entity_ids: Optional[Union[int, List[int]]] = None):
        df_reset = self.get_subset_dataframe(time_ids=time_ids, entity_ids=entity_ids).reset_index(level=[1])
        return TimeSeries.from_group_dataframe(
            df=df_reset,
            group_cols=self._entity_id,
            value_cols=self.features + self.targets
        )