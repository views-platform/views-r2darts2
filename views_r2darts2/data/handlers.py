from views_pipeline_core.data.handlers import _ViewsDataset
from typing import Optional, List, Union
from darts import TimeSeries


class _ViewsDatasetDarts(_ViewsDataset):
    """
    This class is a subclass of _ViewsDataset and is used to handle data for the Darts library.
    It provides methods to load and preprocess data specifically for Darts models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def as_darts_timeseries(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
    ):
        """
        Converts the internal data subset into a Darts TimeSeries object.

        Parameters:
            time_ids (Optional[Union[int, List[int]]]): Specific time indices or list of time indices to filter the data. If None, all time indices are included.
            entity_ids (Optional[Union[int, List[int]]]): Specific entity indices or list of entity indices to filter the data. If None, all entity indices are included.

        Returns:
            TimeSeries: A Darts TimeSeries object constructed from the filtered dataframe, grouped by entity and containing the specified features and targets.
        """
        df_reset = self.get_subset_dataframe(
            time_ids=time_ids, entity_ids=entity_ids
        ).reset_index(level=[1])
        return TimeSeries.from_group_dataframe(
            df=df_reset,
            group_cols=self._entity_id,
            value_cols=self.features + self.targets,
        )
