from views_pipeline_core.data.handlers import _ViewsDataset
from typing import Optional, List, Union
from darts import TimeSeries
from views_r2darts2.utils.gates import ReproducibilityGate


class _ViewsDatasetDarts(_ViewsDataset):
    """
    Handles the transformation of multi-index VIEWS dataframes into Darts-compatible TimeSeries collections.

    Intent Contract:
        - Purpose: Act as the data boundary between the generic VIEWS pipeline and the Darts ecosystem,
          ensuring correct mapping of time and entity dimensions.
        - Non-Goals: Does not perform data cleaning or temporal slicing (slicing is handled by the forecaster).
        - Guarantees:
            - Ensures that feature and target columns are correctly grouped by entity.
            - Guarantees that the resulting TimeSeries collection preserves the multi-index semantic structure.
        - Failure Behavior: Raises KeyError if specified target or feature columns are missing from the source dataframe.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # HANDSHAKE: Audit the incoming dataframe schema immediately
        ReproducibilityGate.Data.audit_dataframe_schema(
            df=self.get_subset_dataframe(),
            expected_targets=self.targets,
            expected_features=self.features,
        )

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
