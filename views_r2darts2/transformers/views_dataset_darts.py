from views_pipeline_core.data.handlers import _ViewsDataset
from views_pipeline_core.files.utils import read_dataframe
from views_pipeline_core.configs.pipeline import PipelineConfig
from typing import Optional, List, Union
import numpy as np
from darts import TimeSeries
from views_r2darts2.infrastructure.reproducibility_gate import ReproducibilityGate


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

    @staticmethod
    def from_views_path(path_raw: str, run_type: str, config: dict, cached_path=None):
        """
        Factory method to load a VIEWS dataset from a raw path and configuration.

        Args:
            path_raw (str): Path to the directory containing the raw dataframes.
            run_type (str): The run type (e.g., 'validation', 'calibration').
            config (dict): The DNA manifest for the experiment.
            cached_path: Optional pre-resolved path to the cached data file.
                If provided, this path is used directly instead of constructing
                one from path_raw and run_type.

        Returns:
            _ViewsDatasetDarts: Initialized dataset object.
        """
        if cached_path is not None:
            file_path = str(cached_path)
        else:
            file_path = f"{path_raw}/{run_type}_viewser_df{PipelineConfig.dataframe_format}"
        df_source = read_dataframe(file_path)

        return _ViewsDatasetDarts(
            source=df_source,
            targets=config.get("targets"),
            broadcast_features=True,
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
        df_subset = self.get_subset_dataframe(time_ids=time_ids, entity_ids=entity_ids)

        # Enforce float32 precision at the Data Airlock boundary (ADR-010)
        cols_to_cast = self.features + self.targets
        df_subset[cols_to_cast] = df_subset[cols_to_cast].astype(np.float32)

        df_reset = df_subset.reset_index(level=[1])
        return TimeSeries.from_group_dataframe(
            df=df_reset,
            group_cols=self._entity_id,
            value_cols=self.features + self.targets,
        )
