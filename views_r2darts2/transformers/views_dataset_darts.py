from views_pipeline_core.data.handlers import _ViewsDataset
from views_pipeline_core.files.utils import read_dataframe
from views_pipeline_core.configs.pipeline import PipelineConfig
from typing import Optional, List, Union
import logging
import numpy as np
from darts import TimeSeries
from views_r2darts2.infrastructure.reproducibility_gate import ReproducibilityGate

logger = logging.getLogger(__name__)


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
    def from_views_path(path_raw: str, run_type: str, config: dict):
        """
        Factory method to load a VIEWS dataset from a raw path and configuration.
        
        Args:
            path_raw (str): Path to the directory containing the raw dataframes.
            run_type (str): The run type (e.g., 'validation', 'calibration').
            config (dict): The DNA manifest for the experiment.
            
        Returns:
            _ViewsDatasetDarts: Initialized dataset object.
        """
        file_path = f"{path_raw}/{run_type}_viewser_df{PipelineConfig.dataframe_format}"
        df_viewser = read_dataframe(file_path)
        
        return _ViewsDatasetDarts(
            source=df_viewser,
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

        # --- Per-entity target statistics as static covariates ---
        # These give the model a semantically meaningful per-entity fingerprint
        # (mean conflict level, variability) that static covariates like country_id
        # (an arbitrary integer) cannot provide.
        #
        # stats are computed exclusively from df_subset — the
        # already time-filtered visible input window. The model sees every row
        # in df_subset as input already; summarizing them as static covariates
        # introduces no information beyond what is present in the input.
        #
        # TRANSFORM NOTE: values are in raw (pre-transform) space. AsinhTransform
        # (or whichever target_scaler is configured) is applied downstream by the
        # forecaster on the TimeSeries objects after this method returns.
        # The raw-space fingerprint is still geometrically meaningful:
        # Ukraine (mean ~500 deaths) is far from Switzerland (mean ~0) in raw space.
        target_col = self.targets[0]
        stats = (
            df_reset.groupby(self._entity_id)[target_col]
            .agg(target_mu="mean", target_sigma="std")
            .fillna(0.0)
            .astype(np.float32)
            .reset_index()
        )
        n_entities = len(stats)
        mu_min, mu_max = stats["target_mu"].min(), stats["target_mu"].max()
        sigma_min, sigma_max = stats["target_sigma"].min(), stats["target_sigma"].max()
        n_rows = len(df_reset)
        n_time_steps = df_reset.index.nunique()
        logger.info(
            f"Static covariates: injecting target_mu and target_sigma for {n_entities} entities "
            f"computed from {n_rows} rows ({n_time_steps} time steps) in the visible input window. "
            f"Values are in raw (pre-transform) space — no future target data used. "
            f"μ ∈ [{mu_min:.2f}, {mu_max:.2f}], σ ∈ [{sigma_min:.2f}, {sigma_max:.2f}]."
        )
        df_reset = df_reset.join(stats.set_index(self._entity_id), on=self._entity_id)

        return TimeSeries.from_group_dataframe(
            df=df_reset,
            group_cols=self._entity_id,
            value_cols=self.features + self.targets,
            static_cols=["target_mu", "target_sigma"],
        )
