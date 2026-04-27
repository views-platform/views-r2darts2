from views_pipeline_core.data.handlers import _ViewsDataset
from views_pipeline_core.files.utils import read_dataframe
from views_pipeline_core.configs.pipeline import PipelineConfig
from typing import Optional, List, Union
import logging
import numpy as np
from darts import TimeSeries
from views_r2darts2.infrastructure.encoders import CYCLIC_ENCODERS_BY_RESOLUTION
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
        stat_time_range: Optional[tuple] = None,
    ):
        """
        Converts the internal data subset into a Darts TimeSeries object.

        Parameters:
            time_ids (Optional[Union[int, List[int]]]): Specific time indices or list of time indices to filter the data. If None, all time indices are included.
            entity_ids (Optional[Union[int, List[int]]]): Specific entity indices or list of entity indices to filter the data. If None, all entity indices are included.
            stat_time_range (Optional[tuple]): A (start, end) tuple of time_id boundaries
                (inclusive) for computing static covariate statistics. If None, statistics
                are computed from the entire df_subset — which may leak test-period data
                if df_subset is unfiltered. Callers should always pass the training
                partition boundaries here to prevent leakage.

        Returns:
            TimeSeries: A Darts TimeSeries object constructed from the filtered dataframe, grouped by entity and containing the specified features and targets.
        """
        df_subset = self.get_subset_dataframe(time_ids=time_ids, entity_ids=entity_ids)

        # Enforce float32 precision at the Data Airlock boundary (ADR-010)
        cols_to_cast = self.features + self.targets
        df_subset[cols_to_cast] = df_subset[cols_to_cast].astype(np.float32)

        df_reset = df_subset.reset_index(level=[1])

        # --- Cyclic time encoders (past covariates) ---
        # Infer temporal resolution from the index name (month_id → "m",
        # week_id → "w", day_id → "d", year_id → "y") and inject the
        # corresponding sin/cos encoders from encoders.py.
        # Zero leakage — purely calendar-derived, known for all time steps.
        # Values already in [-1, 1]; no scaling needed.
        resolution = self._time_id.split("_")[0][0]  # "month_id" → "m", "week_id" → "w", etc.
        cyclic_encoders = CYCLIC_ENCODERS_BY_RESOLUTION.get(resolution)
        if cyclic_encoders is not None:
            for enc_fn in cyclic_encoders:
                col_name = enc_fn.__name__
                df_reset[col_name] = enc_fn(df_reset.index)
                if col_name not in self.features:
                    self.features.append(col_name)
            logger.info(
                f"Cyclic encoders: injected {[fn.__name__ for fn in cyclic_encoders]} "
                f"for resolution '{resolution}' ({self._time_id})."
            )

        # --- Per-entity static covariates (entity fingerprint) ---
        #
        # Statistics are computed from df_stat — a time-
        # filtered slice of df_reset restricted to stat_time_range (the
        # training partition). This guarantees that no test/forecast-period
        # target values contaminate the static covariates. The full df_reset
        # (including test rows) still gets the stats joined and is returned
        # as TimeSeries — but the stats themselves only reflect training data.
        #
        # VALUES are in raw (pre-transform) space. AsinhTransform (or
        # whichever target_scaler is configured) is applied downstream by
        # the forecaster on the TimeSeries objects after this method returns.
        #
        # Features injected (all per-entity, all from stat_time_range only):
        #   target_mu      — mean target level
        #   target_sigma   — std of target (spread / volatility)
        #   target_max     — peak observed value (captures spike extremity)
        #   target_trend   — OLS slope of target over time (regime change detector)
        #   target_sparsity— fraction of zero-valued months (structural break signal)
        if stat_time_range is not None:
            stat_start, stat_end = stat_time_range
            df_stat = df_reset.loc[
                (df_reset.index >= stat_start) & (df_reset.index <= stat_end)
            ]
            logger.info(
                f"Static covariate stats restricted to time range [{stat_start}, {stat_end}] "
                f"({df_stat.index.nunique()} time steps, {len(df_stat)} rows). "
                f"Full dataframe has {df_reset.index.nunique()} time steps."
            )
        else:
            df_stat = df_reset
            logger.warning(
                "stat_time_range not provided — static covariate stats computed from "
                "the FULL dataframe. This may cause leakage if test-period data is present."
            )

        target_col = self.targets[0]
        grouped = df_stat.groupby(self._entity_id)[target_col]

        stats = grouped.agg(
            target_mu="mean",
            target_sigma="std",
            target_max="max",
        ).fillna(0.0)

        # OLS slope: Σ(t - t̄)(y - ȳ) / Σ(t - t̄)² per entity
        # Uses integer time index position (0,1,2,...) to avoid scale issues
        def _ols_slope(s):
            t = np.arange(len(s), dtype=np.float64)
            t_centered = t - t.mean()
            y_centered = s.values - s.values.mean()
            denom = (t_centered ** 2).sum()
            if denom == 0:
                return 0.0
            return float((t_centered * y_centered).sum() / denom)

        stats["target_trend"] = grouped.apply(_ols_slope)

        # Sparsity: fraction of months with exactly zero target
        stats["target_sparsity"] = grouped.apply(lambda s: (s == 0).mean())

        stats = stats.astype(np.float32).reset_index()

        static_cov_names = [
            "target_mu", "target_sigma", "target_max",
            "target_trend", "target_sparsity",
        ]

        n_entities = len(stats)
        n_stat_rows = len(df_stat)
        n_stat_time_steps = df_stat.index.nunique()
        logger.info(
            f"Static covariates: injecting {static_cov_names} for {n_entities} entities "
            f"computed from {n_stat_rows} rows ({n_stat_time_steps} time steps). "
            f"All values in raw (pre-transform) space — no test/forecast data used. "
            f"μ ∈ [{stats['target_mu'].min():.2f}, {stats['target_mu'].max():.2f}], "
            f"σ ∈ [{stats['target_sigma'].min():.2f}, {stats['target_sigma'].max():.2f}], "
            f"max ∈ [{stats['target_max'].min():.2f}, {stats['target_max'].max():.2f}], "
            f"trend ∈ [{stats['target_trend'].min():.4f}, {stats['target_trend'].max():.4f}], "
            f"sparsity ∈ [{stats['target_sparsity'].min():.2f}, {stats['target_sparsity'].max():.2f}]."
        )
        df_reset = df_reset.join(stats.set_index(self._entity_id), on=self._entity_id)

        return TimeSeries.from_group_dataframe(
            df=df_reset,
            group_cols=self._entity_id,
            value_cols=self.features + self.targets,
            static_cols=static_cov_names,
            n_jobs=-1,
            verbose=True,
        )
