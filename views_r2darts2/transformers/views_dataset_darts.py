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

        # Enforce float32 precision at the Data Airlock boundary (ADR-010).
        # Only cast columns already present in the raw dataframe — cyclic
        # encoder columns (month_sin, month_cos, etc.) are appended to
        # self.features by prior calls but don't exist in df_subset yet;
        # they are injected below on df_reset and cast there.
        cols_to_cast = [c for c in self.features + self.targets if c in df_subset.columns]
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
                df_reset[col_name] = enc_fn(df_reset.index).astype(np.float32)
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

        # Compute fingerprint stats for every target and name them {target_col}_{stat}.
        # This supports both single-target models (e.g. "lr_ged_sb_mu") and
        # multi-target models (e.g. "lr_ged_sb_mu", "lr_ged_ns_mu", ...) without
        # ambiguity. TFT's VSN treats these as plain feature names — the exact
        # strings are irrelevant to the model, only their values matter.
        stat_frames = []
        for target_col in self.targets:
            grouped = df_stat.groupby(self._entity_id)[target_col]
            col_stats = grouped.agg(
                **{
                    f"{target_col}_mu": "mean",
                    f"{target_col}_sigma": "std",
                    f"{target_col}_max": "max",
                }
            ).fillna(0.0)
            col_stats[f"{target_col}_trend"] = grouped.apply(_ols_slope)
            # Sparsity: fraction of time steps with exactly zero target
            col_stats[f"{target_col}_sparsity"] = grouped.apply(lambda s: (s == 0).mean())
            stat_frames.append(col_stats)

        # Merge per-target stat DataFrames on the shared entity_id index
        stats = stat_frames[0]
        for frame in stat_frames[1:]:
            stats = stats.join(frame)
        stats = stats.astype(np.float32).reset_index()

        static_cov_names = [
            f"{col}_{stat}"
            for col in self.targets
            for stat in ("mu", "sigma", "max", "trend", "sparsity")
        ]

        n_entities = len(stats)
        n_stat_rows = len(df_stat)
        n_stat_time_steps = df_stat.index.nunique()
        logger.info(
            f"Static covariates: injecting {static_cov_names} "
            f"({len(self.targets)} target(s) × 5 stats = {len(static_cov_names)} cols) "
            f"for {n_entities} entities computed from {n_stat_rows} rows "
            f"({n_stat_time_steps} time steps, raw pre-transform space)."
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
