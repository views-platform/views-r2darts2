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
        stat_time_range: Optional[tuple] = None,
        static_cov_transform: Optional[str] = None,
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
            static_cov_transform (Optional[str]): Name of a transform to apply to the
                computed static covariate statistics (mu, sigma, max, trend) before
                injection. Supported: 'AsinhTransform', 'LogTransform', 'SqrtTransform',
                'FourthRootTransform'. When None, stats are injected in raw space.
                Sparsity (already in [0, 1]) is never transformed.

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
        # When static_cov_transform is provided (e.g. 'AsinhTransform'), mu,
        # sigma, max, and trend are transformed before injection so that their
        # scale matches the model's internal representation space. Sparsity
        # (fraction in [0, 1]) is never transformed.
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

        # Resolve the transform chain for static cov stats.
        # Supports single transforms ("AsinhTransform") or chains
        # ("AsinhTransform->MaxAbsScaler"). Element-wise transforms are
        # numpy-level (not Darts Scaler) because static covariates are
        # scalars per entity, not time series. Cross-entity scalers
        # (MaxAbsScaler, StandardScaler) normalize across countries.
        _STATIC_COV_ELEMENTWISE = {
            "AsinhTransform": np.arcsinh,
            "LogTransform": np.log1p,
            "SqrtTransform": lambda x: np.sqrt(np.maximum(x, 0)),
            "FourthRootTransform": lambda x: np.power(1.0 + np.maximum(x, 0.0), 0.25) - 1.0,
        }
        _STATIC_COV_CROSS_ENTITY = {"MaxAbsScaler", "StandardScaler"}

        transform_fn = None
        cross_entity_scaler = None
        if static_cov_transform is not None:
            steps = [s.strip() for s in static_cov_transform.split("->")]
            for step in steps:
                if step in _STATIC_COV_ELEMENTWISE:
                    transform_fn = _STATIC_COV_ELEMENTWISE[step]
                elif step in _STATIC_COV_CROSS_ENTITY:
                    cross_entity_scaler = step
                else:
                    raise ValueError(
                        f"Unknown static_cov_transform step '{step}'. "
                        f"Available elementwise: {list(_STATIC_COV_ELEMENTWISE.keys())}. "
                        f"Available cross-entity: {sorted(_STATIC_COV_CROSS_ENTITY)}."
                    )

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

            # Apply element-wise transform to scale-sensitive stats.
            # Sparsity is already in [0, 1] — never transformed.
            if transform_fn is not None:
                for stat in ("mu", "sigma", "max", "trend"):
                    col_name = f"{target_col}_{stat}"
                    col_stats[col_name] = transform_fn(col_stats[col_name].values).astype(np.float32)

            # Apply cross-entity scaler if specified in the chain.
            if cross_entity_scaler == "MaxAbsScaler":
                for stat in ("mu", "sigma", "max", "trend"):
                    col_name = f"{target_col}_{stat}"
                    abs_max = col_stats[col_name].abs().max()
                    if abs_max > 0:
                        col_stats[col_name] = (col_stats[col_name] / abs_max).astype(np.float32)
            elif cross_entity_scaler == "StandardScaler":
                for stat in ("mu", "sigma", "max", "trend"):
                    col_name = f"{target_col}_{stat}"
                    mean = col_stats[col_name].mean()
                    std = col_stats[col_name].std()
                    if std > 0:
                        col_stats[col_name] = ((col_stats[col_name] - mean) / std).astype(np.float32)

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
        transform_label = static_cov_transform or "raw"
        logger.info(
            f"Static covariates: injecting {static_cov_names} "
            f"({len(self.targets)} target(s) × 5 stats = {len(static_cov_names)} cols) "
            f"for {n_entities} entities computed from {n_stat_rows} rows "
            f"({n_stat_time_steps} time steps, {transform_label} space)."
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
