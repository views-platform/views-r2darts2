from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from darts import TimeSeries
from views_frames.feature_frame import FeatureFrame
import pandas as pd
from views_frames.index import SpatioTemporalIndex
from views_frames.spatial_level import SpatialLevel

from views_r2darts2.infrastructure.encoders import CYCLIC_ENCODERS_BY_RESOLUTION

logger = logging.getLogger(__name__)


class _ViewsDatasetDarts:
    """
    Frames-native dataset boundary for Darts.

    This class intentionally has no dependency on pandas or _ViewsDataset.
    It consumes views-frames FeatureFrame artifacts and exposes Darts-friendly
    grouped TimeSeries collections.
    """

    _ALL_STAT_NAMES = ("mu", "sigma", "max", "trend", "sparsity")

    def __init__(
        self,
        target_frame: FeatureFrame,
        feature_frame: Optional[FeatureFrame] = None,
        targets: Optional[List[str]] = None,
    ):
        self._target_frame = target_frame
        self._feature_frame = feature_frame

        self._time_id = "month_id"
        if target_frame.index.level == SpatialLevel.PGM:
            self._entity_id = "priogrid_id"
        else:
            self._entity_id = "country_id"

        self.targets = list(targets) if targets else list(target_frame.feature_names)
        if len(self.targets) != target_frame.n_features:
            raise ValueError(
                "Number of targets does not match target frame feature axis: "
                f"targets={len(self.targets)} frame_features={target_frame.n_features}"
            )

        self.features = (
            list(feature_frame.feature_names)
            if feature_frame is not None
            else []
        )

        self._t_time = target_frame.index.time.astype(np.int64, copy=False)
        self._t_unit = target_frame.index.unit.astype(np.int64, copy=False)
        self._t_values = target_frame.values[:, :, 0].astype(np.float32, copy=False)

        if feature_frame is not None:
            if feature_frame.n_rows != target_frame.n_rows:
                raise ValueError(
                    "FeatureFrame and TargetFrame must share identical row count."
                )
            if not np.array_equal(feature_frame.index.time, target_frame.index.time):
                raise ValueError("FeatureFrame and TargetFrame time indices are not aligned.")
            if not np.array_equal(feature_frame.index.unit, target_frame.index.unit):
                raise ValueError("FeatureFrame and TargetFrame unit indices are not aligned.")

            self._f_values = feature_frame.values[:, :, 0].astype(np.float32, copy=False)
        else:
            self._f_values = None

        self._unique_time = np.unique(self._t_time)
        self._unique_unit = np.unique(self._t_unit)
        self._last_entity_order: List[int] = []

    @property
    def last_entity_order(self) -> List[int]:
        return list(self._last_entity_order)

    @staticmethod
    def _first_existing_path(candidates: List[str]) -> Optional[str]:
        for p in candidates:
            if p and os.path.exists(p):
                return p
        return None

    @staticmethod
    def from_dataframe(
        dataframe,
        targets: List[str],
        features: Optional[List[str]] = None,
    ):
        """
        Build a frames-native dataset from a pandas DataFrame.

        Accepted layouts:
        - MultiIndex: (month_id, country_id|priogrid_id|priogrid_gid)
        - Flat columns: month_id + country_id/priogrid_id/priogrid_gid
        """
        import pandas as pd

        if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
            raise ValueError("dataframe must be a non-empty pandas DataFrame")

        df = dataframe.copy()

        if isinstance(df.index, pd.MultiIndex):
            if len(df.index.names) < 2:
                raise ValueError("MultiIndex dataframe must have at least two levels")
            time_col, unit_col = df.index.names[0], df.index.names[1]
            if unit_col == "priogrid_gid":
                unit_col = "priogrid_id"
                df.index = df.index.rename([time_col, unit_col])
            df_reset = df.reset_index()
        else:
            candidate_units = ["country_id", "priogrid_id", "priogrid_gid"]
            time_col = "month_id"
            if time_col not in df.columns:
                raise ValueError("DataFrame must include month_id column or MultiIndex")
            unit_col = next((c for c in candidate_units if c in df.columns), None)
            if unit_col is None:
                raise ValueError(
                    "DataFrame must include one of country_id, priogrid_id, priogrid_gid"
                )
            if unit_col == "priogrid_gid":
                df = df.rename(columns={"priogrid_gid": "priogrid_id"})
                unit_col = "priogrid_id"
            df_reset = df

        if not targets:
            raise ValueError("targets must be a non-empty list")
        missing_targets = [t for t in targets if t not in df_reset.columns]
        if missing_targets:
            raise ValueError(f"Missing targets in dataframe: {missing_targets}")

        if features is None:
            excluded = {time_col, unit_col, *targets}
            features = [c for c in df_reset.columns if c not in excluded]

        df_reset = df_reset.sort_values([time_col, unit_col], kind="stable")

        if unit_col == "priogrid_id":
            level = SpatialLevel.PGM
        else:
            level = SpatialLevel.CM

        st_index = SpatioTemporalIndex(
            time=df_reset[time_col].to_numpy(dtype=np.int64),
            unit=df_reset[unit_col].to_numpy(dtype=np.int64),
            level=level,
        )

        target_values = df_reset[targets].to_numpy(dtype=np.float32)
        target_frame = FeatureFrame.from_2d(
            y_features_2d=target_values,
            index=st_index,
            feature_names=list(targets),
        )

        feature_frame = None
        if features:
            feature_values = df_reset[features].to_numpy(dtype=np.float32)
            feature_frame = FeatureFrame.from_2d(
                y_features_2d=feature_values,
                index=st_index,
                feature_names=list(features),
            )

        return _ViewsDatasetDarts(
            target_frame=target_frame,
            feature_frame=feature_frame,
            targets=list(targets),
        )

    @staticmethod
    def from_views_path(path_raw: str, run_type: str, config: dict, cached_path=None):
        """
        Load a frames-native dataset.

        Resolution order:
        1. config['frames_target_path'] / config['frames_feature_path']
        2. <path_raw>/<run_type>_target_frame and <path_raw>/<run_type>_feature_frame
        3. <path_raw>/<run_type>/target_frame and <path_raw>/<run_type>/feature_frame
        """
        if cached_path is not None:
            cached_root = str(cached_path)
        else:
            cached_root = None

        target_candidates = [
            config.get("frames_target_path"),
            f"{path_raw}/{run_type}_target_frame",
            f"{path_raw}/{run_type}/target_frame",
            f"{cached_root}/target_frame" if cached_root else None,
        ]
        feature_candidates = [
            config.get("frames_feature_path"),
            f"{path_raw}/{run_type}_feature_frame",
            f"{path_raw}/{run_type}/feature_frame",
            f"{cached_root}/feature_frame" if cached_root else None,
        ]

        target_path = _ViewsDatasetDarts._first_existing_path([p for p in target_candidates if p])

        feature_path = _ViewsDatasetDarts._first_existing_path([p for p in feature_candidates if p])

        if target_path is not None:
            logger.info(f"Loading target frame from: {target_path}")
            target_frame = FeatureFrame.load(target_path, mmap=True)

            feature_frame = None
            if feature_path is not None:
                logger.info(f"Loading feature frame from: {feature_path}")
                feature_frame = FeatureFrame.load(feature_path, mmap=True)

            return _ViewsDatasetDarts(
                target_frame=target_frame,
                feature_frame=feature_frame,
                targets=config.get("targets"),
            )

        # Fall back to the legacy multi-index parquet/feather file at path_raw.
        # path_raw may be:
        #   a) a directory containing {run_type}_viewser_df.{ext} files, or
        #   b) a direct path to a parquet/feather file (path_raw itself is the file).
        from views_pipeline_core.configs.pipeline import PipelineConfig
        from views_pipeline_core.files.utils import read_dataframe

        legacy_candidates = [
            f"{path_raw}/{run_type}_viewser_df{PipelineConfig.dataframe_format}",
            f"{path_raw}/{run_type}_viewser_df.parquet",
            # path_raw may itself be a direct file path
            str(path_raw) if os.path.isfile(str(path_raw)) else None,
        ]
        legacy_path = _ViewsDatasetDarts._first_existing_path(
            [p for p in legacy_candidates if p]
        )
        if legacy_path is None:
            raise FileNotFoundError(
                "Could not locate frame artifacts or a legacy dataframe file.\n"
                f"  Frame paths searched: {[p for p in target_candidates if p]}\n"
                f"  Legacy paths searched: {[p for p in legacy_candidates if p]}"
            )

        logger.info(
            "Frame artifacts not found — loading legacy dataframe and converting to "
            f"views-frames in-memory: {legacy_path}"
        )
        df_source = read_dataframe(legacy_path)
        return _ViewsDatasetDarts.from_dataframe(
            dataframe=df_source,
            targets=config.get("targets") or [],
            features=config.get("features"),
        )

    def _resolve_time_ids(self, time_ids: Optional[Union[int, List[int]]]) -> np.ndarray:
        if time_ids is None:
            return self._unique_time
        if isinstance(time_ids, int):
            return np.asarray([time_ids], dtype=np.int64)
        return np.asarray(sorted(set(time_ids)), dtype=np.int64)

    def _resolve_entity_ids(
        self,
        entity_ids: Optional[Union[int, List[int]]],
        selected_times: np.ndarray,
    ) -> np.ndarray:
        if entity_ids is None:
            mask = np.isin(self._t_time, selected_times)
            return np.asarray(sorted(set(self._t_unit[mask].tolist())), dtype=np.int64)
        if isinstance(entity_ids, int):
            return np.asarray([entity_ids], dtype=np.int64)
        return np.asarray(sorted(set(entity_ids)), dtype=np.int64)

    @staticmethod
    def _build_stat_transform(static_cov_transform: Optional[str]):
        if static_cov_transform is None:
            return None

        elementwise = {
            "AsinhTransform": np.arcsinh,
            "LogTransform": np.log1p,
            "SqrtTransform": lambda x: np.sqrt(np.maximum(x, 0)),
            "FourthRootTransform": lambda x: np.power(1.0 + np.maximum(x, 0.0), 0.25) - 1.0,
        }

        fn = None
        for step in [s.strip() for s in static_cov_transform.split("->")]:
            if step in elementwise:
                fn = elementwise[step]
                continue
            if step in {"MaxAbsScaler", "StandardScaler"}:
                # Cross-entity normalizations are intentionally skipped here; they are
                # better applied in the explicit scaler path to avoid hidden leakage.
                continue
            raise ValueError(f"Unknown static_cov_transform step: {step}")
        return fn

    def _compute_static_covariates(
        self,
        entity_id: int,
        stat_time_range: Optional[tuple],
        static_cov_transform: Optional[str],
        static_cov_stats: Optional[List[str]],
    ) -> Optional[np.ndarray]:
        requested = tuple(static_cov_stats) if static_cov_stats else self._ALL_STAT_NAMES
        unknown = set(requested) - set(self._ALL_STAT_NAMES)
        if unknown:
            raise ValueError(f"Unknown static_cov_stats: {unknown}")

        mask = self._t_unit == entity_id
        if stat_time_range is not None:
            stat_start, stat_end = stat_time_range
            mask = mask & (self._t_time >= stat_start) & (self._t_time <= stat_end)

        if not np.any(mask):
            return None

        vals = self._t_values[mask]
        times = self._t_time[mask]
        order = np.argsort(times, kind="stable")
        vals = vals[order]

        transform_fn = self._build_stat_transform(static_cov_transform)
        pieces: List[np.ndarray] = []
        for target_idx in range(vals.shape[1]):
            series = vals[:, target_idx].astype(np.float64)
            stat_map: Dict[str, float] = {}
            if "mu" in requested:
                stat_map["mu"] = float(np.mean(series))
            if "sigma" in requested:
                stat_map["sigma"] = float(np.std(series))
            if "max" in requested:
                stat_map["max"] = float(np.max(series))
            if "trend" in requested:
                t = np.arange(len(series), dtype=np.float64)
                tc = t - t.mean()
                yc = series - series.mean()
                denom = float(np.sum(tc * tc))
                stat_map["trend"] = 0.0 if denom == 0.0 else float(np.sum(tc * yc) / denom)
            if "sparsity" in requested:
                stat_map["sparsity"] = float(np.mean(series == 0.0))

            for stat in requested:
                x = stat_map[stat]
                if transform_fn is not None and stat != "sparsity":
                    x = float(transform_fn(np.asarray([x], dtype=np.float64))[0])
                pieces.append(np.asarray([x], dtype=np.float32))

        if not pieces:
            return None
        return np.concatenate(pieces, axis=0)

    def as_views_frames(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        use_cyclic_encoders: bool = False,
    ) -> Tuple[FeatureFrame, Optional[FeatureFrame]]:
        selected_times = self._resolve_time_ids(time_ids)
        selected_entities = self._resolve_entity_ids(entity_ids, selected_times)

        row_mask = np.isin(self._t_time, selected_times) & np.isin(self._t_unit, selected_entities)
        row_idx = np.where(row_mask)[0]

        target_sel = self._target_frame.select(row_idx)
        feature_sel = self._feature_frame.select(row_idx) if self._feature_frame is not None else None

        if use_cyclic_encoders:
            resolution = self._time_id.split("_")[0][0]
            cyclic_encoders = CYCLIC_ENCODERS_BY_RESOLUTION.get(resolution)
            if cyclic_encoders:
                tvals = target_sel.index.time.astype(np.float32)
                enc = [fn(tvals).astype(np.float32) for fn in cyclic_encoders]
                enc_matrix = np.stack(enc, axis=1)[:, :, np.newaxis]
                enc_names = [fn.__name__ for fn in cyclic_encoders]

                if feature_sel is None:
                    feature_sel = FeatureFrame(enc_matrix, target_sel.index, enc_names)
                else:
                    merged_vals = np.concatenate([feature_sel.values, enc_matrix], axis=1)
                    merged_names = feature_sel.feature_names + enc_names
                    feature_sel = FeatureFrame(merged_vals, feature_sel.index, merged_names)

        return target_sel, feature_sel

    def as_darts_timeseries(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        stat_time_range: Optional[tuple] = None,
        static_cov_transform: Optional[str] = None,
        static_cov_stats: Optional[List[str]] = None,
        inject_static_covariates: bool = False,
        use_cyclic_encoders: bool = False,
    ):
        """
        Convert frames-native arrays into per-entity Darts TimeSeries.
        """
        selected_times = self._resolve_time_ids(time_ids)
        selected_entities = self._resolve_entity_ids(entity_ids, selected_times)
        selected_times = np.sort(selected_times)
        selected_entities = np.sort(selected_entities)
        self._last_entity_order = selected_entities.astype(int).tolist()

        target_sel, feature_sel = self.as_views_frames(
            time_ids=selected_times.tolist(),
            entity_ids=selected_entities.tolist(),
            use_cyclic_encoders=use_cyclic_encoders,
        )

        t_time = target_sel.index.time.astype(np.int64, copy=False)
        t_unit = target_sel.index.unit.astype(np.int64, copy=False)
        t_values = target_sel.values[:, :, 0].astype(np.float32, copy=False)

        f_values = None
        feature_names: List[str] = []
        if feature_sel is not None:
            f_values = feature_sel.values[:, :, 0].astype(np.float32, copy=False)
            feature_names = list(feature_sel.feature_names)

        min_t = int(selected_times.min())
        max_t = int(selected_times.max())
        full_times_np = np.arange(min_t, max_t + 1, dtype=np.int64)
        full_times = pd.Index(full_times_np)  # Darts requires a pandas Index
        time_to_pos = {t: i for i, t in enumerate(full_times_np.tolist())}

        row_map: Dict[Tuple[int, int], int] = {
            (int(tt), int(uu)): idx for idx, (tt, uu) in enumerate(zip(t_time.tolist(), t_unit.tolist()))
        }

        out = []
        for ent in selected_entities.tolist():
            target_dense = np.zeros((len(full_times_np), len(self.targets)), dtype=np.float32)
            feature_dense = (
                np.zeros((len(full_times_np), len(feature_names)), dtype=np.float32)
                if f_values is not None
                else None
            )

            for t in full_times_np.tolist():
                ridx = row_map.get((int(t), int(ent)))
                if ridx is None:
                    continue
                pos = time_to_pos[int(t)]
                target_dense[pos, :] = t_values[ridx, :]
                if feature_dense is not None:
                    feature_dense[pos, :] = f_values[ridx, :]

            if feature_dense is not None:
                values = np.concatenate([feature_dense, target_dense], axis=1)
                cols = feature_names + self.targets
            else:
                values = target_dense
                cols = list(self.targets)

            static_cov = None
            if inject_static_covariates:
                static_cov = self._compute_static_covariates(
                    entity_id=int(ent),
                    stat_time_range=stat_time_range,
                    static_cov_transform=static_cov_transform,
                    static_cov_stats=static_cov_stats,
                )

            ts = TimeSeries.from_times_and_values(
                times=full_times,
                values=values.astype(np.float32),
                columns=cols,
                static_covariates=static_cov,
            )
            out.append(ts)

        return out
