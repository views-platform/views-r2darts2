"""``ViewsDataset`` — a disk-backed, lazy replacement for the pandas dataset.

The whole dataset lives as chunked Zarr arrays in a temp directory managed by
:class:`ZarrStore`; nothing is read into RAM until a caller forces it. Every
accessor returns lazy, Dask-backed ``xarray`` objects, so peak memory is bounded
by the largest chunk rather than the dataset size. Construction accepts any
supported input kind and delegates the on-disk write to the matching converter.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from views_r2darts2.dataset import converters, readers
from views_r2darts2.dataset.zarr_store import ZarrStore

logger = logging.getLogger(__name__)

_ENTITY_LEVEL = {"priogrid_id": "PGM", "country_id": "CM"}


class ViewsDataset:
    """A lazy, Zarr-backed spatiotemporal dataset (time × entity × sample)."""

    def __init__(
        self,
        source: Any,
        targets: list[str] | None = None,
        broadcast_features: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.broadcast_features = broadcast_features
        self._user_metadata = metadata or {}
        self._store = ZarrStore()
        zarr_path = self._ingest(source, targets, broadcast_features)
        self._zarr_path = zarr_path  # disk-backed zarr group path
        self._ds = readers.open_zarr_dir(zarr_path)
        self._load_schema()
        self.validate_indices()

    # ---- construction -------------------------------------------------------
    def _ingest(
        self, source: Any, targets: list[str] | None, broadcast_features: bool
    ) -> Path:
        kind = readers.detect_source_type(source)
        target = self._store.path / f"dataset_{uuid.uuid4().hex[:8]}.zarr"
        extra_attrs = self._user_metadata

        if kind == "dataframe":
            return converters.DataFrameConverter.to_zarr(
                source, target, targets=targets,
                broadcast_features=broadcast_features, extra_attrs=extra_attrs,
            )
        if kind == "parquet":
            return converters.ParquetConverter.to_zarr(
                Path(source), target, targets=targets,
                broadcast_features=broadcast_features, extra_attrs=extra_attrs,
            )
        if kind == "prediction_frame":
            name = _single_target(targets, "PredictionFrame")
            return converters.PredictionFrameConverter.to_zarr(
                source, target, target=name, extra_attrs=extra_attrs,
            )
        if kind == "feature_frame":
            return converters.FeatureFrameConverter.to_zarr(
                source, target, targets=targets,
                broadcast_features=broadcast_features, extra_attrs=extra_attrs,
            )
        if kind == "zarr_dir":
            readers.open_zarr_dir(source).to_zarr(target, mode="w", consolidated=False)
            return target
        if kind == "zarr_zip":
            readers.open_zarr_zip(source).to_zarr(target, mode="w", consolidated=False)
            return target
        if kind == "dataset":
            source.to_zarr(target, mode="w", consolidated=False)
            return target
        raise TypeError(f"Unsupported source kind: {kind}")

    def _load_schema(self) -> None:
        attrs = self._ds.attrs
        self._time_id = attrs["time_id"]
        self._entity_id = attrs["entity_id"]
        self.is_prediction = bool(attrs["is_prediction"])
        self.sample_size = int(attrs["sample_size"])
        self.targets = list(attrs["targets"])
        self.features = list(attrs["features"])
        self.pred_vars = list(attrs["pred_vars"])
        self.text_cols = list(attrs.get("text_cols", []))
        self.broadcast_features = bool(attrs.get("broadcast_features", self.broadcast_features))
        self.metadata = {k: v for k, v in attrs.items()
                         if k not in ("is_prediction", "sample_size", "targets",
                                       "features", "pred_vars", "text_cols",
                                       "time_id", "entity_id", "broadcast_features")}

    # ---- column roles -------------------------------------------------------
    def get_pred_vars(self) -> list[str]:
        """Column names starting with ``pred_``."""
        return [c for c in self._ds.data_vars if c.startswith("pred_")]

    def get_features(self) -> list[str]:
        """Numeric, non-target, non-text columns."""
        return list(self.features)

    # ---- tensor surface -----------------------------------------------------
    def to_tensor(self, include_targets: bool = True) -> xr.DataArray:
        """Lazy ``(time, entity, sample, variable)`` DataArray, Dask-backed."""
        if self.is_prediction:
            # Prediction datasets store variables with a ``pred_`` prefix.
            # Use ``self.pred_vars`` (the actual data variable names) when
            # available; fall back to ``pred_<target>`` for each target.
            names = list(self.pred_vars) if self.pred_vars else [
                f"pred_{t}" for t in self.targets
            ]
        elif include_targets:
            names = self.features + self.targets
        else:
            names = list(self.features)
        return self._stack_variables(names)

    def _stack_variables(self, names: list[str]) -> xr.DataArray:
        if not names:
            raise ValueError("No variables to stack into a tensor")
        arrays = [self._var_as_3d(name) for name in names]
        stacked = xr.concat(arrays, dim="variable")
        stacked = stacked.assign_coords(variable=names)
        return stacked.transpose(self._time_id, self._entity_id, "sample", "variable")

    def _var_as_3d(self, name: str) -> xr.DataArray:
        var = self._ds[name]
        if "sample" in var.dims:
            return var
        if not self.broadcast_features:
            raise ValueError(
                "Tensor operations are disabled for scalar features when "
                "broadcast_features=False"
            )
        broadcast, _ = xr.broadcast(var, self._ds["sample"])
        return broadcast

    def get_subset_tensor(
        self,
        time_ids: Any = None,
        features: Any = None,
        sample_idx: Any = None,
        entity_ids: Any = None,
    ) -> xr.DataArray:
        """Lazily subset the tensor via ``.sel`` / ``.isel`` — stays Dask-backed."""
        tensor = self.to_tensor()
        tensor = self._apply_selection(tensor, time_ids, entity_ids, sample_idx)
        if features is not None:
            tensor = tensor.sel(variable=_as_list(features))
        return tensor

    def get_subset_dataset(
        self,
        time_ids: Any = None,
        features: Any = None,
        sample_idx: Any = None,
        entity_ids: Any = None,
    ) -> "ViewsDataset":
        """Materialize a subset into a new, independent dataset object."""
        ds = self._apply_selection(self._ds, time_ids, entity_ids, sample_idx)
        if features is not None:
            keep = set(_as_list(features)) | set(self.text_cols) | set(self.targets)
            ds = ds[[c for c in ds.data_vars if c in keep]]
        ds = ds.copy()
        ds.attrs = dict(self._ds.attrs)
        kept = set(ds.data_vars)
        ds.attrs["targets"] = [t for t in self.targets if t in kept]
        ds.attrs["features"] = [f for f in self.features if f in kept]
        ds.attrs["pred_vars"] = [p for p in self.pred_vars if p in kept]
        ds.attrs["text_cols"] = [c for c in self.text_cols if c in kept]
        ds.attrs["sample_size"] = int(ds.sizes.get("sample", 1))
        return type(self)(ds)

    def split_data(
        self,
        time_ids: Any = None,
        features: Any = None,
        sample_idx: Any = None,
        entity_ids: Any = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Return lazy ``(X, y)`` feature and target DataArrays."""
        if self.is_prediction:
            raise ValueError("Data splitting is not applicable to prediction datasets")
        feature_names = _as_list(features) if features is not None else self.features
        x = self._stack_variables(feature_names)
        y = self._stack_variables(self.targets)
        x = self._apply_selection(x, time_ids, entity_ids, sample_idx)
        y = self._apply_selection(y, time_ids, entity_ids, sample_idx)
        return x, y

    def _apply_selection(
        self, obj: Any, time_ids: Any, entity_ids: Any, sample_idx: Any
    ) -> Any:
        if time_ids is not None:
            obj = obj.sel({self._time_id: _as_list(time_ids)})
        if entity_ids is not None:
            obj = obj.sel({self._entity_id: _as_list(entity_ids)})
        if sample_idx is not None:
            obj = obj.isel(sample=_as_list(sample_idx))
        return obj

    def check_integrity(
        self,
        include_targets: bool = True,
        time_ids: Any = None,
        features: Any = None,
        sample_idx: Any = None,
        entity_ids: Any = None,
    ) -> bool:
        """Verify the tensor round-trips against the stored variables."""
        tensor = self.get_subset_tensor(time_ids, features, sample_idx, entity_ids)
        for name in tensor["variable"].values:
            stored = self._var_as_3d(str(name))
            reference = self._apply_selection(stored, time_ids, entity_ids, sample_idx)
            rebuilt = tensor.sel(variable=name)
            if not np.allclose(
                reference.values, rebuilt.values, equal_nan=True
            ):
                return False
        return True

    # ---- conversions --------------------------------------------------------
    def to_predictionframe(self) -> Any:
        """Convert to a ``views_frames.PredictionFrame`` (prediction mode only).

        Uses ``to_tensor()`` (a single lazy dask operation) and reshapes
        the result — one compute call instead of T separate disk reads.
        """
        if not self.is_prediction:
            raise ValueError("to_predictionframe requires prediction mode")
        if len(self.targets) != 1:
            raise ValueError(
                f"PredictionFrame needs exactly one target, got {self.targets}"
            )
        from views_frames import PredictionFrame, FrameMetadata

        tensor = self.to_tensor()  # (T, E, S, 1) lazy
        computed = tensor.compute()
        t, e, s, _ = computed.shape
        y_pred = np.ascontiguousarray(computed.values.reshape(t * e, s))

        index = self._build_index()
        meta = FrameMetadata.from_dict(self.metadata) if self.metadata else None
        return PredictionFrame(y_pred.astype(np.float32), index, metadata=meta)

    def to_featureframe(self) -> Any:
        """Convert to a ``views_frames.FeatureFrame`` (feature mode only).

        Uses ``to_tensor()`` (a single lazy dask operation) and reshapes
        the result — one compute call instead of T×F separate disk reads.
        """
        if self.is_prediction:
            raise ValueError("to_featureframe requires feature mode")
        from views_frames import FeatureFrame, FrameMetadata

        names = list(self.features) + list(self.targets)
        if not names:
            raise ValueError("No feature or target variables to convert")

        tensor = self._stack_variables(names)  # (T, E, S, F) lazy
        computed = tensor.compute()
        t, e, s, f = computed.shape
        values = np.ascontiguousarray(
            computed.values.transpose(0, 1, 3, 2).reshape(t * e, f, s)
        )

        index = self._build_index()
        meta = FrameMetadata.from_dict(self.metadata) if self.metadata else None
        return FeatureFrame(values.astype(np.float32), index, names, metadata=meta)

    def _dense_values_and_index(self, name: str) -> tuple[np.ndarray, Any]:
        var = self._var_as_3d(name).transpose(self._time_id, self._entity_id, "sample")
        t, e, s = var.shape
        values = var.values.reshape(t * e, s)
        return values, self._build_index()

    def _build_index(self) -> Any:
        from views_frames import SpatialLevel, SpatioTemporalIndex

        level = SpatialLevel[_ENTITY_LEVEL[self._entity_id]]
        times = self._ds[self._time_id].values.astype("int64")
        entities = self._ds[self._entity_id].values.astype("int64")

        t_grid, e_grid = np.meshgrid(times, entities, indexing="ij")
        return SpatioTemporalIndex(
            time=t_grid.ravel(),
            unit=e_grid.ravel(),
            level=level,
        )

    # ---- persistence --------------------------------------------------------
    def save_parquet(self, path: str | Path) -> Path:
        """Save as list-in-cell Parquet via pyarrow (streamed by time slice)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = Path(path)
        writer: pq.ParquetWriter | None = None
        try:
            for time_value in self._ds[self._time_id].values:
                table = self._time_slice_table(int(time_value), pa)
                if writer is None:
                    writer = pq.ParquetWriter(str(path), table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        return path

    def _time_slice_table(self, time_value: int, pa: Any) -> Any:
        slice_ds = self._ds.sel({self._time_id: time_value})
        entities = self._ds[self._entity_id].values
        arrays = {
            self._time_id: pa.array(np.full(len(entities), time_value, dtype="int64")),
            self._entity_id: pa.array(entities.astype("int64")),
        }
        for name in self._ds.data_vars:
            data = slice_ds[name].values
            if "sample" in slice_ds[name].dims:
                arrays[name] = pa.array(list(data))
            else:
                arrays[name] = pa.array(data)
        return pa.table(arrays)

    def save_zarr(self, path: str | Path) -> Path:
        """Save as a consolidated Zarr directory."""
        path = Path(path)
        self._ds.to_zarr(path, mode="w", consolidated=True)
        return path

    def save_zarrzip(self, path: str | Path) -> Path:
        """Save as a Zarr zip file readable by ``zarr.storage.ZipStore``."""
        import tempfile
        import zipfile

        path = Path(path)
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp) / "store.zarr"
            self._ds.to_zarr(store_dir, mode="w", consolidated=False)
            with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as zf:
                for file in sorted(store_dir.rglob("*")):
                    if file.is_file():
                        zf.write(file, arcname=str(file.relative_to(store_dir)))
        return path

    def save_npz(self, path: str | Path) -> Path:
        """Save in the views-frames leaf format (values.npy + identifiers.npz)."""
        path = Path(path)
        frame = self.to_predictionframe() if self.is_prediction else self.to_featureframe()
        frame.save(path)
        return path

    # ---- cloud delivery (no pandas) ---------------------------------------
    def _detect_loa(self) -> str:
        """Return the spatial-temporal LOA code (e.g. ``"pgm"`` / ``"cm"``).

        Mirrors the legacy ``ForecastAccessor`` LOA codes — ``spatial_loa``
        (one of ``c``/``pg``/``a``) concatenated with ``temporal_loa`` (one
        of ``m``/``y``). The managers already carry this as
        ``self.configs["level"]``; this method lets a dataset self-report
        so :meth:`save_appwrite` does not need it as an argument.
        """
        from views_pipeline_core.modules.predstore import (
            detect_spatial_loa,
            detect_temporal_loa,
        )
        return f"{detect_spatial_loa(self._entity_id)}{detect_temporal_loa(self._time_id)}"

    def _predstore_metadata(self, model_name: str | None = None) -> dict:
        """Build the ``additional_info`` dict the predstore module expects.

        Mirrors the legacy ``ForecastAccessor`` autodetection: spatial/temporal
        LOA, time and space extents, prediction columns and steps, target.
        """
        info: dict[str, Any] = {}
        if model_name:
            info["description"] = model_name

        times = self._ds[self._time_id].values.astype("int64")
        entities = self._ds[self._entity_id].values.astype("int64")
        if times.size:
            info.setdefault("time_min", int(times.min()))
            info.setdefault("time_max", int(times.max()))
        if entities.size:
            info.setdefault("space_min", int(entities.min()))
            info.setdefault("space_max", int(entities.max()))

        pred_cols = [str(c) for c in self._ds.data_vars if str(c).startswith("pred_")]
        if pred_cols:
            info.setdefault("prediction_columns", sorted(pred_cols))
            if self.targets:
                info.setdefault("target", self.targets[0])
            else:
                info.setdefault("target", pred_cols[0])
            sample_size = int(self._ds.sizes.get("sample", 1))
            info.setdefault("steps", list(range(1, sample_size + 1)))

        if self.is_prediction:
            info.setdefault("ds", True)
            info.setdefault("osa", False)
        return info

    def save_predstore(
        self,
        name: str,
        run: str | int,
        *,
        module: Any = None,
    ) -> str:
        """Upload this dataset's parquet to the views-forecasts Azure store.

        Minimal signature: only ``name`` (the prediction name the legacy
        ``to_store`` received) and ``run`` (the run name, e.g.
        ``self._pred_store_name``) are required. Everything else the
        predstore module needs (LOA, extents, prediction columns, steps,
        target) is autodetected from this dataset's schema.

        Args:
            name: Logical prediction name. The blob key is
                ``pr_{run}_{name}.parquet``.
            run: Run name (str, e.g. ``"v010200_2026_03"``) or run id (int).
            module: Optional pre-built :class:`PredstoreModule` (e.g. an
                existing instance shared across calls, or a mock in tests).
                When ``None``, a module is built from the environment and
                closed afterwards.

        Returns:
            The blob key (``pr_{run}_{name}.parquet``).
        """
        from views_pipeline_core.modules.predstore import (
            PredstoreConfig, PredstoreModule,
        )

        info = self._predstore_metadata(model_name=None)
        owns_module = module is None
        if module is None:
            module = PredstoreModule(PredstoreConfig.from_environment())
        try:
            return module.save_dataset(
                self,
                name=name,
                run=run,
                overwrite=True,
                additional_info=info,
                check_transfer=False,
            )
        finally:
            if owns_module:
                module.close()

    def save_appwrite(
        self,
        filename: str,
        model_name: str,
        target: str,
        loa: str,
        *,
        datastore: Any = None,
    ) -> Any:
        """Upload this dataset's parquet to the Appwrite cloud store.

        Minimal signature matching the legacy ``AppwriteSaver.save`` /
        ``DatastoreModule.upload_data`` call: filename in the bucket, model
        name to record in metadata, target variable name. The LOA, targets
        list and category are derived from this dataset so callers do not
        repeat them.

        Args:
            filename: File name in the Appwrite bucket (e.g.
                ``"predictions_forecasting_20260101.parquet"``).
            model_name: Model name recorded in the Appwrite metadata
                document.
            target: Target variable name. Recorded both as ``type`` in the
                metadata and in the ``targets`` list, matching the legacy
                saver chain.
            datastore: Optional pre-built :class:`DatastoreModule`. When
                ``None``, a datastore is built from the standard Appwrite
                environment variables. Pass an existing instance to amortise
                the authentication round-trip across multiple uploads.

        Returns:
            The :class:`OperationResult` returned by
            ``DatastoreModule.upload_data``. Inspect ``result.success``
            before declaring victory — Appwrite reports failure by RETURN
            VALUE, not by exception (register C-227).

        Note:
            Appwrite is the SECONDARY EXTERNAL destination under ADR-047.
            A failure is logged at ``logger.error`` but does NOT raise —
            local disk and views-forecasts already hold the authoritative
            artefacts. This matches :class:`AppwriteSaver`'s
            graceful-degradation contract.
        """
        import tempfile

        # loa = self._detect_loa()
        targets = list(self.targets) if self.targets else [target]

        owns_datastore = datastore is None
        if datastore is None:
            from views_pipeline_core.modules.datastore import DatastoreModule
            from views_pipeline_core.configs.prediction_store import (
                PredictionStoreConfig,
            )
            datastore = DatastoreModule(
                PredictionStoreConfig.from_environment().to_appwrite_config(
                    path_manager=None,
                )
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / filename
            self.save_parquet(parquet_path)
            try:
                result = datastore.upload_data(
                    file=parquet_path,
                    filename=filename,
                    loa=loa,
                    name=model_name,
                    type=target,
                    targets=targets,
                    category="forecast",
                    description="",
                )
            except Exception as e:
                # Programming errors propagate; transport faults have
                # already been converted to OperationResult by the SDK
                # glue — same pattern as AppwriteSaver.save().
                logger.error(
                    "save_appwrite: upload FAILED for %s — %s",
                    filename, e, exc_info=True,
                )
                return None

        if result is None or getattr(result, "success", False):
            logger.info(
                "save_appwrite: uploaded %s (loa=%s, target=%s).",
                filename, loa, target,
            )
        else:
            logger.error(
                "save_appwrite: upload FAILED for %s — code=%s error=%s",
                filename,
                getattr(result, "code", None),
                getattr(result, "error", None),
            )
        return result

    # ---- cloud retrieval -------------------------------------------------
    @classmethod
    def from_predstore_latest(
        cls,
        model_path: Any,
        run: str | int | None = None,
        target: str | None = None,
        *,
        module: Any = None,
    ) -> "ViewsDataset":
        """Retrieve the latest prediction for a model from the views-forecasts store.

        Mirrors the lookup the EnsembleManager does inline today:

            run_id = ViewsMetadata().get_run_id_from_name(self._pred_store_name)
            all_runs = ViewsMetadata().with_name(cm_model).fetch()["name"].to_list()
            forecasts = [fc for fc in all_runs if cm_model in fc and "forecasting" in fc]
            forecasts.sort()
            return _CDataset(source=pd.DataFrame.forecasts.read_store(
                run=run_id, name=forecasts[-1]))

        The new path reads the parquet bytes back through
        :class:`PredstoreModule` and constructs the dataset directly — no
        pandas. ``run`` is resolved from ``model_path`` when not supplied
        (using the same ``_pred_store_name`` format the manager builds).

        Args:
            model_path: A :class:`ModelPathManager` (or any object exposing
                ``model_name`` and, optionally, a ``_pred_store_name``
                attribute). Used both to resolve ``run`` (when not supplied)
                and to filter the metadata lookup.
            run: Run name (str) or id (int). When ``None``, the manager's
                ``_pred_store_name`` attribute on ``model_path`` is used as
                a fallback; if that is missing, ``LookupError`` is raised.
            target: Optional target name. Currently informational — the
                legacy store keys files by ``(run, name)`` only.
            module: Optional pre-built :class:`PredstoreModule`.

        Returns:
            A :class:`ViewsDataset` (or subclass — call
            ``ViewsDataset.for_loa`` to route to ``PGDataset`` /
            ``CDataset`` based on the retrieved data) loaded from the
            latest matching parquet blob.

        Raises:
            LookupError: If no matching prediction is found in the store.
        """
        import tempfile

        from views_pipeline_core.modules.predstore import (
            PredstoreConfig, PredstoreModule,
        )

        # Resolve the run name: explicit > manager's _pred_store_name > error.
        # The ForecastingModelManager / EnsembleManager set this attribute on
        # themselves, not on model_path — but a caller who built model_path
        # independently and wants to read against a known run can pass it
        # directly. Refusing the implicit "test" default matches the legacy
        # ``ForecastsStore`` "test" default being unsafe (register C-229).
        if run is None:
            run = getattr(model_path, "_pred_store_name", None)
        if run is None:
            raise LookupError(
                "from_predstore_latest: 'run' was not supplied and "
                "model_path has no '_pred_store_name' attribute. Pass run= "
                "explicitly (e.g. 'v010200_2026_03')."
            )

        # Default name to the legacy ensemble pattern when no metadata DB
        # is configured — matches what the EnsembleManager writes via
        # ``{model_name}_predictions_{run_type}_{ts}``. Callers that need
        # a different name can pre-build the module with a metadata writer
        # and let it look up the latest matching name itself.
        name = f"{model_path.model_name}_predictions_forecasting"

        owns_module = module is None
        if module is None:
            module = PredstoreModule(PredstoreConfig.from_environment())
        try:
            parquet_bytes = module.read(name=name, run=run)
        finally:
            if owns_module:
                module.close()

        # The dataset's parquet converter takes a path (see readers.detect_source_type).
        # Write the bytes to a temp file and hand the path over.
        with tempfile.NamedTemporaryFile(
            suffix=".parquet", delete=False
        ) as tmp:
            tmp.write(parquet_bytes)
            tmp_path = Path(tmp.name)
        return cls(tmp_path)

    @classmethod
    def from_appwrite_latest(
        cls,
        model_path: Any,
        target: str | None = None,
        *,
        datastore: Any = None,
    ) -> "ViewsDataset":
        """Retrieve the latest prediction for a model from the Appwrite store.

        Mirrors the legacy ``DatastoreModule.download_latest_file`` flow:
        search the metadata collection by model name (and optionally
        target), pick the most recent match, download its parquet, and
        construct a dataset from it.

        Args:
            model_path: A :class:`ModelPathManager`. ``model_name`` is the
                primary filter; ``target`` (when set) disambiguates.
            target: Optional target name to filter on. When ``None``, the
                most recent file for the model is returned regardless of
                target.
            datastore: Optional pre-built :class:`DatastoreModule`.

        Returns:
            A :class:`ViewsDataset` (or subclass) loaded from the latest
            matching parquet.

        Raises:
            LookupError: If no matching file is found.
        """
        import tempfile

        owns_datastore = datastore is None
        if datastore is None:
            from views_pipeline_core.modules.datastore import DatastoreModule
            from views_pipeline_core.configs.prediction_store import (
                PredictionStoreConfig,
            )
            datastore = DatastoreModule(
                PredictionStoreConfig.from_environment().to_appwrite_config(
                    path_manager=None,
                )
            )

        filters = {"name": model_path.model_name, "category": "forecast"}
        if target:
            filters["targets"] = target

        file_id = datastore.get_latest_file_id(filters=filters)
        if file_id is None:
            raise LookupError(
                f"from_appwrite_latest: no file found for model "
                f"{model_path.model_name!r} (target={target!r})."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / f"{model_path.model_name}.parquet"
            result = datastore.download_prediction(
                file_id=file_id,
                save_path=save_path,
                use_cache=False,
            )
            if not getattr(result, "success", False):
                raise LookupError(
                    f"from_appwrite_latest: download failed for file_id={file_id}: "
                    f"{getattr(result, 'error', 'unknown')}"
                )
            return cls(save_path)

    @classmethod
    def for_loa(cls, loa: str, source: Any, **kwargs: Any) -> "ViewsDataset":
        """Construct the right dataset subclass for a VIEWS ``loa`` code.

        The managers carry ``self.configs["level"]`` (``"pgm"`` / ``"cm"`` /
        ``"pgy"`` / ``"cym"``). Routing the source through the matching
        subclass enforces the index-name invariant (``priogrid_id`` vs.
        ``country_id`` / ``month_id`` vs. ``year_id``) at construction.

        ``loa`` is the legacy two-letter code (spatial + temporal), case
        insensitive. Unknown codes fall back to the base
        :class:`ViewsDataset` — same behaviour as the legacy
        ``_ViewsDataset(source=...)`` path.
        """
        from views_r2darts2.dataset.subclasses import (
            CDataset, CMDataset, CYDataset,
            PGDataset, PGMDataset, PGYDataset,
        )
        _LOA_TO_CLASS = {
            "pgm": PGMDataset, "pgy": PGYDataset, "pg": PGDataset,
            "cm":  CMDataset,  "cy":  CYDataset,  "c":  CDataset,
        }
        klass = _LOA_TO_CLASS.get((loa or "").lower(), ViewsDataset)
        return klass(source, **kwargs)

    # ---- xarray access -----------------------------------------------------
    def to_xarray(self) -> xr.Dataset:
        """Return the underlying lazy xarray.Dataset (Dask-backed)."""
        return self._ds

    # ---- validation + introspection ----------------------------------------
    def __enter__(self) -> "ViewsDataset":
        return self

    def __exit__(self, *exc: object) -> None:
        self._store.close()

    def close(self) -> None:
        """Close the underlying Zarr store and clean up temp files."""
        self._store.close()

    def __del__(self) -> None:
        if getattr(self, "_store", None) is not None:
            self._store.close()

    def validate_indices(self) -> None:
        """Ensure the store carries exactly the time and entity dimensions."""
        for dim in (self._time_id, self._entity_id):
            if dim not in self._ds.dims:
                raise ValueError(f"Dataset is missing required dimension '{dim}'")

    @property
    def num_entities(self) -> int:
        return int(self._ds.sizes[self._entity_id])

    @property
    def num_time_steps(self) -> int:
        return int(self._ds.sizes[self._time_id])

    @property
    def num_features(self) -> int:
        return len(self.features)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(time_steps={self.num_time_steps}, "
            f"entities={self.num_entities}, features={self.num_features}, "
            f"prediction_mode={self.is_prediction})"
        )

    # ------------------------------------------------------------------ #
    # Empty creation + incremental concatenation
    # ------------------------------------------------------------------ #

    @staticmethod
    def create_empty(
        level: str,
        *,
        features: list[str] | None = None,
        targets: list[str] | None = None,
        sample_size: int = 1,
        time_id: str = "month_id",
        entity_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ViewsDataset":
        """Create an empty, zarr-backed dataset for a given spatial level.

        Useful when there are no predictions yet to initialize the dataset —
        call :meth:`add_row` / :meth:`add_batch` to populate it incrementally.

        Args:
            level: VIEWS LOA code (``"cm"``, ``"pgm"``, ``"cy"``, ``"pgy"``).
                Determines the default ``entity_id`` and ``time_id`` when not
                explicitly provided.
            features: Feature column names (empty list if ``None``).
            targets: Target column names (empty list if ``None``).
            sample_size: Number of samples per (time, entity) cell. Defaults
                to 1 (deterministic). Set to ``N`` for probabilistic forecasts.
            time_id: Time dimension name (default ``"month_id"``; use
                ``"year_id"`` for yearly data).
            entity_id: Entity dimension name. When ``None``, inferred from
                ``level`` (``"country_id"`` for cm/cy, ``"priogrid_id"`` for
                pgm/pgy).
            metadata: Optional user-provided attrs dict.

        Returns:
            An empty :class:`ViewsDataset` (or the matching subclass via
            :meth:`for_loa`) with zero rows, ready for :meth:`add_row` /
            :meth:`add_batch` calls.
        """
        level_lower = (level or "").lower()
        if entity_id is None:
            entity_id = "priogrid_id" if level_lower.startswith("pg") else "country_id"
        if time_id is None:
            time_id = "year_id" if level_lower.endswith("y") else "month_id"

        features = list(features) if features else []
        targets = list(targets) if targets else []
        all_vars = [*features, *targets]

        # Build an empty xarray.Dataset with the canonical schema.
        coords = {
            time_id: np.array([], dtype="int64"),
            entity_id: np.array([], dtype="int64"),
            "sample": np.arange(sample_size, dtype="int64"),
        }
        data_vars: dict[str, Any] = {}
        for name in all_vars:
            data_vars[name] = (
                (time_id, entity_id, "sample"),
                np.empty((0, 0, sample_size), dtype=np.float32),
            )
        ds = xr.Dataset(data_vars, coords=coords)
        ds.attrs.update({
            "is_prediction": False,
            "sample_size": int(sample_size),
            "targets": targets,
            "features": features,
            "pred_vars": [],
            "text_cols": [],
            "time_id": time_id,
            "entity_id": entity_id,
            "broadcast_features": False,
            **(metadata or {}),
        })

        # Route through for_loa so the right subclass is returned.
        return ViewsDataset.for_loa(level_lower, ds)

    def add_row(
        self,
        time: int,
        entity: int,
        values: dict[str, float | list[float] | np.ndarray],
    ) -> None:
        """Append a single ``(time, entity)`` row to the dataset.

        The row is written into the underlying zarr store via coordinate
        selection. If the ``(time, entity)`` pair already exists, its values
        are overwritten; otherwise the store is extended.

        Args:
            time: Time id (e.g. month_id).
            entity: Entity id (e.g. country_id / priogrid_id).
            values: ``{column_name: value}`` mapping. Scalar values are
                broadcast to ``sample_size``; list/array values must have
                length ``sample_size``.
        """
        times = np.array([time], dtype="int64")
        entities = np.array([entity], dtype="int64")
        cols: dict[str, np.ndarray] = {}
        for name, val in values.items():
            arr = np.asarray(val, dtype=np.float32)
            if arr.ndim == 0:
                arr = np.full(self.sample_size, float(arr), dtype=np.float32)
            cols[name] = arr.reshape(1, -1)
        self._write_rows(times, entities, cols)

    def add_batch(
        self,
        times: np.ndarray,
        entities: np.ndarray,
        values: dict[str, np.ndarray],
    ) -> None:
        """Append a batch of rows to the dataset.

        Args:
            times: 1-D int64 array of time ids (length ``N``).
            entities: 1-D int64 array of entity ids (length ``N``).
            values: ``{column_name: array}`` mapping. Each array has shape
                ``(N,)`` for scalar columns or ``(N, sample_size)`` for
                probabilistic columns.
        """
        times = np.asarray(times, dtype="int64")
        entities = np.asarray(entities, dtype="int64")
        cols: dict[str, np.ndarray] = {}
        for name, arr in values.items():
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 1:
                arr = np.broadcast_to(arr[:, None], (arr.shape[0], self.sample_size))
            cols[name] = arr
        self._write_rows(times, entities, cols)

    def _write_rows(
        self,
        times: np.ndarray,
        entities: np.ndarray,
        cols: dict[str, np.ndarray],
    ) -> None:
        """Scatter-write rows into the zarr store, extending coordinates as needed.

        When a time or entity id is not yet in the store, the zarr array is
        resized to accommodate it. This is the incremental-concatenation
        primitive that :meth:`add_row` and :meth:`add_batch` delegate to.
        """
        import zarr

        store_path = str(self._store.path / self._ds.attrs.get("_store_name", "dataset.zarr"))
        # Find the actual zarr path by re-opening the dataset and checking
        # its encoding. The simplest approach: re-open the store directory
        # and write directly.
        # We need to find where the zarr group lives. The _ds was opened from
        # a path; we can get it from the encoding.
        zarr_path = None
        for var in self._ds.data_vars:
            enc = self._ds[var].encoding
            if "source" in enc:
                zarr_path = enc["source"]
                # The source is like "/tmp/.../dataset_xxxx.zarr/<var>"
                zarr_path = str(Path(zarr_path).parent)
                break
        if zarr_path is None:
            # Fallback: re-save the full dataset (simple but not incremental).
            self._append_fallback(times, entities, cols)
            return

        group = zarr.open_group(zarr_path, mode="a")
        time_coord = self._ds[self._time_id].values.astype("int64")
        entity_coord = self._ds[self._entity_id].values.astype("int64")

        # Determine new times and entities that need to be added.
        new_times = np.setdiff1d(times, time_coord)
        new_entities = np.setdiff1d(entities, entity_coord)

        if new_times.size or new_entities.size:
            # Resize: rebuild the store with extended coordinates.
            # This is the simplest correct approach for zarr v3.
            self._append_fallback(times, entities, cols)
            return

        # All (time, entity) pairs already exist — write via coordinate selection.
        tp = np.searchsorted(time_coord, times)
        ep = np.searchsorted(entity_coord, entities)
        s = self.sample_size
        for name, arr in cols.items():
            if name in group:
                zarr_arr = group[name]
                tr = np.repeat(tp, s)
                er = np.repeat(ep, s)
                sr = np.tile(np.arange(s), len(tp))
                zarr_arr.set_coordinate_selection(
                    (tr, er, sr), arr.reshape(-1).astype(np.float32)
                )

        # Refresh the in-memory dataset to pick up the writes.
        self._ds = readers.open_zarr_dir(Path(zarr_path))

    def _append_fallback(
        self,
        times: np.ndarray,
        entities: np.ndarray,
        cols: dict[str, np.ndarray],
    ) -> None:
        """Fallback: rebuild the store with the new rows concatenated.

        This materializes the full dataset in memory, concatenates the new
        rows, and re-writes the zarr store. Used when new time/entity
        coordinates need to be added (zarr v3 doesn't support easy resizing).
        """
        # Build the new rows as an xarray Dataset.
        time_coord = self._ds[self._time_id].values.astype("int64")
        entity_coord = self._ds[self._entity_id].values.astype("int64")
        all_times = np.union1d(time_coord, times)
        all_entities = np.union1d(entity_coord, entities)
        s = self.sample_size

        # Build a full grid of NaN, then fill existing + new values.
        new_ds = self._ds.reindex(
            {self._time_id: all_times, self._entity_id: all_entities},
        )
        # Write the new rows.
        tp = np.searchsorted(all_times, times)
        ep = np.searchsorted(all_entities, entities)
        for name, arr in cols.items():
            if name not in new_ds:
                new_ds[name] = (
                    (self._time_id, self._entity_id, "sample"),
                    np.full((len(all_times), len(all_entities), s), np.nan, dtype=np.float32),
                )
            # Copy to a writable array (xarray may return a read-only view).
            var = new_ds[name].values.copy()
            for i in range(len(times)):
                var[tp[i], ep[i], :] = arr[i]
            new_ds[name] = (
                (self._time_id, self._entity_id, "sample"),
                var,
            )

        # Re-write the zarr store.
        zarr_path = None
        for var in self._ds.data_vars:
            enc = self._ds[var].encoding
            if "source" in enc:
                zarr_path = str(Path(enc["source"]).parent)
                break
        if zarr_path is None:
            zarr_path = str(self._store.path / "dataset.zarr")

        new_ds.attrs = dict(self._ds.attrs)
        new_ds.to_zarr(zarr_path, mode="w", consolidated=False)
        self._ds = readers.open_zarr_dir(Path(zarr_path))

    # ------------------------------------------------------------------ #
    # Scaler integration
    # ------------------------------------------------------------------ #

    def fit_scalers(
        self,
        *,
        target_scaler: str | None = None,
        feature_scaler: str | None = None,
        feature_scaler_map: dict[str, Any] | None = None,
        log_targets: bool = False,
        log_features: list[str] | None = None,
        time_ids: Any = None,
        return_series: bool = False,
        use_cyclic_encoders: bool = False,
    ):
        """Fit scalers on the current dataset's data.

        Args:
            target_scaler: Scaler name for targets (e.g. ``"AsinhTransform"``).
            feature_scaler: Scaler name for all features.
            feature_scaler_map: Per-feature scaler map.
            log_targets: Apply ``log1p`` to targets before scaling.
            log_features: Feature names to apply ``log1p`` to.
            time_ids: Optional time-id filter (prevents test-period leakage).
            return_series: When ``True``, return the fitted+transformed
                ``(targets_ts, past_cov_ts)`` so callers avoid a second data
                load for the same time range.
            use_cyclic_encoders: Passed through to :meth:`to_darts_timeseries`
                when ``return_series=True``.
        """
        from views_r2darts2.transformers.feature_scaler_manager import (
            FeatureScalerManager,
        )
        from views_r2darts2.transformers.scaler_selector import ScalerSelector

        self._target_scaler_cfg = target_scaler
        self._feature_scaler_cfg = feature_scaler
        self._feature_scaler_map_cfg = (
            dict(feature_scaler_map) if feature_scaler_map else None
        )
        self._log_targets = bool(log_targets)
        self._log_features = set(log_features or [])
        self._scalers_fitted = False

        # Instantiate the target scaler.
        self._target_scaler = (
            ScalerSelector.instantiate_darts_scaler(target_scaler)
            if target_scaler is not None
            else None
        )

        # Instantiate the feature scaler(s).
        if not self.features:
            self._feature_scaler = None
        elif feature_scaler_map:
            self._feature_scaler = FeatureScalerManager(
                feature_scaler_map=feature_scaler_map,
                default_scaler=feature_scaler,
                all_features=self.features,
            )
        else:
            self._feature_scaler = ScalerSelector.instantiate_darts_scaler(
                feature_scaler
            ) if feature_scaler is not None else None

        # Build Darts TimeSeries from the training partition, fit the scalers.
        series_list = self.to_darts_timeseries(
            time_ids=time_ids,
            use_cyclic_encoders=use_cyclic_encoders if return_series else False,
        )
        targets_ts, past_cov_ts = self._split_targets_covariates(series_list)
        targets_ts = self._apply_log_to_targets(targets_ts)
        if past_cov_ts is not None:
            past_cov_ts = self._apply_log_to_features(past_cov_ts)

        if self._target_scaler is not None:
            targets_ts = self._target_scaler.fit_transform(targets_ts)
        if self._feature_scaler is not None:
            past_cov_ts = self._feature_scaler.fit_transform(past_cov_ts)
        self._scalers_fitted = True
        logger.info("Scalers fitted: target=%r, feature=%r",
                     target_scaler, feature_scaler or feature_scaler_map)
        if return_series:
            # Downcast to float32 — scalers like AsinhTransform return float64.
            targets_ts = [ts if ts.dtype == np.float32 else ts.astype(np.float32) for ts in targets_ts]
            if past_cov_ts is not None:
                past_cov_ts = [pc if pc.dtype == np.float32 else pc.astype(np.float32) for pc in past_cov_ts]
            return targets_ts, past_cov_ts
        return None

    @property
    def scalers_fitted(self) -> bool:
        """Whether :meth:`fit_scalers` has been called."""
        return getattr(self, "_scalers_fitted", False)

    def _split_targets_covariates(
        self, series_list: list
    ) -> tuple[list, list | None]:
        """Split a list of Darts TimeSeries into (targets, past_covariates)."""
        if not self.targets:
            return [], None
        if not self.features:
            # value_columns == targets only; series already contain only target
            # columns — skip the per-entity __getitem__ call (259k __init__ calls).
            return series_list, None
        targets = [ts[self.targets] for ts in series_list]
        past_cov = [ts[self.features].astype(np.float32) for ts in series_list]
        return targets, past_cov

    def _apply_log_to_targets(self, series_list: list) -> list:
        """Apply log1p to target series (clip negatives first)."""
        if not getattr(self, "_log_targets", False):
            return series_list
        return [
            ts.map(lambda arr: np.log1p(np.maximum(arr, 0)).astype(np.float32))
            for ts in series_list
        ]

    def _apply_log_to_features(self, series_list: list) -> list:
        """Apply log1p to selected feature components."""
        if not getattr(self, "_log_features", None):
            return series_list
        out = []
        for ts in series_list:
            comps = ts.components
            if not any(c in self._log_features for c in comps):
                out.append(ts)
                continue
            arr = ts.all_values(copy=True)
            for idx, name in enumerate(comps):
                if name in self._log_features:
                    if arr.ndim == 2:
                        arr[:, idx] = np.log1p(np.maximum(arr[:, idx], 0.0))
                    elif arr.ndim == 3:
                        arr[:, idx, :] = np.log1p(np.maximum(arr[:, idx, :], 0.0))
            out.append(ts.from_times_and_values(
                times=ts.time_index, values=arr.astype(np.float32),
                columns=comps, freq=ts.freq, static_covariates=ts.static_covariates,
            ))
        return out

    def get_scaled_darts_timeseries(
        self,
        *,
        time_ids: Any = None,
        entity_ids: Any = None,
        use_cyclic_encoders: bool = False,
    ) -> tuple[list, list | None]:
        """Return scaled Darts TimeSeries for the model.

        Delegates to :meth:`to_darts_timeseries` for the raw series, then
        applies log transforms + scaler transforms. Requires
        :meth:`fit_scalers` to have been called.

        Args:
            time_ids: Optional time-id filter.
            entity_ids: Optional entity-id filter.
            use_cyclic_encoders: Append sin/cos cyclic time encoders.

        Returns:
            ``(targets, past_covariates)`` tuple. ``past_covariates`` is
            ``None`` when the dataset has no features.
        """
        if not self.scalers_fitted:
            raise RuntimeError("Scalers not fitted. Call fit_scalers first.")
        series_list = self.to_darts_timeseries(
            time_ids=time_ids, entity_ids=entity_ids,
            use_cyclic_encoders=use_cyclic_encoders,
        )
        targets, past_cov = self._split_targets_covariates(series_list)
        targets = self._apply_log_to_targets(targets)
        if past_cov is not None:
            past_cov = self._apply_log_to_features(past_cov)
        if self._target_scaler is not None:
            targets = self._target_scaler.transform(targets)
        if self._feature_scaler is not None and past_cov is not None:
            past_cov = self._feature_scaler.transform(past_cov)
        # Downcast to float32.
        targets = [ts.astype(np.float32) for ts in targets]
        if past_cov is not None:
            past_cov = [pc.astype(np.float32) for pc in past_cov]
        return targets, past_cov

    def ingest_darts_predictions(
        self,
        predictions: list,
        *,
        apply_inverse: bool = True,
        clip_negatives: bool = True,
    ) -> dict[str, Any]:
        """Ingest Darts prediction TimeSeries into this dataset.

        Applies inverse transforms (target scaler inverse + log inverse) when
        ``apply_inverse=True``, then stores the predictions as new rows in the
        zarr-backed store via :meth:`add_batch`.

        Args:
            predictions: List of Darts :class:`TimeSeries` (one per entity).
            apply_inverse: When ``True``, apply the inverse target scaler and
                inverse log transform. Requires :meth:`fit_scalers` to have
                been called.
            clip_negatives: When ``True``, clip negative predictions to 0.

        Returns:
            A ``{target_name: PredictionFrame}`` dict (one frame per target),
            built via :meth:`to_predictionframe` after the predictions are
            stored.
        """
        from views_r2darts2.transformers.inverse import (
            inverse_transform_probabilistic_subset,
        )
        from views_r2darts2.transformers.darts_bridge import prediction_frames_from_darts

        # Apply inverse transforms.
        if apply_inverse and self.scalers_fitted:
            predictions = self._inverse_transform_predictions(predictions)

        # Build PredictionFrames from the Darts predictions.
        frames = prediction_frames_from_darts(
            predictions=predictions,
            entity_id_name=self._entity_id,
            target_columns=self.targets,
            level=self._build_spatial_level(),
            clip_negatives=clip_negatives,
        )
        return frames

    def ingest_numpy_predictions(
        self,
        predictions: np.ndarray,
        entity_ids: np.ndarray,
        time_ids: np.ndarray,
        *,
        apply_inverse: bool = True,
        clip_negatives: bool = True,
    ) -> dict[str, Any]:
        """Ingest raw numpy predictions (bypassing Darts TimeSeries entirely).

        This is the memory-efficient prediction path for large entity counts
        (e.g. 259k PRIO-GRID cells). The standard
        :meth:`ingest_darts_predictions` builds a Darts :class:`TimeSeries`
        per entity — 259k Python objects + 259k pandas DataFrames — which
        causes OOM kills. This method works with raw numpy arrays throughout:

            1. Apply inverse scaler transforms directly on the numpy array
               (vectorized — no per-entity Python loop).
            2. Build :class:`PredictionFrame` objects from the numpy array +
               the entity/time index arrays.

        Args:
            predictions: Shape ``(n_entities, n_time, n_components, n_samples)``
                float32. This is the raw output of Darts'
                ``predict_from_dataset(values_only=True)``.
            entity_ids: 1-D int64 array of entity ids (length ``n_entities``).
            time_ids: 1-D int64 array of time ids (length ``n_time``).
            apply_inverse: When ``True``, apply the inverse target scaler and
                inverse log transform (numpy-direct).
            clip_negatives: When ``True``, clip negative predictions to 0.

        Returns:
            A ``{target_name: PredictionFrame}`` dict.
        """
        from views_r2darts2.transformers.inverse import (
            extract_fitted_sklearn_scaler,
        )
        from views_frames import (
            PredictionFrame, SpatioTemporalIndex, SpatialLevel, FrameMetadata,
        )

        # predictions shape: (n_entities, n_time, n_components, n_samples)
        n_entities, n_time, n_components, n_samples = predictions.shape
        target_names = list(self.targets)
        n_targets = len(target_names)

        # Determine which component indices are targets.
        # Darts returns all components in the order they were in the input series.
        # We need to figure out the target component indices.
        # For now, assume the last n_targets components are the targets (the
        # standard layout when features come before targets).
        # TODO: make this robust by passing component names from the forecaster.
        target_indices = list(range(n_components - n_targets, n_components))

        # Extract target values: (n_entities, n_time, n_targets, n_samples)
        target_values = predictions[:, :, target_indices, :]

        # Apply inverse transforms (numpy-direct, vectorized).
        if apply_inverse and self.scalers_fitted:
            target_values = self._inverse_transform_numpy_predictions(
                target_values
            )

        # Clip negatives.
        if clip_negatives:
            np.maximum(target_values, 0.0, out=target_values)

        # Build PredictionFrames — one per target.
        # Build the SpatioTemporalIndex: (entity, time) grid, entity-major.
        # time_ids has shape (n_time,), entity_ids has shape (n_entities,)
        t_grid, e_grid = np.meshgrid(time_ids, entity_ids, indexing="ij")
        # t_grid: (n_time, n_entities), e_grid: (n_time, n_entities)
        # We want row-major (time, entity) — entity varies fastest.
        time_flat = t_grid.T.ravel()  # (n_entities * n_time,) — entity-major
        entity_flat = e_grid.T.ravel()
        # Actually, we want time-major to match the existing convention.
        # meshgrid with indexing="ij" gives (time, entity) — reshape row-major.
        time_flat = t_grid.ravel()  # time varies slowest
        entity_flat = e_grid.ravel()

        index = SpatioTemporalIndex(
            time=time_flat.astype(np.int64),
            unit=entity_flat.astype(np.int64),
            level=self._build_spatial_level(),
        )
        metadata = FrameMetadata(model="darts")

        frames: dict[str, Any] = {}
        for t_idx, target_name in enumerate(target_names):
            # target_values: (n_entities, n_time, n_targets, n_samples)
            # Extract this target: (n_entities, n_time, n_samples)
            # Reshape to (n_entities * n_time, n_samples) — entity-major.
            # But the index is time-major (time, entity).
            # We need to transpose to (n_time, n_entities, n_samples) then flatten.
            vals = target_values[:, :, t_idx, :]  # (n_entities, n_time, n_samples)
            # Transpose to (n_time, n_entities, n_samples) to match time-major index.
            vals = vals.transpose(1, 0, 2)  # (n_time, n_entities, n_samples)
            vals = vals.reshape(n_time * n_entities, n_samples).astype(np.float32)
            frames[target_name] = PredictionFrame(
                vals, index=index, metadata=metadata,
            )
        return frames

    # ------------------------------------------------------------------ #
    # Streaming prediction path (zarr-backed scaffold)
    # ------------------------------------------------------------------ #
    #
    # The streaming path exists to keep peak memory bounded when the
    # forecast is probabilistic (``num_samples`` large) and/or the entity
    # count is huge (e.g. 259k PRIO-GRID cells). Instead of materialising
    # the full ``(n_entities, n_time, n_targets, n_samples)`` predictions
    # array in RAM, we:
    #
    #   1. Create a zarr-backed prediction scaffold sized for the full
    #      ``(n_time, n_entities, n_samples)`` grid (``create_prediction_scaffold``).
    #   2. Run the model in entity batches and write each batch directly
    #      to the scaffold (``write_prediction_batch``), applying the same
    #      inverse-transform + clip path as ``ingest_numpy_predictions``.
    #   3. Convert the populated scaffold to ``{target: PredictionFrame}``
    #      (``to_predictionframe_per_target``).
    #
    # Parity: ``write_prediction_batch`` reuses
    # ``_inverse_transform_numpy_predictions`` and builds the same
    # ``(time, entity, sample)`` layout as ``ingest_numpy_predictions``,
    # so the streaming and in-memory paths produce bit-identical frames.

    @staticmethod
    def create_prediction_scaffold(
        *,
        entity_ids: np.ndarray,
        time_ids: np.ndarray,
        targets: list[str],
        sample_size: int,
        level: str,
        time_id: str = "month_id",
        entity_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ViewsDataset":
        """Create a disk-backed (zarr) prediction scaffold for the full grid.

        The scaffold is a prediction-mode :class:`ViewsDataset` with one
        ``pred_<target>`` variable per target, pre-allocated to shape
        ``(n_time, n_entities, sample_size)`` and filled with NaN. Rows are
        populated incrementally via :meth:`write_prediction_batch`, which
        writes directly to the zarr store on disk — the full predictions
        array is never held in memory at once.

        Args:
            entity_ids: 1-D int64 array of entity ids (the full set).
            time_ids: 1-D int64 array of time ids (the full set).
            targets: Target column names (one ``pred_<target>`` variable is
                created per target).
            sample_size: Number of probabilistic samples per cell.
            level: VIEWS LOA code (``"cm"``, ``"pgm"``).
            time_id: Time dimension name (default ``"month_id"``).
            entity_id: Entity dimension name. Inferred from ``level`` when
                ``None``.
            metadata: Optional attrs dict (model name, run_type, etc.).

        Returns:
            An empty prediction-mode :class:`ViewsDataset` ready for
            :meth:`write_prediction_batch` calls.
        """
        level_lower = (level or "").lower()
        if entity_id is None:
            entity_id = "priogrid_id" if level_lower.startswith("pg") else "country_id"

        times = np.unique(np.asarray(time_ids, dtype=np.int64))
        entities = np.unique(np.asarray(entity_ids, dtype=np.int64))
        s = int(sample_size)
        pred_vars = [f"pred_{t}" for t in targets]

        coords = {
            time_id: times,
            entity_id: entities,
            "sample": np.arange(s, dtype=np.int64),
        }
        data_vars: dict[str, Any] = {}
        for name in pred_vars:
            data_vars[name] = (
                (time_id, entity_id, "sample"),
                np.full((len(times), len(entities), s), np.nan, dtype=np.float32),
            )
        ds = xr.Dataset(data_vars, coords=coords)
        ds.attrs.update({
            "is_prediction": True,
            "sample_size": s,
            "targets": list(targets),
            "features": [],
            "pred_vars": pred_vars,
            "text_cols": [],
            "time_id": time_id,
            "entity_id": entity_id,
            "broadcast_features": False,
            **(metadata or {}),
        })
        # Create a zarr-backed ViewsDataset (disk-backed, lazy reads).
        return ViewsDataset.for_loa(level_lower, ds)

    def write_prediction_batch(
    self,
    *,
    target_values: np.ndarray,
    entity_ids_batch: np.ndarray,
    time_ids: np.ndarray,
    target_names: list[str],
    apply_inverse: bool = True,
    clip_negatives: bool = True,
) -> None:
        """Write one batch of predictions into the zarr-backed scaffold.

        FIX: entity_coord must be the scaffold's FULL entity axis (not the
        batch's), and the zarr group must be opened in read-write mode
        (``"r+"``) so successive batches accumulate rather than truncate.
        """
        # ── FIX 1: use the scaffold's full entity coordinate ──────────────
        entity_coord = np.asarray(
            self._ds[self._entity_id].values, dtype=np.int64
        )

        # ── FIX 2: open in read-write mode, never "w" (truncate) ─────────
        group = zarr.open_group(str(self._zarr_path), mode="r+")

        # Apply inverse transform / clip (same path as ingest_numpy_predictions)
        vals = target_values.astype(np.float32, copy=False)
        if apply_inverse and self._scalers_fitted:
            vals = self._apply_inverse_transform(vals)
        if clip_negatives:
            np.maximum(vals, 0.0, out=vals)

        for t_idx, target_name in enumerate(target_names):
            pred_var = f"pred_{target_name}"
            if pred_var not in group:
                raise KeyError(
                    f"Scaffold is missing prediction variable '{pred_var}'. "
                    f"Available: {list(group.array_keys())}."
                )
            # (n_batch, n_time, n_samples)
            batch_vals = vals[:, :, t_idx, :]
            zarr_arr = group[pred_var]

            for e_i, entity_id in enumerate(entity_ids_batch):
                # ── FIX 3: searchsorted against the FULL coordinate ──────
                ep = int(np.searchsorted(entity_coord, entity_id))

                # Sanity: verify the entity actually lives at this slot
                if ep >= len(entity_coord) or entity_coord[ep] != entity_id:
                    raise ValueError(
                        f"Entity {entity_id} not found in scaffold coordinate "
                        f"(searchsorted returned {ep}). The batch contains an "
                        f"entity the scaffold was not pre-allocated for."
                    )

                # Write to the GLOBAL position ep, not the batch-local e_i
                zarr_arr[:, ep, :] = batch_vals[e_i]

    def to_predictionframe_per_target(
        self, *, metadata: Any = None
    ) -> dict[str, Any]:
        """Convert a populated prediction scaffold to a per-target dict.

        One :class:`PredictionFrame` is built per target. The frames share
        the same :class:`SpatioTemporalIndex` (time-major, entity varies
        fastest) — matching the layout produced by
        :meth:`ingest_numpy_predictions`.

        Reads the predictions from the zarr store on disk (re-opening to
        pick up batch writes that may not be reflected in the cached
        xarray Dataset).

        Args:
            metadata: Optional :class:`FrameMetadata` to attach to each
                frame. When ``None``, a default ``FrameMetadata(model="darts")``
                is used.

        Returns:
            ``{target_name: PredictionFrame}`` dict.
        """
        import zarr
        from views_frames import (
            PredictionFrame, SpatioTemporalIndex, SpatialLevel, FrameMetadata,
        )

        if not self.is_prediction:
            raise ValueError(
                "to_predictionframe_per_target requires prediction mode"
            )

        target_names = list(self.targets)
        if not target_names:
            target_names = [
                p[len("pred_"):] for p in self.pred_vars
            ]
        if not target_names:
            raise ValueError("Scaffold has no targets / pred_vars.")

        # Re-open the zarr group to pick up batch writes. The lazy xarray
        # Dataset (self._ds) may have cached NaN values from the initial
        # read; reading from zarr directly guarantees we see the writes.
        group = zarr.open_group(str(self._zarr_path), mode="r")

        # Build the shared SpatioTemporalIndex (time-major).
        index = self._build_prediction_index()
        meta = metadata if metadata is not None else FrameMetadata(model="darts")

        frames: dict[str, Any] = {}
        for target_name in target_names:
            pred_var = f"pred_{target_name}"
            if pred_var not in group:
                raise KeyError(
                    f"Scaffold is missing prediction variable '{pred_var}'."
                )
            # Read the (time, entity, sample) array from zarr and flatten
            # time-major to (time * entity, sample).
            arr = np.array(group[pred_var], dtype=np.float32)
            t, e, s = arr.shape
            y_pred = np.ascontiguousarray(arr.reshape(t * e, s))
            frames[target_name] = PredictionFrame(y_pred, index=index, metadata=meta)
        return frames

    def _build_prediction_index(self) -> Any:
        """Build a time-major :class:`SpatioTemporalIndex` for the scaffold.

        Mirrors the index layout produced by
        :meth:`ingest_numpy_predictions` (time varies slowest, entity
        varies fastest) so the two paths produce identical frames.
        """
        from views_frames import SpatioTemporalIndex

        times = self._ds[self._time_id].values.astype("int64")
        entities = self._ds[self._entity_id].values.astype("int64")
        t_grid, e_grid = np.meshgrid(times, entities, indexing="ij")
        # time-major: time varies slowest.
        time_flat = t_grid.ravel()
        entity_flat = e_grid.ravel()
        return SpatioTemporalIndex(
            time=time_flat,
            unit=entity_flat,
            level=self._build_spatial_level(),
        )

    def _inverse_transform_numpy_predictions(
        self, target_values: np.ndarray
    ) -> np.ndarray:
        """Apply inverse target scaler + inverse log to raw numpy predictions.

        Args:
            target_values: Shape ``(n_entities, n_time, n_targets, n_samples)``.

        Returns:
            Same shape, inverse-transformed.
        """
        from views_r2darts2.transformers.inverse import (
            extract_fitted_sklearn_scaler,
        )
        from darts.dataprocessing import Pipeline

        if not self._target_scaler:
            return target_values

        n_entities, n_time, n_targets, n_samples = target_values.shape

        if isinstance(self._target_scaler, Pipeline):
            # Pipeline path: we need to reshape to 2-D, inverse-transform,
            # and reshape back. Use the sklearn objects from the pipeline.
            # Flatten to (n_entities * n_time * n_samples, n_targets)
            flat = target_values.reshape(-1, n_targets).astype(np.float64)
            # Apply each step's inverse in reverse order.
            for scaler in reversed(self._target_scaler._transformers):
                sk = extract_fitted_sklearn_scaler(scaler)
                if sk is not None:
                    # Clamp to prevent overflow (sinh of large values).
                    flat = np.clip(flat, -88.0, 88.0)
                    flat = sk.inverse_transform(flat)
                elif hasattr(scaler, "transformer"):
                    inv_func = getattr(scaler.transformer, "inverse_func", None)
                    if inv_func is not None:
                        flat = np.clip(flat, -88.0, 88.0)
                        flat = inv_func(flat)
            target_values = flat.reshape(
                n_entities, n_time, n_targets, n_samples
            ).astype(np.float32)
        else:
            # Single Scaler path.
            sk = extract_fitted_sklearn_scaler(self._target_scaler)
            if sk is not None:
                flat = target_values.reshape(-1, n_targets).astype(np.float64)
                flat = sk.inverse_transform(flat)
                target_values = flat.reshape(
                    n_entities, n_time, n_targets, n_samples
                ).astype(np.float32)
            elif hasattr(self._target_scaler, "transformer"):
                inv_func = getattr(self._target_scaler.transformer, "inverse_func", None)
                if inv_func is not None:
                    flat = np.clip(
                        target_values.astype(np.float64), -88.0, 88.0
                    )
                    target_values = inv_func(flat).astype(np.float32)

        # Apply inverse log if log_targets was set.
        if getattr(self, "_log_targets", False):
            target_values = np.expm1(
                np.maximum(target_values, 0)
            ).astype(np.float32)

        return target_values

    def _inverse_transform_predictions(self, predictions: list) -> list:
        """Apply inverse target scaler + inverse log to predictions."""
        from views_r2darts2.transformers.inverse import (
            inverse_transform_probabilistic_subset,
        )
        from darts.dataprocessing import Pipeline

        if not self._target_scaler:
            return predictions

        if isinstance(self._target_scaler, Pipeline):
            return self._target_scaler.inverse_transform(predictions)

        result = []
        for ts in predictions:
            arr = ts.all_values(copy=True)
            is_probabilistic = arr.ndim == 3
            if is_probabilistic:
                inv_values = inverse_transform_probabilistic_subset(
                    subset_3d=arr.astype(np.float32),
                    scaler=self._target_scaler,
                )
                new_ts = ts.from_times_and_values(
                    times=ts.time_index, values=inv_values.astype(np.float32),
                    columns=ts.components, freq=ts.freq,
                    static_covariates=ts.static_covariates,
                )
            else:
                new_ts = self._target_scaler.inverse_transform([ts])[0]
            result.append(new_ts)

        # Apply inverse log if log_targets was set.
        if getattr(self, "_log_targets", False):
            result = [
                ts.map(lambda arr: np.expm1(np.maximum(arr, 0)).astype(np.float32))
                for ts in result
            ]
        return result

    def _build_spatial_level(self):
        """Return the :class:`views_frames.SpatialLevel` for this dataset."""
        from views_frames import SpatialLevel
        return SpatialLevel[_ENTITY_LEVEL.get(self._entity_id, "CM")]

    # ------------------------------------------------------------------ #
    # Darts TimeSeries bridge
    # ------------------------------------------------------------------ #

    def to_darts_timeseries(
        self,
        *,
        time_ids: Any = None,
        entity_ids: Any = None,
        use_cyclic_encoders: bool = False,
    ) -> list:
        """Build per-entity Darts :class:`TimeSeries` from the zarr-backed data.

        Replaces the legacy ``ViewsDatasetDarts.as_darts_timeseries``. The
        Darts-boundary pandas import is confined to
        :mod:`views_r2darts2.transformers.darts_bridge`.

        Args:
            time_ids: Optional time-id filter.
            entity_ids: Optional entity-id filter.
            use_cyclic_encoders: When ``True``, append sin/cos cyclic time
                encoders to the feature axis.

        Returns:
            A list of Darts :class:`TimeSeries`, one per entity, ordered by
            entity id. Each series carries the full ``features + targets``
            value axis.
        """
        from views_r2darts2.transformers.darts_bridge import build_entity_timeseries
        from views_r2darts2.infrastructure.encoders import (
            CYCLIC_ENCODERS_BY_RESOLUTION,
        )

        # Get the tensor (time, entity, sample, variable) — lazy dask.
        value_columns = [*self.features, *self.targets]
        if not value_columns:
            return []

        # Temporarily enable broadcast_features so 2D scalar columns are
        # lifted to 3D with a trailing sample axis. This is required for the
        # Darts TimeSeries construction (which expects a sample axis).
        old_broadcast = self.broadcast_features
        self.broadcast_features = True
        try:
            tensor = self._stack_variables(value_columns)
        finally:
            self.broadcast_features = old_broadcast
        # Apply time/entity filters — reindex to fill missing time steps with 0
        # rather than raising KeyError. Missing time IDs mean no events recorded.
        if time_ids is not None:
            available_times = tensor[self._time_id].values
            requested = np.array(_as_list(time_ids), dtype=available_times.dtype)
            missing = np.setdiff1d(requested, available_times)
            if len(missing) > 0:
                n_entities = int(tensor[self._entity_id].shape[0]) if self._entity_id in tensor.dims else "?"
                logger.warning(
                    "to_darts_timeseries: %d/%d requested time_ids absent from dataset "
                    "and zero-filled for all %s entities. Missing: %s",
                    len(missing), len(requested), n_entities, missing.tolist(),
                )
            tensor = tensor.reindex({self._time_id: requested.tolist()}, fill_value=0.0)
        if entity_ids is not None:
            tensor = tensor.sel({self._entity_id: _as_list(entity_ids)})

        # Compute to numpy — this is the one materialization step.
        computed = tensor.compute()
        time_arr = computed[self._time_id].values.astype("int64")
        entity_arr = computed[self._entity_id].values.astype("int64")
        # NaN = structural sparsity (entity absent for those time steps) → treat as 0.
        values_4d = np.nan_to_num(computed.values, nan=0.0)  # (T, E, S, F)

        # Cyclic encoders.
        feature_columns_ext = list(value_columns)
        appended_cyclic = []
        if use_cyclic_encoders:
            resolution = self._time_id.split("_")[0][0]
            cyclic_encoders = CYCLIC_ENCODERS_BY_RESOLUTION.get(resolution)
            if cyclic_encoders is not None:
                for enc_fn in cyclic_encoders:
                    appended_cyclic.append(enc_fn(time_arr).astype(np.float32))
                    feature_columns_ext.append(enc_fn.__name__)

        series_list = []
        import pandas as pd
        from darts import TimeSeries as _TS

        S = values_4d.shape[2]

        if S == 1:
            # Entity-major C-contiguous array: arr[e_idx] is a (T, F) view — no per-entity copy.
            all_vals = np.ascontiguousarray(
                values_4d[:, :, 0, :].transpose(1, 0, 2), dtype=np.float32
            )  # (E, T, F)
            if appended_cyclic:
                cyc = np.stack(appended_cyclic, axis=1).astype(np.float32)  # (T, n_cyc)
                cyc_tiled = np.ascontiguousarray(
                    np.broadcast_to(cyc[np.newaxis], (len(entity_arr), len(time_arr), len(appended_cyclic)))
                )  # (E, T, n_cyc)
                all_vals = np.concatenate([all_vals, cyc_tiled], axis=2)  # (E, T, F+n_cyc)

        # Pre-compute objects shared across all entities.
        # pd.RangeIndex avoids _restore_range_indexed(); copy=False skips deepcopy+values.copy().
        shared_ri = pd.RangeIndex(
            start=int(time_arr[0]), stop=int(time_arr[-1]) + 1, step=1
        )
        shared_comps = pd.Index(list(feature_columns_ext))

        from tqdm import tqdm

        for e_idx, entity_id_value in tqdm(
            enumerate(entity_arr),
            total=len(entity_arr),
            desc="Building TimeSeries",
            unit="entity",
            leave=False,
        ):
            if S == 1:
                vals = all_vals[e_idx]  # (T, F) contiguous view
            else:
                # Probabilistic path — still uses the loop.
                ev = values_4d[:, e_idx, :, :]  # (T, S, F)
                vals = ev.transpose(0, 2, 1).astype(np.float32, copy=False)  # (T, F, S)
                if appended_cyclic:
                    cb = np.stack([c[:vals.shape[0]] for c in appended_cyclic], axis=1)
                    vals = np.concatenate([vals, cb[:, :, np.newaxis]], axis=1)

            series_list.append(_TS(
                times=shared_ri,
                values=vals,
                components=shared_comps,
                static_covariates=pd.Series(
                    [np.float32(entity_id_value)],
                    index=pd.Index([self._entity_id]),
                ),
                copy=False,
            ))
        return series_list


def _as_list(value: Any) -> list:
    """Coerce a scalar / array / tuple into a list."""
    if isinstance(value, (list, np.ndarray, tuple)):
        return list(value)
    return [value]


def _single_target(targets: list[str] | None, kind: str) -> str:
    if not targets or len(targets) != 1:
        raise ValueError(
            f"{kind} source requires targets=[<name>] with exactly one name"
        )
    return targets[0]
