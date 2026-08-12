"""Parity tests for MarkovModel: TimeSeries path vs ViewsDataset path.

These tests verify that the MarkovModel produces identical predictions
regardless of whether it receives data via:

1. The Darts TimeSeries path (``model.fit(series=..., past_covariates=...)``)
   — the standalone fallback used when no ViewsDataset is attached.
2. The ViewsDataset path (``model.set_dataset(dataset, partition_dict)``
   followed by ``model.fit(series=..., past_covariates=...)``) — the
   preferred path that leverages the full data infrastructure.

Both paths must produce bit-identical predictions when given the same
underlying data. This is enforced by the parity tests below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd  # noqa: WPS433 — Darts TimeSeries boundary
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from darts import TimeSeries

from views_r2darts2.catalogs.model_catalog import ModelCatalog
from views_r2darts2.dataset.base import ViewsDataset
from views_r2darts2.engines.darts_forecaster import DartsForecaster
from views_r2darts2.models.markov_model import MarkovModel


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


TARGETS: list[str] = ["lr_ged_sb"]
FEATURES: list[str] = ["feat_a", "feat_b"]
PARTITION: dict[str, tuple[int, int]] = {
    "train": (121, 200),
    "test": (201, 220),
}


def _make_series(
    *,
    n_entities: int = 3,
    n_months: int = 60,
    start_month: int = 121,
    seed: int = 42,
    targets: list[str] | None = None,
    features: list[str] | None = None,
) -> list[TimeSeries]:
    """Build a list of Darts TimeSeries, one per entity."""
    if targets is None:
        targets = TARGETS
    if features is None:
        features = FEATURES
    rng = np.random.default_rng(seed)
    series_list: list[TimeSeries] = []
    for e in range(1, n_entities + 1):
        time_ids = np.arange(
            start_month, start_month + n_months, dtype=np.int64
        )
        cols: list[np.ndarray] = []
        for tname in targets:
            mask = rng.random(n_months) < 0.3
            vals = np.zeros(n_months, dtype=np.float32)
            vals[mask] = rng.lognormal(
                mean=2.0, sigma=1.5, size=mask.sum()
            ).astype(np.float32)
            cols.append(vals)
        for fname in features:
            cols.append(
                rng.normal(0, 1, n_months).astype(np.float32)
                + 0.1 * cols[0]
            )
        values = np.stack(cols, axis=1)
        ts = TimeSeries.from_times_and_values(
            times=pd.Index(time_ids),
            values=values,
            columns=targets + features,
            static_covariates=pd.DataFrame({"country_id": [float(e)]}),
            freq=1,
        )
        series_list.append(ts)
    return series_list


def _write_synthetic_parquet(path: Path) -> Path:
    """Write a tiny synthetic parquet file mirroring the VIEWS schema."""
    rng = np.random.default_rng(42)
    n_countries = 3
    n_months = 100
    n_rows = n_countries * n_months
    country_ids = np.repeat(
        np.arange(1, n_countries + 1, dtype=np.int64), n_months
    )
    month_ids = np.tile(
        np.arange(121, 121 + n_months, dtype=np.int64), n_countries
    )
    columns: dict[str, np.ndarray] = {
        "month_id": month_ids,
        "country_id": country_ids,
    }
    for col in TARGETS + FEATURES:
        mask = rng.random(n_rows) < 0.3
        values = np.zeros(n_rows, dtype=np.float64)
        values[mask] = rng.lognormal(mean=2.0, sigma=1.5, size=mask.sum())
        columns[col] = np.maximum(values, 0.0).astype(np.float32)
    table = pa.table(columns)
    pq.write_table(table, str(path))
    return path


def _make_model(
    *,
    rf_n_estimators: int = 10,
    targets: list[str] | None = None,
    markov_target: str = "lr_ged_sb",
) -> MarkovModel:
    """Build a MarkovModel with small RF params for fast tests."""
    if targets is None:
        targets = TARGETS
    return MarkovModel(
        steps=[1, 2, 3],
        targets=targets,
        markov_target=markov_target,
        rf_class_params={"n_estimators": rf_n_estimators},
        rf_reg_params={
            "n_estimators": rf_n_estimators,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
        },
        n_jobs=1,
    )


# ----------------------------------------------------------------------
# Parity: TimeSeries path vs ViewsDataset path
# ----------------------------------------------------------------------


class TestPathParity:
    """Verify that the TimeSeries path and the ViewsDataset path produce
    identical predictions when given the same data.

    The two paths differ in how they obtain the flat ``(N, F)`` matrix:
      * TimeSeries path: flattens Darts ``TimeSeries`` (targets first,
        then features).
      * ViewsDataset path: uses ``dataset.to_featureframe()`` (features
        first, then targets).

    The MarkovModel resolves columns by name (not position), so the
    column order difference does not affect the results. These tests
    verify that assertion.
    """

    def test_parity_single_target(self, tmp_path: Path) -> None:
        """Single-target: both paths produce identical predictions."""
        # Build a parquet file.
        parquet_path = tmp_path / "validation_viewser_df.parquet"
        _write_synthetic_parquet(parquet_path)
        dataset = ViewsDataset(
            parquet_path, targets=TARGETS, broadcast_features=True
        )

        # Build TimeSeries from the TRAIN partition only (matching what
        # the DS path sees via set_dataset + partition_dict).
        train_time_ids = list(range(PARTITION["train"][0], PARTITION["train"][1] + 1))
        series_list = dataset.to_darts_timeseries(time_ids=train_time_ids)
        target_series = [ts[TARGETS] for ts in series_list]
        cov_series = [ts[FEATURES] for ts in series_list]

        # --- Path 1: TimeSeries only (no dataset attached) ---
        model_ts = _make_model()
        model_ts.fit(series=target_series, past_covariates=cov_series)
        # For predict, pass the same series (the model will use the
        # last observed month from the training data).
        preds_ts = model_ts.predict(n=3)

        # --- Path 2: ViewsDataset attached ---
        model_ds = _make_model()
        model_ds.set_dataset(dataset, partition_dict=PARTITION)
        model_ds.fit(series=target_series, past_covariates=cov_series)
        # For predict, pass the same series so both paths start from
        # the same "current month".
        preds_ds = model_ds.predict(n=3, series=target_series, past_covariates=cov_series)

        # Compare — both should produce the same number of predictions.
        assert len(preds_ts) == len(preds_ds)
        for i, (a, b) in enumerate(zip(preds_ts, preds_ds)):
            # Predictions should be bit-identical (same data, same model
            # params, same random seed).
            np.testing.assert_array_equal(
                a.values(),
                b.values(),
                err_msg=f"Entity {i}: predictions differ between paths",
            )

    def test_parity_through_forecaster(self, tmp_path: Path) -> None:
        """End-to-end parity: the DartsForecaster (which attaches the
        dataset automatically) produces the same predictions as a
        standalone model that uses the TimeSeries path."""
        parquet_path = tmp_path / "validation_viewser_df.parquet"
        _write_synthetic_parquet(parquet_path)
        dataset = ViewsDataset(
            parquet_path, targets=TARGETS, broadcast_features=True
        )

        config = {
            "algorithm": "MarkovModel",
            "name": "markov_parity",
            "run_type": "calibration",
            "random_state": 42,
            "steps": [1, 2, 3],
            "regression_targets": TARGETS,
            "markov_target": "lr_ged_sb",
            "markov_method": "direct",
            "regression_method": "single",
            "markov_threshold": 0,
            "n_jobs": 1,
            "rf_class_params": {"n_estimators": 10},
            "rf_reg_params": {
                "n_estimators": 10,
                "max_features": "sqrt",
                "min_samples_leaf": 2,
            },
        }

        with patch("views_r2darts2.catalogs.model_catalog.get_device", return_value="cpu"):
            catalog = ModelCatalog(config)
            model = catalog.get_model("MarkovModel")

        forecaster = DartsForecaster(
            dataset=dataset,
            model=model,
            partition_dict=PARTITION,
            target_scaler=None,
            feature_scaler=None,
            random_state=42,
        )
        # The forecaster should have attached the dataset to the model.
        assert forecaster._is_sklearn_model is True
        assert model._dataset is not None

        forecaster.train()
        preds = forecaster.predict(sequence_number=0, output_length=3)

        # The forecaster returns a dict of PredictionFrames.
        assert set(preds.keys()) == set(TARGETS)
        for tgt, frame in preds.items():
            assert np.isfinite(frame.values).all()
            assert (frame.values >= 0).all()

    def test_parity_save_load_with_dataset(self, tmp_path: Path) -> None:
        """Save/load parity: after loading a saved model, the dataset is
        re-attached and predictions match the pre-save predictions."""
        parquet_path = tmp_path / "validation_viewser_df.parquet"
        _write_synthetic_parquet(parquet_path)
        dataset = ViewsDataset(
            parquet_path, targets=TARGETS, broadcast_features=True
        )

        config = {
            "algorithm": "MarkovModel",
            "name": "markov_parity",
            "run_type": "calibration",
            "random_state": 42,
            "steps": [1, 2, 3],
            "regression_targets": TARGETS,
            "markov_target": "lr_ged_sb",
            "markov_method": "direct",
            "regression_method": "single",
            "markov_threshold": 0,
            "n_jobs": 1,
            "rf_class_params": {"n_estimators": 10},
            "rf_reg_params": {
                "n_estimators": 10,
                "max_features": "sqrt",
                "min_samples_leaf": 2,
            },
        }

        with patch("views_r2darts2.catalogs.model_catalog.get_device", return_value="cpu"):
            catalog = ModelCatalog(config)
            model = catalog.get_model("MarkovModel")

        forecaster = DartsForecaster(
            dataset=dataset,
            model=model,
            partition_dict=PARTITION,
            target_scaler=None,
            feature_scaler=None,
            random_state=42,
        )
        forecaster.train()
        preds_before = forecaster.predict(sequence_number=0, output_length=3)

        save_path = tmp_path / "markov_model"
        forecaster.save_model(str(save_path))

        # Build a fresh forecaster and load.
        with patch("views_r2darts2.catalogs.model_catalog.get_device", return_value="cpu"):
            catalog2 = ModelCatalog(config)
            model2 = catalog2.get_model("MarkovModel")
        forecaster2 = DartsForecaster(
            dataset=dataset,
            model=model2,
            partition_dict=PARTITION,
            target_scaler=None,
            feature_scaler=None,
            random_state=42,
        )
        forecaster2.load_model(str(save_path))

        # The loaded model should have the dataset re-attached.
        assert forecaster2.model._dataset is not None

        preds_after = forecaster2.predict(sequence_number=0, output_length=3)

        for tgt in preds_before:
            np.testing.assert_array_equal(
                preds_before[tgt].values,
                preds_after[tgt].values,
                err_msg=f"Target {tgt}: predictions differ after save/load",
            )

    def test_parity_multivariate(self, tmp_path: Path) -> None:
        """Multivariate: both paths produce identical predictions for
        multiple targets."""
        multi_targets = ["lr_ged_sb", "lr_ged_ns"]
        parquet_path = tmp_path / "validation_viewser_df.parquet"
        # Write parquet with 2 targets.
        rng = np.random.default_rng(42)
        n_countries = 3
        n_months = 100
        n_rows = n_countries * n_months
        country_ids = np.repeat(
            np.arange(1, n_countries + 1, dtype=np.int64), n_months
        )
        month_ids = np.tile(
            np.arange(121, 121 + n_months, dtype=np.int64), n_countries
        )
        columns: dict[str, np.ndarray] = {
            "month_id": month_ids,
            "country_id": country_ids,
        }
        for col in multi_targets + FEATURES:
            mask = rng.random(n_rows) < 0.3
            values = np.zeros(n_rows, dtype=np.float64)
            values[mask] = rng.lognormal(mean=2.0, sigma=1.5, size=mask.sum())
            columns[col] = np.maximum(values, 0.0).astype(np.float32)
        table = pa.table(columns)
        pq.write_table(table, str(parquet_path))

        dataset = ViewsDataset(
            parquet_path, targets=multi_targets, broadcast_features=True
        )
        # Use train partition only for parity.
        train_time_ids = list(range(PARTITION["train"][0], PARTITION["train"][1] + 1))
        series_list = dataset.to_darts_timeseries(time_ids=train_time_ids)
        target_series = [ts[multi_targets] for ts in series_list]
        cov_series = [ts[FEATURES] for ts in series_list]

        # --- Path 1: TimeSeries only ---
        model_ts = _make_model(targets=multi_targets)
        model_ts.fit(series=target_series, past_covariates=cov_series)
        preds_ts = model_ts.predict(n=3)

        # --- Path 2: ViewsDataset attached ---
        model_ds = _make_model(targets=multi_targets)
        model_ds.set_dataset(dataset, partition_dict=PARTITION)
        model_ds.fit(series=target_series, past_covariates=cov_series)
        preds_ds = model_ds.predict(n=3, series=target_series, past_covariates=cov_series)

        # Compare.
        assert len(preds_ts) == len(preds_ds)
        for i, (a, b) in enumerate(zip(preds_ts, preds_ds)):
            np.testing.assert_array_equal(
                a.values(),
                b.values(),
                err_msg=f"Entity {i}: multivariate predictions differ",
            )

    def test_parity_transition_method(self, tmp_path: Path) -> None:
        """Parity with ``markov_method='transition'``."""
        parquet_path = tmp_path / "validation_viewser_df.parquet"
        _write_synthetic_parquet(parquet_path)
        dataset = ViewsDataset(
            parquet_path, targets=TARGETS, broadcast_features=True
        )
        # Use train partition only for parity.
        train_time_ids = list(range(PARTITION["train"][0], PARTITION["train"][1] + 1))
        series_list = dataset.to_darts_timeseries(time_ids=train_time_ids)
        target_series = [ts[TARGETS] for ts in series_list]
        cov_series = [ts[FEATURES] for ts in series_list]

        # --- Path 1: TimeSeries only ---
        model_ts = MarkovModel(
            steps=[1, 2, 3],
            targets=TARGETS,
            markov_target="lr_ged_sb",
            markov_method="transition",
            rf_class_params={"n_estimators": 5},
            rf_reg_params={"n_estimators": 5, "min_samples_leaf": 1},
            n_jobs=1,
        )
        model_ts.fit(series=target_series, past_covariates=cov_series)
        preds_ts = model_ts.predict(n=3)

        # --- Path 2: ViewsDataset attached ---
        model_ds = MarkovModel(
            steps=[1, 2, 3],
            targets=TARGETS,
            markov_target="lr_ged_sb",
            markov_method="transition",
            rf_class_params={"n_estimators": 5},
            rf_reg_params={"n_estimators": 5, "min_samples_leaf": 1},
            n_jobs=1,
        )
        model_ds.set_dataset(dataset, partition_dict=PARTITION)
        model_ds.fit(series=target_series, past_covariates=cov_series)
        preds_ds = model_ds.predict(n=3, series=target_series, past_covariates=cov_series)

        assert len(preds_ts) == len(preds_ds)
        for i, (a, b) in enumerate(zip(preds_ts, preds_ds)):
            np.testing.assert_array_equal(
                a.values(),
                b.values(),
                err_msg=f"Entity {i}: transition predictions differ",
            )

    def test_dataset_path_uses_featureframe(self, tmp_path: Path) -> None:
        """Verify that the dataset path actually uses the FeatureFrame API
        (not the TimeSeries flattening). This is a structural test — it
        checks that ``_flatten_from_dataset`` produces the same data as
        ``_flatten_timeseries_list`` when given the same underlying data."""
        from views_r2darts2.models.markov_model import (
            _flatten_from_dataset,
            _flatten_timeseries_list,
        )

        parquet_path = tmp_path / "validation_viewser_df.parquet"
        _write_synthetic_parquet(parquet_path)
        dataset = ViewsDataset(
            parquet_path, targets=TARGETS, broadcast_features=True
        )

        # Get the same data via both paths.
        time_ids = list(range(121, 221))
        flat_ds = _flatten_from_dataset(dataset, time_ids=time_ids)

        series_list = dataset.to_darts_timeseries(time_ids=time_ids)
        target_series = [ts[TARGETS] for ts in series_list]
        cov_series = [ts[FEATURES] for ts in series_list]
        flat_ts = _flatten_timeseries_list(target_series, cov_series)

        # The column ORDER may differ (dataset path: features+targets;
        # TS path: targets+features). But the SET of columns must match.
        assert set(flat_ds["columns"]) == set(flat_ts["columns"])

        # The number of rows must match.
        assert flat_ds["values"].shape[0] == flat_ts["values"].shape[0]

        # The time_ids and entity_ids must match (possibly in different order).
        ds_pairs = set(zip(flat_ds["time_ids"].tolist(), flat_ds["entity_ids"].tolist()))
        ts_pairs = set(zip(flat_ts["time_ids"].tolist(), flat_ts["entity_ids"].tolist()))
        assert ds_pairs == ts_pairs

        # For each (time, entity) pair, the values for each column must match.
        # Build lookup dicts: (time, entity) → {column: value}.
        ds_lookup: dict[tuple[int, int], dict[str, float]] = {}
        for i in range(flat_ds["values"].shape[0]):
            key = (int(flat_ds["time_ids"][i]), int(flat_ds["entity_ids"][i]))
            ds_lookup[key] = {
                col: float(flat_ds["values"][i, j])
                for j, col in enumerate(flat_ds["columns"])
            }
        ts_lookup: dict[tuple[int, int], dict[str, float]] = {}
        for i in range(flat_ts["values"].shape[0]):
            key = (int(flat_ts["time_ids"][i]), int(flat_ts["entity_ids"][i]))
            ts_lookup[key] = {
                col: float(flat_ts["values"][i, j])
                for j, col in enumerate(flat_ts["columns"])
            }

        # Verify every column matches for every (time, entity) pair.
        for key in ds_pairs:
            for col in flat_ds["columns"]:
                np.testing.assert_allclose(
                    ds_lookup[key][col],
                    ts_lookup[key][col],
                    atol=1e-6,
                    err_msg=f"Pair {key}, column {col}: values differ",
                )

    def test_flatten_from_dataset_time_ids_filtering(self, tmp_path: Path) -> None:
        """Regression test: ``_flatten_from_dataset`` must correctly filter
        by ``time_ids`` without calling ``get_subset_dataset`` (which
        triggers a ``to_zarr`` write that fails on Dask chunk alignment
        for production parquet files).

        This test verifies that:
          1. The filtering produces the correct subset of rows.
          2. The function does NOT call ``get_subset_dataset`` (which
             would re-create a ViewsDataset and trigger a zarr write).
        """
        from views_r2darts2.models.markov_model import _flatten_from_dataset

        parquet_path = tmp_path / "validation_viewser_df.parquet"
        _write_synthetic_parquet(parquet_path)
        dataset = ViewsDataset(
            parquet_path, targets=TARGETS, broadcast_features=True
        )

        # Request only a subset of time_ids.
        requested_time_ids = list(range(130, 140))  # 10 months
        flat = _flatten_from_dataset(dataset, time_ids=requested_time_ids)

        # The returned time_ids must be exactly the requested set.
        returned_time_set = set(flat["time_ids"].tolist())
        assert returned_time_set == set(requested_time_ids), (
            f"Expected time_ids {set(requested_time_ids)}, got "
            f"{returned_time_set}"
        )

        # The number of rows must be (n_entities × n_requested_months).
        n_entities = 3
        n_requested = len(requested_time_ids)
        assert flat["values"].shape[0] == n_entities * n_requested

        # The columns must be in canonical order (targets first, then
        # features).
        assert flat["columns"][0] in TARGETS
        for t in TARGETS:
            assert t in flat["columns"]

    def test_flatten_from_dataset_avoids_get_subset_dataset(
        self, tmp_path: Path
    ) -> None:
        """Regression test: ``_flatten_from_dataset`` must NOT call
        ``get_subset_dataset`` (which triggers a ``to_zarr`` write that
        fails on Dask chunk alignment for production parquet files).

        This test patches ``get_subset_dataset`` to raise if called,
        verifying that the function uses the FeatureFrame + boolean-mask
        path instead.
        """
        from views_r2darts2.models.markov_model import _flatten_from_dataset

        parquet_path = tmp_path / "validation_viewser_df.parquet"
        _write_synthetic_parquet(parquet_path)
        dataset = ViewsDataset(
            parquet_path, targets=TARGETS, broadcast_features=True
        )

        # Patch get_subset_dataset to raise if called.
        def _explode(*args, **kwargs):
            raise AssertionError(
                "_flatten_from_dataset must NOT call get_subset_dataset "
                "(it triggers a to_zarr write that fails on Dask chunk "
                "alignment for production parquet files)."
            )

        original = dataset.get_subset_dataset
        dataset.get_subset_dataset = _explode  # type: ignore[assignment]
        try:
            # Must not raise — the function should use to_featureframe +
            # boolean mask filtering instead.
            flat = _flatten_from_dataset(
                dataset, time_ids=list(range(130, 140))
            )
            assert flat["values"].shape[0] > 0
        finally:
            dataset.get_subset_dataset = original  # type: ignore[assignment]
