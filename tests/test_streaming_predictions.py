"""Parity tests for the streaming prediction path.

Verifies that the streaming path (entity-batched ``predict_from_dataset``
+ zarr-backed scaffold) produces bit-identical ``PredictionFrame`` output
to the in-memory path (single ``predict_from_dataset`` call +
``ingest_numpy_predictions``).

The streaming path exists to keep peak memory bounded for probabilistic
forecasts (``num_samples`` large) and/or huge entity counts (e.g. 259k
PRIO-GRID cells). It must not change the output — only the memory profile.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd  # noqa: WPS433 — Darts TimeSeries boundary
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import (
    TorchForecastingModel,
)

from views_frames import PredictionFrame
from views_r2darts2.dataset.base import ViewsDataset
from views_r2darts2.engines.darts_forecaster import DartsForecaster


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

TARGETS: list[str] = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
FEATURES: list[str] = [
    "lr_ged_sb_delta",
    "lr_splag_1_ged_sb",
    "lr_decay_ged_sb_1",
]
ENTITY_IDS: list[int] = [1, 2, 3]
PARTITION: dict[str, tuple[int, int]] = {
    "train": (121, 200),
    "test": (201, 220),
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _write_synthetic_parquet(path: Path) -> Path:
    """Write a tiny synthetic parquet mirroring the VIEWS schema."""
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


def _make_mock_model(
    input_chunk_length: int = 12,
    output_chunk_length: int = 6,
) -> Mock:
    """Build a ``Mock(spec=TorchForecastingModel)`` for predict tests."""
    m = Mock(spec=TorchForecastingModel)
    m.input_chunk_length = input_chunk_length
    m.output_chunk_length = output_chunk_length
    m.model = Mock()
    m.model.parameters.side_effect = lambda: iter(
        [Mock(device=torch.device("cpu"))]
    )
    return m


def _build_dataset(tmp_path: Path) -> ViewsDataset:
    """Build a ViewsDataset from a synthetic parquet, subsetted to 3 entities."""
    parquet_path = tmp_path / "validation_viewser_df.parquet"
    _write_synthetic_parquet(parquet_path)
    full = ViewsDataset(parquet_path, targets=TARGETS, broadcast_features=True)
    return full.get_subset_dataset(entity_ids=ENTITY_IDS)


def _seed_mock_predictions(
    mock_model: Mock,
    *,
    n_entities: int,
    n_time: int,
    n_targets: int,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Configure ``mock_model`` to return deterministic predictions.

    The mock uses ``side_effect`` to return predictions sized to the
    number of entities in each batch. The ``_build_inference_dataset``
    mock tracks the batch size via a list — each call appends the number
    of input series, and ``predict_from_dataset`` reads the next count.
    This guarantees the streaming path (multiple batches) and the
    in-memory path (one batch) see identical per-entity predictions.
    """
    rng = np.random.default_rng(seed)
    full_preds = rng.uniform(
        0.1, 2.0, size=(n_entities, n_time, n_targets, n_samples)
    ).astype(np.float32)
    batch_sizes: list[int] = []
    entity_offset = [0]  # mutable closure — tracks cumulative entity offset.

    def _build_side_effect(*args, **kwargs):
        series = kwargs.get("series", args[1] if len(args) > 1 else [])
        batch_sizes.append(len(series))
        return Mock()

    def _predict_side_effect(*args, **kwargs):
        if not batch_sizes:
            batch_len = n_entities
        else:
            batch_len = batch_sizes.pop(0)
        batch_len = min(batch_len, n_entities)
        offset = entity_offset[0]
        entity_offset[0] += batch_len
        return (
            full_preds[offset:offset + batch_len].copy(),
            [{}] * batch_len,
            list(range(batch_len)),
        )

    mock_model._build_inference_dataset.side_effect = _build_side_effect
    mock_model.predict_from_dataset.side_effect = _predict_side_effect
    return full_preds


# ----------------------------------------------------------------------
# Streaming-vs-in-memory parity
# ----------------------------------------------------------------------


class TestStreamingParity:
    """Verify the streaming path produces identical output to the in-memory
    path.

    Both paths are exercised on the same dataset + mock model. The mock
    returns the same predictions for every ``predict_from_dataset`` call,
    so the only difference between the two paths is how the predictions
    are assembled into ``PredictionFrame`` objects.
    """

    def test_deterministic_parity(self, tmp_path: Path) -> None:
        """Single-sample predictions: streaming == in-memory."""
        frames_inmem = self._run_in_memory(tmp_path, num_samples=1)
        frames_stream = self._run_streaming(tmp_path, num_samples=1)
        self._assert_parity(frames_inmem, frames_stream)

    def test_probabilistic_parity(self, tmp_path: Path) -> None:
        """Multi-sample (probabilistic) predictions: streaming == in-memory."""
        frames_inmem = self._run_in_memory(tmp_path, num_samples=5)
        frames_stream = self._run_streaming(tmp_path, num_samples=5)
        self._assert_parity(frames_inmem, frames_stream)

    def test_parity_with_scaler(self, tmp_path: Path) -> None:
        """Parity holds when a target scaler is fitted (inverse transform)."""
        frames_inmem = self._run_in_memory(
            tmp_path, num_samples=1, target_scaler="MinMaxScaler"
        )
        frames_stream = self._run_streaming(
            tmp_path, num_samples=1, target_scaler="MinMaxScaler"
        )
        self._assert_parity(frames_inmem, frames_stream)

    # ------------------------------------------------------------------ #
    # Helpers that run each path
    # ------------------------------------------------------------------ #

    def _run_in_memory(
        self,
        tmp_path: Path,
        *,
        num_samples: int,
        target_scaler: str | None = None,
    ) -> dict[str, PredictionFrame]:
        """Run the in-memory predict path (threshold high → no streaming)."""
        dataset = _build_dataset(tmp_path)
        mock_model = _make_mock_model()
        n_entities = len(ENTITY_IDS)
        n_time = 6
        n_targets = len(TARGETS)
        _seed_mock_predictions(
            mock_model,
            n_entities=n_entities,
            n_time=n_time,
            n_targets=n_targets,
            n_samples=num_samples,
        )
        fc = DartsForecaster(
            dataset=dataset,
            model=mock_model,
            partition_dict=PARTITION,
            target_scaler=target_scaler,
            random_state=42,
        )
        fc.dataset.fit_scalers(
            target_scaler=target_scaler,
            feature_scaler="RobustScaler" if target_scaler else None,
            time_ids=list(range(121, 201)),
        )
        fc.scaler_fitted = True
        # Force the in-memory path.
        fc.STREAMING_CELL_THRESHOLD = 10**18
        return fc.predict(
            sequence_number=0,
            output_length=n_time,
            num_samples=num_samples,
            mc_dropout=False,
        )

    def _run_streaming(
        self,
        tmp_path: Path,
        *,
        num_samples: int,
        target_scaler: str | None = None,
    ) -> dict[str, PredictionFrame]:
        """Run the streaming predict path (threshold low → always stream)."""
        dataset = _build_dataset(tmp_path)
        mock_model = _make_mock_model()
        n_entities = len(ENTITY_IDS)
        n_time = 6
        n_targets = len(TARGETS)
        _seed_mock_predictions(
            mock_model,
            n_entities=n_entities,
            n_time=n_time,
            n_targets=n_targets,
            n_samples=num_samples,
        )
        fc = DartsForecaster(
            dataset=dataset,
            model=mock_model,
            partition_dict=PARTITION,
            target_scaler=target_scaler,
            random_state=42,
        )
        fc.dataset.fit_scalers(
            target_scaler=target_scaler,
            feature_scaler="RobustScaler" if target_scaler else None,
            time_ids=list(range(121, 201)),
        )
        fc.scaler_fitted = True
        # Force the streaming path.
        fc.STREAMING_CELL_THRESHOLD = 0
        fc.STREAMING_ENTITY_BATCH = 2  # multiple batches
        return fc.predict(
            sequence_number=0,
            output_length=n_time,
            num_samples=num_samples,
            mc_dropout=False,
        )

    def _assert_parity(
        self,
        frames_inmem: dict[str, PredictionFrame],
        frames_stream: dict[str, PredictionFrame],
    ) -> None:
        """Assert the two frame dicts are bit-identical."""
        assert set(frames_inmem.keys()) == set(frames_stream.keys())
        for tgt in frames_inmem:
            a = frames_inmem[tgt]
            b = frames_stream[tgt]
            assert a.n_rows == b.n_rows, (
                f"target '{tgt}': row count {a.n_rows} != {b.n_rows}"
            )
            np.testing.assert_array_equal(
                a.values,
                b.values,
                err_msg=f"target '{tgt}': values differ",
            )


# ----------------------------------------------------------------------
# Threshold + batch-size logic
# ----------------------------------------------------------------------


class TestStreamingThreshold:
    """Tests for ``_should_stream_predictions`` and the batch loop."""

    def test_small_forecast_does_not_stream(self, tmp_path: Path) -> None:
        """A small deterministic forecast uses the in-memory path."""
        dataset = _build_dataset(tmp_path)
        fc = DartsForecaster(
            dataset=dataset,
            model=_make_mock_model(),
            partition_dict=PARTITION,
            target_scaler=None,
            random_state=42,
        )
        # 3 entities × 6 steps × 3 targets × 1 sample = 54 cells.
        assert fc._should_stream_predictions(
            n_entities=3, n_time=6, num_samples=1
        ) is False

    def test_large_forecast_streams(self, tmp_path: Path) -> None:
        """A large probabilistic forecast uses the streaming path."""
        dataset = _build_dataset(tmp_path)
        fc = DartsForecaster(
            dataset=dataset,
            model=_make_mock_model(),
            partition_dict=PARTITION,
            target_scaler=None,
            random_state=42,
        )
        # 259k entities × 36 steps × 3 targets × 500 samples ≈ 14B cells.
        assert fc._should_stream_predictions(
            n_entities=259_000, n_time=36, num_samples=500
        ) is True

    def test_streaming_calls_predict_in_batches(self, tmp_path: Path) -> None:
        """The streaming path calls ``predict_from_dataset`` once per batch."""
        dataset = _build_dataset(tmp_path)
        mock_model = _make_mock_model()
        n_entities = len(ENTITY_IDS)
        n_time = 6
        n_targets = len(TARGETS)
        _seed_mock_predictions(
            mock_model,
            n_entities=n_entities,
            n_time=n_time,
            n_targets=n_targets,
            n_samples=1,
        )
        fc = DartsForecaster(
            dataset=dataset,
            model=mock_model,
            partition_dict=PARTITION,
            target_scaler=None,
            random_state=42,
        )
        fc.dataset.fit_scalers(
            target_scaler=None,
            feature_scaler=None,
            time_ids=list(range(121, 201)),
        )
        fc.scaler_fitted = True
        fc.STREAMING_CELL_THRESHOLD = 0
        fc.STREAMING_ENTITY_BATCH = 1  # one entity per batch → 3 batches

        fc.predict(
            sequence_number=0,
            output_length=n_time,
            num_samples=1,
            mc_dropout=False,
        )
        # 3 entities / batch=1 → 3 predict_from_dataset calls.
        assert mock_model.predict_from_dataset.call_count == 3


# ----------------------------------------------------------------------
# Scaffold API (direct unit tests)
# ----------------------------------------------------------------------


class TestPredictionScaffold:
    """Direct unit tests for the scaffold API on :class:`ViewsDataset`."""

    def test_create_prediction_scaffold(self) -> None:
        """The scaffold has the right shape, targets, and sample size."""
        entity_ids = np.array([1, 2, 3], dtype=np.int64)
        time_ids = np.array([201, 202, 203], dtype=np.int64)
        targets = ["lr_ged_sb"]
        sample_size = 5

        scaffold = ViewsDataset.create_prediction_scaffold(
            entity_ids=entity_ids,
            time_ids=time_ids,
            targets=targets,
            sample_size=sample_size,
            level="cm",
        )
        assert scaffold.is_prediction is True
        assert scaffold.sample_size == sample_size
        assert scaffold.targets == targets
        assert scaffold.pred_vars == ["pred_lr_ged_sb"]
        # Shape: (n_time, n_entities, n_samples).
        arr = scaffold._ds["pred_lr_ged_sb"].values
        assert arr.shape == (3, 3, sample_size)
        # All NaN initially.
        assert np.isnan(arr).all()

    def test_write_prediction_batch_and_read_back(self) -> None:
        """Write a batch, then read it back via to_predictionframe_per_target."""
        entity_ids = np.array([1, 2], dtype=np.int64)
        time_ids = np.array([201, 202], dtype=np.int64)
        targets = ["lr_ged_sb"]
        sample_size = 3

        scaffold = ViewsDataset.create_prediction_scaffold(
            entity_ids=entity_ids,
            time_ids=time_ids,
            targets=targets,
            sample_size=sample_size,
            level="cm",
        )
        # No scalers fitted → inverse transform is skipped.
        scaffold._scalers_fitted = False

        # Write a batch: (n_batch=2, n_time=2, n_targets=1, n_samples=3).
        batch = np.full((2, 2, 1, 3), 0.7, dtype=np.float32)
        scaffold.write_prediction_batch(
            target_values=batch,
            entity_ids_batch=entity_ids,
            time_ids=time_ids,
            target_names=targets,
            apply_inverse=True,
            clip_negatives=True,
        )

        frames = scaffold.to_predictionframe_per_target()
        assert set(frames.keys()) == {"lr_ged_sb"}
        frame = frames["lr_ged_sb"]
        # 2 entities × 2 time steps = 4 rows × 3 samples.
        assert frame.n_rows == 4
        assert frame.values.shape == (4, 3)
        np.testing.assert_allclose(frame.values, np.full((4, 3), 0.7), atol=1e-6)

    def test_write_prediction_batch_clips_negatives(self) -> None:
        """Negative predictions are clipped to 0."""
        entity_ids = np.array([1], dtype=np.int64)
        time_ids = np.array([201], dtype=np.int64)
        scaffold = ViewsDataset.create_prediction_scaffold(
            entity_ids=entity_ids,
            time_ids=time_ids,
            targets=["lr_ged_sb"],
            sample_size=1,
            level="cm",
        )
        scaffold._scalers_fitted = False

        batch = np.full((1, 1, 1, 1), -5.0, dtype=np.float32)
        scaffold.write_prediction_batch(
            target_values=batch,
            entity_ids_batch=entity_ids,
            time_ids=time_ids,
            target_names=["lr_ged_sb"],
            apply_inverse=False,
            clip_negatives=True,
        )
        frames = scaffold.to_predictionframe_per_target()
        assert frames["lr_ged_sb"].values.min() == 0.0
