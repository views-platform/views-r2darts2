"""End-to-end smoke test: DartsForecaster streaming predict via DatasetBuilder.

Verifies that the forecaster's ``_predict_streaming`` method (which uses
``ViewsDataset.builder``) produces correct ``PredictionFrame`` output when
driven by a mock torch model. The mock returns deterministic predictions
sized to the entity batch, so the streaming path's output can be verified
against expected values.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock

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


TARGETS: list[str] = ["lr_ged_sb", "lr_ged_ns"]
FEATURES: list[str] = ["feat_a", "feat_b"]
ENTITY_IDS: list[int] = [1, 2, 3]
PARTITION: dict[str, tuple[int, int]] = {
    "train": (121, 200),
    "test": (201, 220),
}


def _write_synthetic_parquet(path: Path) -> Path:
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
    m = Mock(spec=TorchForecastingModel)
    m.input_chunk_length = input_chunk_length
    m.output_chunk_length = output_chunk_length
    m.model = Mock()
    m.model.parameters.side_effect = lambda: iter(
        [Mock(device=torch.device("cpu"))]
    )
    return m


def _seed_mock_predictions(
    mock_model: Mock,
    *,
    n_entities: int,
    n_time: int,
    n_targets: int,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    full_preds = rng.uniform(
        0.1, 2.0, size=(n_entities, n_time, n_targets, n_samples)
    ).astype(np.float32)
    batch_sizes: list[int] = []
    entity_offset = [0]

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


class TestStreamingPredictViaBuilder:
    """Verify the forecaster's streaming predict path produces correct output."""

    def test_all_entities_present_in_output(self, tmp_path: Path) -> None:
        """Every entity in the input must appear in the streaming output."""
        parquet_path = tmp_path / "validation_viewser_df.parquet"
        _write_synthetic_parquet(parquet_path)
        full = ViewsDataset(parquet_path, targets=TARGETS, broadcast_features=True)
        dataset = full.get_subset_dataset(entity_ids=ENTITY_IDS)

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
            feature_scaler=None,
            random_state=42,
        )
        fc.dataset.fit_scalers(
            target_scaler=None,
            feature_scaler=None,
            time_ids=list(range(121, 201)),
        )
        fc.scaler_fitted = True
        fc.STREAMING_ENTITY_BATCH = 1  # one entity per batch → 3 batches

        frames = fc.predict(
            sequence_number=0,
            output_length=n_time,
            num_samples=1,
            mc_dropout=False,
        )

        assert set(frames.keys()) == set(TARGETS)
        for tgt, frame in frames.items():
            assert frame.n_rows == n_entities * n_time, (
                f"target '{tgt}': expected {n_entities * n_time} rows, "
                f"got {frame.n_rows}"
            )
            frame_entity_ids = set(frame.index.unit.tolist())
            assert frame_entity_ids == set(ENTITY_IDS), (
                f"target '{tgt}': expected entities {set(ENTITY_IDS)}, "
                f"got {frame_entity_ids}"
            )
            assert not np.isnan(frame.values).any(), (
                f"target '{tgt}': NaN values in output"
            )

    def test_streaming_calls_predict_in_batches(self, tmp_path: Path) -> None:
        """The streaming path calls ``predict_from_dataset`` once per batch."""
        parquet_path = tmp_path / "validation_viewser_df.parquet"
        _write_synthetic_parquet(parquet_path)
        full = ViewsDataset(parquet_path, targets=TARGETS, broadcast_features=True)
        dataset = full.get_subset_dataset(entity_ids=ENTITY_IDS)

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
            feature_scaler=None,
            random_state=42,
        )
        fc.dataset.fit_scalers(
            target_scaler=None,
            feature_scaler=None,
            time_ids=list(range(121, 201)),
        )
        fc.scaler_fitted = True
        fc.STREAMING_ENTITY_BATCH = 1

        fc.predict(
            sequence_number=0,
            output_length=n_time,
            num_samples=1,
            mc_dropout=False,
        )
        assert mock_model.predict_from_dataset.call_count == 3

    def test_predictions_are_non_negative(self, tmp_path: Path) -> None:
        """Predictions must be non-negative (clipped)."""
        parquet_path = tmp_path / "validation_viewser_df.parquet"
        _write_synthetic_parquet(parquet_path)
        full = ViewsDataset(parquet_path, targets=TARGETS, broadcast_features=True)
        dataset = full.get_subset_dataset(entity_ids=ENTITY_IDS)

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
            feature_scaler=None,
            random_state=42,
        )
        fc.dataset.fit_scalers(
            target_scaler=None,
            feature_scaler=None,
            time_ids=list(range(121, 201)),
        )
        fc.scaler_fitted = True
        fc.STREAMING_ENTITY_BATCH = 2

        frames = fc.predict(
            sequence_number=0,
            output_length=n_time,
            num_samples=1,
            mc_dropout=False,
        )
        for tgt, frame in frames.items():
            assert (frame.values >= 0).all(), (
                f"target '{tgt}': negative values in output"
            )
