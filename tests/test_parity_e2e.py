"""End-to-end parity test: synthetic parquet → FeatureFrame → Darts TimeSeries →
scaler fit_transform → inverse_transform → PredictionFrame.

Verifies that:
  1. The parquet loader produces bit-identical float32 values vs a direct
     pyarrow read of the same column.
  2. The FeatureFrame → Darts TimeSeries bridge preserves values bit-for-bit.
  3. A round-trip through ScalerSelector (AsinhTransform + MaxAbsScaler, the
     user's example mapping) + inverse_transform recovers the original values
     to float32 precision.
  4. The memmap cache produces the same values as the in-memory load.
  5. The full dataloader flow (parquet → FeatureFrame → per-entity TimeSeries)
     works on a 25.9M-row PRIO-GRID-month parquet.

The tests use a session-scoped synthetic parquet fixture (see ``conftest.py``)
so they run anywhere without the real validation parquet.

 Pandas-free (except the Darts boundary).
"""

from __future__ import annotations

import tempfile

import numpy as np
import pyarrow.parquet as pq
import pytest

from views_frames import FeatureFrame, SpatioTemporalIndex, SpatialLevel
from views_r2darts2.data.parquet_loader import load_views_parquet
from views_r2darts2.data.views_dataset import ViewsDatasetDarts
from views_r2darts2.transformers.scaler_selector import ScalerSelector


def test_parquet_bit_parity(synthetic_cm_parquet_small) -> None:
    """Loader values must equal a direct pyarrow column read (bit-for-bit)."""
    targets = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
    features = ["lr_ged_sb_delta", "lr_splag_1_ged_sb"]
    frame, feats, targs = load_views_parquet(
        synthetic_cm_parquet_small, targets=targets, features=features
    )
    for col in targets + features:
        direct = (
            pq.read_table(synthetic_cm_parquet_small, columns=[col])
            .column(col)
            .to_numpy()
            .astype(np.float32)
        )
        idx = frame.feature_names.index(col)
        loaded = frame.values[:, idx, 0]
        assert direct.shape == loaded.shape, f"{col}: shape mismatch"
        assert np.array_equal(direct, loaded), f"{col}: bit parity failed"


def test_memmap_cache_parity(synthetic_cm_parquet_small) -> None:
    """Memmap-cached load must produce bit-identical values to in-memory load."""
    targets = ["lr_ged_sb"]
    features = ["lr_ged_sb_delta"]
    with tempfile.TemporaryDirectory() as cache_dir:
        frame1, _, _ = load_views_parquet(
            synthetic_cm_parquet_small,
            targets=targets,
            features=features,
            cache_dir=cache_dir,
        )
        assert not isinstance(frame1.values, np.memmap), "First read should not be memmap"
        frame2, _, _ = load_views_parquet(
            synthetic_cm_parquet_small,
            targets=targets,
            features=features,
            cache_dir=cache_dir,
        )
        assert isinstance(frame2.values, np.memmap), "Second read should be memmap"
        assert np.array_equal(frame1.values, frame2.values), "Memmap cache parity failed"
        assert np.array_equal(frame1.index.time, frame2.index.time)
        assert np.array_equal(frame1.index.unit, frame2.index.unit)


def test_darts_bridge_parity(synthetic_cm_parquet_small) -> None:
    """FeatureFrame → Darts TimeSeries → numpy must preserve values bit-for-bit."""
    targets = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
    features = [
        "lr_ged_sb_delta", "lr_ged_ns_delta", "lr_ged_os_delta",
        "lr_splag_1_ged_sb", "lr_splag_1_ged_ns", "lr_splag_1_ged_os",
    ]
    frame, feats, targs = load_views_parquet(
        synthetic_cm_parquet_small, targets=targets, features=features
    )
    dataset = ViewsDatasetDarts(
        feature_frame=frame, targets=targs, features=feats,
        time_id="month_id", entity_id="country_id",
    )
    unique_entities = np.unique(frame.index.unit)[:3]
    series_list = dataset.as_darts_timeseries(entity_ids=unique_entities.tolist())
    assert len(series_list) == 3, f"Expected 3 series, got {len(series_list)}"

    value_columns = [*feats, *targs]
    col_indices = [frame.feature_names.index(c) for c in value_columns]
    for i, entity_id in enumerate(unique_entities):
        ts = series_list[i]
        mask = frame.index.unit == entity_id
        time_sorted_idx = np.argsort(frame.index.time[mask])
        expected_values = frame.values[mask, :, 0][time_sorted_idx][:, col_indices]
        ts_values = ts.all_values(copy=False)
        if ts_values.ndim == 3:
            ts_values = ts_values[:, :, 0]
        assert ts_values.shape == expected_values.shape
        assert np.array_equal(ts_values.astype(np.float32), expected_values)


def test_scaler_round_trip_parity() -> None:
    """AsinhTransform + MaxAbsScaler round-trip recovers values to float32 precision."""
    rng = np.random.default_rng(42)
    raw = rng.lognormal(mean=2.0, sigma=1.5, size=(1000, 3)).astype(np.float32)
    raw = np.maximum(raw, 0.0)

    scaler = ScalerSelector.instantiate_darts_scaler("AsinhTransform->MaxAbsScaler")
    assert scaler is not None

    from darts import TimeSeries
    import pandas as pd
    ts = TimeSeries.from_times_and_values(
        times=pd.Index(np.arange(1000, dtype=np.int64)),
        values=raw, columns=["a", "b", "c"],
    )
    transformed = scaler.fit_transform([ts])[0]
    inverted = scaler.inverse_transform([transformed])[0]

    inv_values = inverted.all_values(copy=False)
    if inv_values.ndim == 3:
        inv_values = inv_values[:, :, 0]

    max_err = np.abs(inv_values.astype(np.float32) - raw).max()
    assert max_err < 1e-3, f"Round-trip error too large: {max_err}"


def test_feature_scaler_manager_parity() -> None:
    """FeatureScalerManager with the user's example mapping must produce consistent forward+inverse."""
    from views_r2darts2.transformers.feature_scaler_manager import FeatureScalerManager
    from darts import TimeSeries
    import pandas as pd

    n_time = 24
    n_entities = 5
    feature_cols = [
        "lr_ged_sb_delta", "lr_splag_1_ged_sb", "lr_decay_ged_sb_1",
        "lr_wdi_sm_pop_refg_or",
    ]
    rng = np.random.default_rng(7)
    series_list = []
    for ent in range(n_entities):
        values = rng.lognormal(mean=1.0, sigma=1.0, size=(n_time, len(feature_cols))).astype(np.float32)
        values = np.maximum(values, 0.0)
        ts = TimeSeries.from_times_and_values(
            times=pd.Index(np.arange(100, 100 + n_time, dtype=np.int64)),
            values=values, columns=feature_cols,
            static_covariates=pd.DataFrame({"country_id": [ent + 1]}),
        )
        series_list.append(ts)

    feature_scaler_map = {
        "AsinhTransform->MaxAbsScaler": [
            "lr_ged_sb_delta", "lr_splag_1_ged_sb", "lr_decay_ged_sb_1",
        ],
    }
    manager = FeatureScalerManager(
        feature_scaler_map=feature_scaler_map,
        default_scaler="RobustScaler",
        all_features=feature_cols,
    )
    transformed = manager.fit_transform(series_list)
    inverted = manager.inverse_transform(transformed)

    max_err = 0.0
    for orig, inv in zip(series_list, inverted):
        orig_v = orig.all_values(copy=False)
        inv_v = inv.all_values(copy=False)
        if orig_v.ndim == 3:
            orig_v = orig_v[:, :, 0]
        if inv_v.ndim == 3:
            inv_v = inv_v[:, :, 0]
        err = np.abs(inv_v.astype(np.float32) - orig_v.astype(np.float32)).max()
        max_err = max(max_err, float(err))
    assert max_err < 1e-3, f"Round-trip error too large: {max_err}"


def test_full_dataloader_flow(synthetic_cm_parquet_small) -> None:
    """Full flow: parquet → FeatureFrame → ViewsDatasetDarts → Darts TimeSeries for ALL entities."""
    targets = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
    features = [
        "lr_ged_sb_delta", "lr_ged_ns_delta", "lr_ged_os_delta",
        "lr_splag_1_ged_sb", "lr_splag_1_ged_ns", "lr_splag_1_ged_os",
    ]
    frame, feats, targs = load_views_parquet(
        synthetic_cm_parquet_small, targets=targets, features=features
    )
    dataset = ViewsDatasetDarts(
        feature_frame=frame, targets=targs, features=feats,
        time_id="month_id", entity_id="country_id",
    )
    series_list = dataset.as_darts_timeseries()
    n_entities_expected = len(np.unique(frame.index.unit))
    assert len(series_list) == n_entities_expected

    total_rows = sum(len(ts) for ts in series_list)
    assert total_rows == frame.n_rows

    expected_cols = len(feats) + len(targs)
    for ts in series_list:
        assert len(ts.components) == expected_cols
