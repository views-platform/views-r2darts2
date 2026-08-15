"""End-to-end parity test: synthetic parquet → ViewsDataset → Darts TimeSeries →
scaler fit_transform → inverse_transform → PredictionFrame.

Verifies that:
  1. The parquet loader produces bit-identical float32 values vs a direct
     pyarrow read of the same column.
  2. The ViewsDataset → Darts TimeSeries bridge preserves values bit-for-bit.
  3. A round-trip through ScalerSelector (AsinhTransform + MaxAbsScaler, the
     user's example mapping) + inverse_transform recovers the original values
     to float32 precision.
  4. The full dataloader flow (parquet → ViewsDataset → per-entity TimeSeries)
     works on the 100k-row cm parquet.

The tests use a session-scoped synthetic parquet fixture (see ``conftest.py``)
so they run anywhere without the real validation parquet.

Pandas-free (except the Darts boundary).
"""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest

from views_r2darts2.dataset.base import ViewsDataset
from views_r2darts2.transformers.scaler_selector import ScalerSelector


def test_parquet_bit_parity(synthetic_cm_parquet_small) -> None:
    """Loader values must equal a direct pyarrow column read (bit-for-bit)."""
    targets = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
    features = [
        "lr_ged_sb_delta", "lr_ged_ns_delta", "lr_ged_os_delta",
        "lr_splag_1_ged_sb", "lr_splag_1_ged_ns", "lr_splag_1_ged_os",
        "lr_decay_ged_sb_1", "lr_decay_ged_sb_5", "lr_decay_ged_sb_25",
    ]
    ds = ViewsDataset(
        synthetic_cm_parquet_small,
        targets=targets,
        broadcast_features=True,
    )
    assert ds.features == features
    assert ds.targets == targets

    # The tensor is (T, E, S=1, V) — compute once.
    tensor = ds.to_tensor().compute()
    var_names = [str(v) for v in tensor["variable"].values]
    time_coord = tensor[ds._time_id].values
    entity_coord = tensor[ds._entity_id].values

    # Read time/entity columns to build the (time, entity) → value grid.
    time_col = (
        pq.read_table(synthetic_cm_parquet_small, columns=["month_id"])
        .column("month_id")
        .to_numpy()
        .astype(np.int64)
    )
    entity_col = (
        pq.read_table(synthetic_cm_parquet_small, columns=["country_id"])
        .column("country_id")
        .to_numpy()
        .astype(np.int64)
    )
    time_to_idx = {int(v): i for i, v in enumerate(time_coord)}
    entity_to_idx = {int(v): i for i, v in enumerate(entity_coord)}
    t, e = len(time_coord), len(entity_coord)

    for col in targets + features:
        direct = (
            pq.read_table(synthetic_cm_parquet_small, columns=[col])
            .column(col)
            .to_numpy()
            .astype(np.float32)
        )
        grid = np.full((t, e), np.nan, dtype=np.float32)
        for i in range(len(direct)):
            ti = time_to_idx[int(time_col[i])]
            ei = entity_to_idx[int(entity_col[i])]
            grid[ti, ei] = direct[i]
        var_idx = var_names.index(col)
        loaded = tensor.values[:, :, 0, var_idx]
        assert loaded.shape == grid.shape, f"{col}: shape mismatch"
        assert np.array_equal(grid, loaded, equal_nan=True), (
            f"{col}: bit parity failed"
        )


def test_darts_bridge_parity(synthetic_cm_parquet_small) -> None:
    """ViewsDataset → Darts TimeSeries → numpy must preserve values bit-for-bit."""
    targets = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
    features = [
        "lr_ged_sb_delta", "lr_ged_ns_delta", "lr_ged_os_delta",
        "lr_splag_1_ged_sb", "lr_splag_1_ged_ns", "lr_splag_1_ged_os",
        "lr_decay_ged_sb_1", "lr_decay_ged_sb_5", "lr_decay_ged_sb_25",
    ]
    ds = ViewsDataset(
        synthetic_cm_parquet_small,
        targets=targets,
        broadcast_features=True,
    )
    # Pick the first 3 entities from the parquet.
    entity_col = (
        pq.read_table(synthetic_cm_parquet_small, columns=["country_id"])
        .column("country_id")
        .to_numpy()
        .astype(np.int64)
    )
    unique_entities = np.unique(entity_col)[:3]
    series_list = ds.to_darts_timeseries(entity_ids=unique_entities.tolist())
    assert len(series_list) == 3, f"Expected 3 series, got {len(series_list)}"

    # Compute the dataset's tensor for parity comparison.
    tensor = ds.to_tensor().compute()
    var_names = [str(v) for v in tensor["variable"].values]
    time_coord = tensor[ds._time_id].values
    entity_coord = tensor[ds._entity_id].values

    value_columns = [*features, *targets]
    col_indices = [var_names.index(c) for c in value_columns]
    for i, entity_id in enumerate(unique_entities):
        ts = series_list[i]
        # The series components are [features..., targets...] (with possible
        # cyclic-encoder extensions when use_cyclic_encoders=True; we don't
        # use them here so the components match value_columns exactly).
        ts_components = [str(c) for c in ts.components]
        assert ts_components == value_columns, (
            f"entity {entity_id}: components mismatch"
        )
        # Build the expected (T, F) slice from the tensor.
        e_idx = list(entity_coord).index(int(entity_id))
        expected_values = tensor.values[:, e_idx, 0, :][:, col_indices]
        ts_values = ts.all_values(copy=False)
        if ts_values.ndim == 3:
            ts_values = ts_values[:, :, 0]
        assert ts_values.shape == expected_values.shape
        # NaN-equal comparison (zero-inflated data has many zeros, no NaNs
        # in this fixture, but use equal_nan=True for safety).
        assert np.array_equal(
            ts_values.astype(np.float32),
            expected_values.astype(np.float32),
            equal_nan=True,
        )


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
    """Full flow: parquet → ViewsDataset → Darts TimeSeries for ALL entities."""
    targets = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
    features = [
        "lr_ged_sb_delta", "lr_ged_ns_delta", "lr_ged_os_delta",
        "lr_splag_1_ged_sb", "lr_splag_1_ged_ns", "lr_splag_1_ged_os",
        "lr_decay_ged_sb_1", "lr_decay_ged_sb_5", "lr_decay_ged_sb_25",
    ]
    ds = ViewsDataset(
        synthetic_cm_parquet_small,
        targets=targets,
        broadcast_features=True,
    )
    series_list = ds.to_darts_timeseries()
    n_entities_expected = ds.num_entities
    assert len(series_list) == n_entities_expected

    total_rows = sum(len(ts) for ts in series_list)
    assert total_rows == ds.num_time_steps * ds.num_entities

    # The series carry features + targets (no cyclic encoders requested).
    expected_cols = len(features) + len(targets)
    for ts in series_list:
        assert len(ts.components) == expected_cols


def test_dataset_fit_and_inverse_parity(synthetic_cm_parquet_small) -> None:
    """End-to-end: fit scalers on training window, generate dummy predictions,
    inverse-transform, and verify the result is a ``{target: PredictionFrame}`` dict
    with the right shape.

    This replaces the old parity test that ran a full Darts model — the new
    dataset's ``ingest_darts_predictions`` is the inverse-transform entry point,
    and the test exercises it with synthetic predictions instead of a trained
    model (faster + deterministic).
    """
    from darts import TimeSeries
    import pandas as pd

    targets = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
    features = ["lr_ged_sb_delta", "lr_splag_1_ged_sb"]
    ds = ViewsDataset(
        synthetic_cm_parquet_small,
        targets=targets,
        broadcast_features=True,
    )
    # Fit scalers on the full window with the user's example target scaler.
    ds.fit_scalers(
        target_scaler="AsinhTransform->MaxAbsScaler",
        feature_scaler="RobustScaler",
        time_ids=list(range(121, 221)),
    )
    assert ds.scalers_fitted is True

    # Build synthetic predictions in the SCALED space (one per entity).
    n_entities = 3
    n_steps = 6
    preds = []
    for eid in range(1, n_entities + 1):
        time = np.arange(221, 221 + n_steps, dtype=np.int64)
        # Scaled predictions are small (AsinhTransform→MaxAbs maps to ~[-1, 1]).
        values = np.full(
            (n_steps, len(targets)), 0.3, dtype=np.float32
        )
        ts = TimeSeries.from_times_and_values(
            times=pd.Index(time),
            values=values,
            columns=targets,
            static_covariates=pd.DataFrame({"country_id": [float(eid)]}),
            freq=1,
        )
        preds.append(ts)

    frames = ds.ingest_darts_predictions(
        preds, apply_inverse=True, clip_negatives=True
    )
    assert set(frames.keys()) == set(targets)
    for tgt, frame in frames.items():
        # 3 entities × 6 steps = 18 rows, 1 sample.
        assert frame.n_rows == n_entities * n_steps
        assert frame.values.shape == (n_entities * n_steps, 1)
        # After inverse + clip, all values must be non-negative.
        assert np.all(frame.values >= 0.0)
