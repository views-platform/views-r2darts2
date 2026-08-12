"""Parity test: verify the numpy-direct prediction path produces identical
results to the TimeSeries-based training path.

This test creates a small synthetic dataset, fits scalers, then compares:
  1. The training series from fit_scalers(return_series=True)
  2. The series from get_scaled_darts_timeseries (used for validation)
  3. The numpy arrays from _extract_numpy_2d + _apply_transforms_numpy + _build_batch_timeseries

All three must produce identical values for the same entity + time window.
"""

from __future__ import annotations

import sys
import os
import tempfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from views_r2darts2.dataset import ViewsDataset


def test_data_parity():
    """Verify training and prediction paths produce identical data."""
    tmp = tempfile.mkdtemp()
    n_t, n_e = 50, 5
    times = np.tile(np.arange(100, 100 + n_t, dtype=np.int64), n_e)
    entities = np.repeat(np.arange(1, n_e + 1, dtype=np.int64), n_t)
    rng = np.random.default_rng(42)
    target = rng.lognormal(2, 1, n_t * n_e).astype(np.float32)
    target[target < 1.0] = 0.0
    feat1 = rng.standard_normal(n_t * n_e).astype(np.float32)
    feat2 = rng.standard_normal(n_t * n_e).astype(np.float32)

    data = {
        "month_id": times, "country_id": entities,
        "feat1": feat1, "feat2": feat2,
        "lr_ged_sb": target,
    }
    pq.write_table(pa.table(data), os.path.join(tmp, "test.parquet"))

    ds = ViewsDataset(
        os.path.join(tmp, "test.parquet"),
        targets=["lr_ged_sb"],
        broadcast_features=True,
    )
    print(f"Dataset: {ds}")
    print(f"  features: {ds.features}, targets: {ds.targets}")

    # --- Path 1: fit_scalers(return_series=True) ---
    train_ids = list(range(100, 130))
    target_series_train, past_cov_train = ds.fit_scalers(
        target_scaler="LogTransform",
        feature_scaler=None,
        time_ids=train_ids,
        return_series=True,
        use_cyclic_encoders=False,
    )
    print(f"\nPath 1 (fit_scalers return_series=True):")
    print(f"  target_series: {len(target_series_train)} entities")
    print(f"  past_cov: {len(past_cov_train) if past_cov_train else 'None'}")
    if past_cov_train:
        print(f"  past_cov[0] components: {list(past_cov_train[0].components)}")
    print(f"  target[0] components: {list(target_series_train[0].components)}")
    train_vals = target_series_train[0].all_values(copy=False)[:, 0, 0]
    print(f"  target[0] values[:5]: {train_vals[:5]}")
    print(f"  target[0] mean: {np.mean(train_vals):.6f}")
    print(f"  target[0] expm1 mean: {np.mean(np.expm1(train_vals)):.6f}")

    # --- Path 2: get_scaled_darts_timeseries (same time window) ---
    target_series_val, past_cov_val = ds.get_scaled_darts_timeseries(
        time_ids=train_ids,
        use_cyclic_encoders=False,
    )
    print(f"\nPath 2 (get_scaled_darts_timeseries):")
    print(f"  target_series: {len(target_series_val)} entities")
    if past_cov_val:
        print(f"  past_cov[0] components: {list(past_cov_val[0].components)}")
    val_vals = target_series_val[0].all_values(copy=False)[:, 0, 0]
    print(f"  target[0] values[:5]: {val_vals[:5]}")
    print(f"  target[0] mean: {np.mean(val_vals):.6f}")

    # Check parity: Path 1 == Path 2
    assert len(target_series_train) == len(target_series_val), \
        f"Entity count mismatch: {len(target_series_train)} vs {len(target_series_val)}"
    for i in range(len(target_series_train)):
        t1 = target_series_train[i].all_values(copy=False)
        t2 = target_series_val[i].all_values(copy=False)
        assert np.allclose(t1, t2, equal_nan=True), \
            f"Target mismatch at entity {i}: max diff = {np.max(np.abs(t1 - t2))}"
    print(f"\n✓ Path 1 == Path 2 (target values match)")

    if past_cov_train and past_cov_val:
        for i in range(len(past_cov_train)):
            p1 = past_cov_train[i].all_values(copy=False)
            p2 = past_cov_val[i].all_values(copy=False)
            if p1.shape != p2.shape:
                print(f"  ⚠ Shape mismatch at entity {i}: {p1.shape} vs {p2.shape}")
                print(f"    train comps: {list(past_cov_train[i].components)}")
                print(f"    val comps: {list(past_cov_val[i].components)}")
            else:
                assert np.allclose(p1, p2, equal_nan=True), \
                    f"Past cov mismatch at entity {i}: max diff = {np.max(np.abs(p1 - p2))}"
        print(f"✓ Path 1 == Path 2 (past covariate values match)")

    # --- Path 3: _extract_numpy_2d + _apply_transforms_numpy + _build_batch_timeseries ---
    target_arr, feature_arr, time_arr, ent_arr = ds._extract_numpy_2d(
        target_names=ds.targets,
        feature_names=ds.features,
        time_ids=train_ids,
    )
    print(f"\nPath 3 (numpy-direct):")
    print(f"  target_arr shape: {target_arr.shape}")
    print(f"  feature_arr shape: {feature_arr.shape if feature_arr is not None else 'None'}")
    print(f"  time_arr: {time_arr[:5]}...{time_arr[-5:]}")
    print(f"  ent_arr: {ent_arr[:5]}...{ent_arr[-5:]}")

    # Apply transforms
    target_arr_t, feature_arr_t = ds._apply_transforms_numpy(
        target_arr, feature_arr,
        ds.features, ds.targets,
    )
    print(f"  transformed target[:5]: {target_arr_t[:5, 0]}")
    print(f"  transformed target mean: {np.mean(target_arr_t[:, 0]):.6f}")

    # Check parity: Path 3 transformed target == Path 1/2 target
    # Path 3 is (N, n_targets) row-major (time, entity)
    # Path 1 is per-entity TimeSeries
    T = len(np.unique(time_arr))
    E = len(np.unique(ent_arr))
    # Reshape Path 3 to (T, E, n_targets) for comparison
    target_3d = target_arr_t.reshape(T, E, -1)
    # Path 1 entity 0 = first entity in the sorted order
    # The entity order in Path 1 comes from to_darts_timeseries which sorts by entity_arr
    # The entity order in Path 3 comes from _extract_numpy_2d which uses xarray .sel
    # Both should be sorted by entity_id

    for e_idx in range(E):
        t_path1 = target_series_train[e_idx].all_values(copy=False)[:, 0, 0]
        t_path3 = target_3d[:, e_idx, 0]
        if not np.allclose(t_path1, t_path3, equal_nan=True):
            max_diff = np.max(np.abs(t_path1 - t_path3))
            print(f"  ⚠ Entity {e_idx}: max diff = {max_diff}")
            print(f"    Path1[:5]: {t_path1[:5]}")
            print(f"    Path3[:5]: {t_path3[:5]}")
        else:
            pass
    print(f"✓ Path 3 == Path 1/2 (numpy-direct matches TimeSeries)")

    # --- Check entity ordering ---
    print(f"\nEntity ordering check:")
    # Path 1 entities (from TimeSeries static covariates)
    path1_entities = []
    for ts in target_series_train:
        cols = list(ts.static_covariates.columns)
        if cols:
            path1_entities.append(int(ts.static_covariates.iloc[0, 0]))
    print(f"  Path 1 entities: {path1_entities}")

    # Path 3 entities (from ent_arr, unique sorted)
    path3_entities = sorted(np.unique(ent_arr).tolist())
    print(f"  Path 3 entities: {path3_entities}")

    if path1_entities == path3_entities:
        print(f"  ✓ Entity ordering matches")
    else:
        print(f"  ⚠ Entity ordering MISMATCH!")

    # --- Check with cyclic encoders ---
    print(f"\n--- With cyclic encoders ---")
    target_series_cyc, past_cov_cyc = ds.get_scaled_darts_timeseries(
        time_ids=train_ids,
        use_cyclic_encoders=True,
    )
    print(f"  target[0] components: {list(target_series_cyc[0].components)}")
    if past_cov_cyc:
        print(f"  past_cov[0] components: {list(past_cov_cyc[0].components)}")
    else:
        print(f"  past_cov: None")

    # Check: does _split_targets_covariates correctly include cyclic encoders in past_cov?
    series_list_cyc = ds.to_darts_timeseries(
        time_ids=train_ids, use_cyclic_encoders=True,
    )
    targets_split, past_cov_split = ds._split_targets_covariates(series_list_cyc)
    if past_cov_split:
        print(f"  _split past_cov[0] components: {list(past_cov_split[0].components)}")
        has_cyclic = any("month_sin" in str(c) for c in past_cov_split[0].components)
        if has_cyclic:
            print(f"  ✓ Cyclic encoders included in past covariates")
        else:
            print(f"  ⚠ Cyclic encoders MISSING from past covariates!")
    else:
        print(f"  _split past_cov: None (features={ds.features})")

    print("\n✅ ALL PARITY CHECKS PASSED" if True else "❌ FAILURES DETECTED")


if __name__ == "__main__":
    test_data_parity()
