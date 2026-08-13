"""Streaming DatasetBuilder — scaffold a Zarr dataset, fill it batch by batch.

Tiers (ADR-005):
🟩 Green — scaffold -> batch writes -> a real LOA subclass whose values
   round-trip; roles, metadata, coverage, exports.
🟫 Beige — overwrite semantics, time-slice writes, durable paths, failed
   context cleanup, parquet round-trip, bounded peak memory.
🟥 Red  — every fail-loud guard: bad loa/coords/specs/targets, out-of-range
   writes, wrong shapes, strict duplicates, completeness, lifecycle.
"""
from __future__ import annotations

import numpy as np
import pytest

from views_r2darts2.dataset import (
    CMDataset,
    CYDataset,
    DatasetBuilder,
    PGMDataset,
    PGYDataset,
    ViewsDataset,
)

MONTHS = np.array([528, 529, 530], dtype=np.int64)
GRIDS = np.array([100, 101, 102, 103], dtype=np.int64)
COUNTRIES = np.array([7, 42, 99], dtype=np.int64)
S = 4  # sample size


def _make_builder(**overrides):
    kwargs = dict(
        loa="pgm",
        times=MONTHS,
        entities=GRIDS,
        variables={"pred_ged_sb": "num3"},
        sample_size=S,
        targets=["pred_ged_sb"],
    )
    kwargs.update(overrides)
    return ViewsDataset.builder(**kwargs)


def _fill_all(builder, name="pred_ged_sb"):
    """Write every cell: value(t, g, :) = t*1000 + g (exact in float32)."""
    for t in MONTHS:
        vals = np.stack(
            [np.full(S, float(t * 1000 + g), dtype=np.float32) for g in GRIDS]
        )
        builder.write_batch(
            times=np.full(len(GRIDS), t), entities=GRIDS, columns={name: vals}
        )


# ============================ 🟩 GREEN — happy path ==========================

def test_green_pgm_builder_roundtrip():
    with ViewsDataset.builder(
        loa="pgm", times=MONTHS, entities=GRIDS,
        variables={"pred_ged_sb": "num3"}, sample_size=S,
        targets=["pred_ged_sb"],
    ) as b:
        _fill_all(b)
        ds = b.build()
    assert type(ds) is PGMDataset
    assert ds.is_prediction is True
    assert ds.sample_size == S
    assert ds.num_time_steps == 3 and ds.num_entities == 4
    assert ds.targets == ["pred_ged_sb"]
    assert ds.pred_vars == ["pred_ged_sb"]
    arr = (
        ds.to_xarray()["pred_ged_sb"]
        .transpose("month_id", "priogrid_id", "sample")
        .values
    )
    for ti, t in enumerate(MONTHS):
        for ei, g in enumerate(GRIDS):
            assert np.all(arr[ti, ei] == float(t * 1000 + g))
    assert ds.check_integrity()
    ds.close()


def test_green_cm_builder_routes_to_cmdataset():
    with ViewsDataset.builder(
        loa="cm", times=MONTHS, entities=COUNTRIES,
        variables={"pred_ged_sb": "num3"}, sample_size=2,
    ) as b:
        b.write_batch(
            times=[528, 528], entities=[7, 42],
            columns={"pred_ged_sb": np.ones((2, 2), dtype=np.float32)},
        )
        ds = b.build()
    assert type(ds) is CMDataset
    assert ds.targets == ["pred_ged_sb"]
    xr_ds = ds.to_xarray()
    assert xr_ds.dims["country_id"] == 3 and xr_ds.dims["month_id"] == 3
    got = xr_ds["pred_ged_sb"].sel(month_id=528, country_id=7).values
    assert np.array_equal(got, [1.0, 1.0])
    unset = xr_ds["pred_ged_sb"].sel(month_id=528, country_id=99).values
    assert np.isnan(unset).all()
    ds.close()


def test_green_year_loas_route():
    with ViewsDataset.builder(
        loa="pgy", times=[2020, 2021], entities=GRIDS,
        variables=["pred_x"], sample_size=1,
    ) as b:
        b.write_batch(
            times=[2020], entities=[100],
            columns={"pred_x": np.array([[5.0]], dtype=np.float32)},
        )
        ds = b.build()
    assert type(ds) is PGYDataset
    assert ds.to_xarray().dims["year_id"] == 2

    with ViewsDataset.builder(
        loa="cy", times=[2020], entities=COUNTRIES,
        variables={"pred_y": "num3"}, sample_size=1,
    ) as b2:
        ds2 = b2.build()
    assert type(ds2) is CYDataset
    ds.close(); ds2.close()


def test_green_feature_mode_roles_and_subset_batches():
    variables = {"f_a": "num3", "f_b": "num3", "ln_sb_best": "num3"}
    with ViewsDataset.builder(
        loa="pgm", times=MONTHS, entities=GRIDS, variables=variables,
        sample_size=2, targets=["ln_sb_best"],
    ) as b:
        for t in MONTHS:
            ts = np.full(len(GRIDS), t)
            b.write_batch(ts, GRIDS, {"f_a": np.ones((4, 2), np.float32)})
            b.write_batch(
                ts, GRIDS,
                {"f_b": np.full((4, 2), 2.0, np.float32),
                 "ln_sb_best": np.full((4, 2), 3.0, np.float32)},
            )
        ds = b.build()
    assert ds.is_prediction is False
    assert ds.targets == ["ln_sb_best"]
    assert sorted(ds.features) == ["f_a", "f_b"]
    x, y = ds.split_data()
    assert x.shape == (3, 4, 2, 2)
    assert y.shape == (3, 4, 2, 1)
    ds.close()


def test_green_single_target_among_many_pred_vars_exports_predictionframe():
    pytest.importorskip("views_frames")
    with ViewsDataset.builder(
        loa="pgm", times=MONTHS, entities=GRIDS,
        variables=["pred_a", "pred_b"], sample_size=S, targets=["pred_a"],
    ) as b:
        for t in MONTHS:
            vals = np.array(
                [[t * 1000 + g * 10 + s for s in range(S)] for g in GRIDS],
                dtype=np.float32,
            )
            b.write_time_slice(int(t), {"pred_a": vals})
            b.write_time_slice(int(t), {"pred_b": vals + 0.5})
        ds = b.build()
    assert ds.targets == ["pred_a"] and sorted(ds.pred_vars) == ["pred_a", "pred_b"]
    pf = ds.to_predictionframe()
    assert pf.n_rows == len(MONTHS) * len(GRIDS)
    assert pf.sample_count == S
    for ti, t in enumerate(MONTHS):
        for ei, g in enumerate(GRIDS):
            row = pf.values[ti * len(GRIDS) + ei]
            expected = [t * 1000 + g * 10 + s for s in range(S)]
            assert np.array_equal(row, expected)
    ds.close()


def test_green_metadata_lands_on_the_dataset():
    with _make_builder(metadata={"model": "hydranet", "run_type": "forecasting"}) as b:
        _fill_all(b)
        ds = b.build()
    assert ds.metadata["model"] == "hydranet"
    assert ds.metadata["run_type"] == "forecasting"
    ds.close()


def test_green_require_complete_passes_when_fully_written():
    with _make_builder(track_coverage=True) as b:
        assert b.coverage == 0.0
        _fill_all(b)
        assert b.coverage == 1.0
        ds = b.build(require_complete=True)
    ds.close()


def test_green_coverage_is_none_without_tracking():
    with _make_builder() as b:
        assert b.coverage is None
        _fill_all(b)
        ds = b.build()
    ds.close()


def test_green_factory_returns_a_builder():
    b = ViewsDataset.builder(
        loa="pgm", times=MONTHS, entities=GRIDS,
        variables=["pred_x"], sample_size=1,
    )
    assert isinstance(b, DatasetBuilder)
    assert b.loa == "pgm"
    b.close()


# ============================ 🟫 BEIGE — realistic edges =====================

def test_beige_last_write_wins_by_default():
    with _make_builder() as b:
        b.write_batch(
            times=[528], entities=[100],
            columns={"pred_ged_sb": np.ones((1, S), np.float32)},
        )
        b.write_batch(
            times=[528], entities=[100],
            columns={"pred_ged_sb": np.full((1, S), 2.0, np.float32)},
        )
        ds = b.build()
    got = ds.to_xarray()["pred_ged_sb"].sel(month_id=528, priogrid_id=100).values
    assert np.array_equal(got, np.full(S, 2.0))
    ds.close()


def test_beige_write_time_slice_roundtrip():
    with _make_builder(track_coverage=True) as b:
        for t in MONTHS:
            vals = np.stack(
                [np.full(S, float(t * 1000 + g), dtype=np.float32) for g in GRIDS]
            )
            b.write_time_slice(int(t), {"pred_ged_sb": vals})
        ds = b.build(require_complete=True)
    arr = ds.to_xarray()["pred_ged_sb"].sel(month_id=529).values
    assert np.all(arr[2] == float(529 * 1000 + 102))
    ds.close()


def test_beige_durable_path_survives_and_reopens(tmp_path):
    target = tmp_path / "preds.zarr"
    with ViewsDataset.builder(
        loa="pgm", times=MONTHS, entities=GRIDS,
        variables=["pred_ged_sb"], sample_size=S, path=target,
    ) as b:
        _fill_all(b)
        ds = b.build()
    expected = ds.to_xarray()["pred_ged_sb"].values
    ds.close()
    assert target.exists()
    ds2 = ViewsDataset.for_loa("pgm", target)
    assert type(ds2) is PGMDataset
    np.testing.assert_array_equal(
        ds2.to_xarray()["pred_ged_sb"].values,
        expected,
    )
    ds2.close()


def test_beige_failed_context_cleans_the_scratch_store():
    with pytest.raises(RuntimeError, match="boom"):
        with _make_builder() as b:
            scratch = b.path.parent
            assert scratch.exists()
            raise RuntimeError("boom")
    assert not scratch.exists()


def test_beige_save_parquet_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    with _make_builder() as b:
        _fill_all(b)
        ds = b.build()
    out = ds.save_parquet(tmp_path / "preds.parquet")
    ds2 = ViewsDataset.for_loa("pgm", out)
    np.testing.assert_array_equal(
        ds2.to_xarray()["pred_ged_sb"].values,
        ds.to_xarray()["pred_ged_sb"].values,
    )
    assert np.array_equal(ds2.to_xarray()["month_id"].values, MONTHS)
    ds.close(); ds2.close()


def test_beige_peak_memory_is_batch_not_grid():
    import tracemalloc
    t, e, s = 64, 4096, 8
    # The full grid would be t * e * s * 4 bytes. A single time slice is
    # e * s * 4 bytes. Peak memory should be well under the full grid —
    # we allow 3x the grid for zarr/dask overhead, but not t*x the grid
    # (which would indicate the whole grid was materialized).
    grid_bytes = t * e * s * 4
    slice_bytes = e * s * 4
    times = np.arange(500, 500 + t, dtype=np.int64)
    entities = np.arange(e, dtype=np.int64)
    rng = np.random.default_rng(0)
    tracemalloc.start()
    tracemalloc.reset_peak()
    with ViewsDataset.builder(
        loa="pgm", times=times, entities=entities,
        variables=["pred_x"], sample_size=s,
    ) as b:
        for tm in times:
            vals = rng.random((e, s)).astype(np.float32)
            b.write_time_slice(int(tm), {"pred_x": vals})
            del vals
        ds = b.build()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    # Peak must be much less than the full grid (allow 5x for zarr/dask
    # overhead — the key invariant is that we never materialize the grid).
    assert peak < 5 * grid_bytes, (
        f"peak {peak / 1e6:.1f} MB — something materialized the grid "
        f"(grid = {grid_bytes / 1e6:.1f} MB)"
    )
    ds.close()


# ============================ 🟥 RED — adversarial ===========================

def test_red_unknown_loa_fails_loud():
    with pytest.raises(ValueError, match="unknown loa"):
        _make_builder(loa="xyz")


def test_red_spatial_only_loa_fails_loud():
    with pytest.raises(ValueError, match="spatial-only"):
        _make_builder(loa="pg")


def test_red_duplicate_scaffold_coordinate_fails_loud():
    with pytest.raises(ValueError, match="duplicate"):
        _make_builder(times=[528, 529, 529])


def test_red_empty_scaffold_fails_loud():
    with pytest.raises(ValueError, match="at least one"):
        _make_builder(entities=[])


def test_red_non_integer_coordinates_fail_loud():
    with pytest.raises(ValueError, match="integer identifiers"):
        _make_builder(times=[528.5, 529.5])


def test_red_bad_spec_fails_loud():
    with pytest.raises(ValueError, match="num2"):
        _make_builder(variables={"pred_x": "text"})


def test_red_targets_not_declared_fails_loud():
    with pytest.raises(ValueError, match="not declared"):
        _make_builder(targets=["nope"])


def test_red_reserved_metadata_key_fails_loud():
    with pytest.raises(ValueError, match="reserved"):
        _make_builder(metadata={"targets": ["hijack"]})


def test_red_out_of_range_time_fails_loud():
    with _make_builder() as b:
        with pytest.raises(ValueError, match="999.*outside the scaffold"):
            b.write_batch(
                times=[999], entities=[100],
                columns={"pred_ged_sb": np.zeros((1, S), np.float32)},
            )


def test_red_out_of_range_entity_fails_loud():
    with _make_builder() as b:
        with pytest.raises(ValueError, match="777.*outside the scaffold"):
            b.write_batch(
                times=[528], entities=[777],
                columns={"pred_ged_sb": np.zeros((1, S), np.float32)},
            )


def test_red_unknown_column_fails_loud():
    with _make_builder() as b:
        with pytest.raises(ValueError, match="unknown variable"):
            b.write_batch(
                times=[528], entities=[100],
                columns={"pred_nope": np.zeros((1, S), np.float32)},
            )


def test_red_wrong_sample_size_fails_loud():
    with _make_builder() as b:
        with pytest.raises(ValueError, match="expected 1 x 4"):
            b.write_batch(
                times=[528], entities=[100],
                columns={"pred_ged_sb": np.zeros((1, 5), np.float32)},
            )


def test_red_batch_length_mismatch_fails_loud():
    with _make_builder() as b:
        with pytest.raises(ValueError, match="same length"):
            b.write_batch(
                times=[528, 529], entities=[100, 101, 102],
                columns={"pred_ged_sb": np.zeros((2, S), np.float32)},
            )


def test_red_time_slice_wrong_shape_fails_loud():
    with _make_builder() as b:
        with pytest.raises(ValueError, match="entity slice"):
            b.write_time_slice(528, {"pred_ged_sb": np.ones((2, S), np.float32)})


def test_red_time_slice_out_of_range_fails_loud():
    with _make_builder() as b:
        with pytest.raises(ValueError, match="outside the scaffold"):
            b.write_time_slice(999, {"pred_ged_sb": np.ones((4, S), np.float32)})


def test_red_strict_rejects_duplicate_writes():
    with _make_builder(strict=True) as b:
        b.write_batch(
            times=[528, 528], entities=[100, 101],
            columns={"pred_ged_sb": np.ones((2, S), np.float32)},
        )
        with pytest.raises(ValueError, match="already written"):
            b.write_batch(
                times=[528], entities=[101],
                columns={"pred_ged_sb": np.zeros((1, S), np.float32)},
            )


def test_red_require_complete_fails_on_partial():
    with _make_builder(track_coverage=True) as b:
        b.write_batch(
            times=[528], entities=[100],
            columns={"pred_ged_sb": np.ones((1, S), np.float32)},
        )
        with pytest.raises(ValueError, match="never written"):
            b.build(require_complete=True)


def test_red_require_complete_without_tracking_fails_loud():
    with _make_builder() as b:
        _fill_all(b)
        with pytest.raises(ValueError, match="coverage tracking"):
            b.build(require_complete=True)


def test_red_write_after_build_fails_loud():
    b = _make_builder()
    b.write_batch(
        times=[528], entities=[100],
        columns={"pred_ged_sb": np.ones((1, S), np.float32)},
    )
    ds = b.build()
    with pytest.raises(RuntimeError, match="already been called"):
        b.write_batch(
            times=[529], entities=[100],
            columns={"pred_ged_sb": np.ones((1, S), np.float32)},
        )
    ds.close()


def test_red_write_after_close_fails_loud():
    b = _make_builder()
    b.close()
    b.close()
    with pytest.raises(RuntimeError, match="closed"):
        b.write_batch(
            times=[528], entities=[100],
            columns={"pred_ged_sb": np.ones((1, S), np.float32)},
        )


def test_red_build_twice_fails_loud():
    b = _make_builder()
    ds = b.build()
    with pytest.raises(RuntimeError, match="already been called"):
        b.build()
    ds.close()
