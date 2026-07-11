"""Tests for :class:`views_r2darts2.transformers.feature_scaler_manager.FeatureScalerManager`.

Covers config parsing (simple + named-group formats), default-scaler
assignment, fit/transform/inverse round-trip, error paths, and the user's
full 82-feature conflict-feature mapping (synthetic 3-entity round-trip).

Google Python Style. ``pandas`` is used only at the Darts boundary (for
``pd.Index``/``pd.DataFrame`` construction in :class:`TimeSeries`).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries

from views_r2darts2.transformers.feature_scaler_manager import (
    FeatureScalerManager,
)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# The user's 82-feature conflict-feature mapping (single scaler chain).
# Built up explicitly from the 87-column parquet vocabulary.
_CONFLICT_FEATURES: list[str] = [
    # 3 delta features
    "lr_ged_sb_delta",
    "lr_ged_ns_delta",
    "lr_ged_os_delta",
    # 4 acled features
    "lr_acled_sb",
    "lr_acled_sb_count",
    "lr_acled_os",
    "lr_acled_ns",
    # 3 splag features
    "lr_splag_1_ged_sb",
    "lr_splag_1_ged_ns",
    "lr_splag_1_ged_os",
    # 5 decay_ged_sb
    "lr_decay_ged_sb_1",
    "lr_decay_ged_sb_5",
    "lr_decay_ged_sb_25",
    "lr_decay_ged_sb_100",
    "lr_decay_ged_sb_500",
    # 5 decay_ged_os
    "lr_decay_ged_os_1",
    "lr_decay_ged_os_5",
    "lr_decay_ged_os_25",
    "lr_decay_ged_os_100",
    "lr_decay_ged_os_500",
    # 5 decay_ged_ns
    "lr_decay_ged_ns_1",
    "lr_decay_ged_ns_5",
    "lr_decay_ged_ns_25",
    "lr_decay_ged_ns_100",
    "lr_decay_ged_ns_500",
    # 3 decay_acled
    "lr_decay_acled_sb_5",
    "lr_decay_acled_os_5",
    "lr_decay_acled_ns_5",
    # 3 splag_decay
    "lr_splag_1_decay_ged_sb_5",
    "lr_splag_1_decay_ged_os_5",
    "lr_splag_1_decay_ged_ns_5",
    # 6 tlag_ged_sb
    "lr_ged_sb_tlag_1",
    "lr_ged_sb_tlag_2",
    "lr_ged_sb_tlag_3",
    "lr_ged_sb_tlag_4",
    "lr_ged_sb_tlag_5",
    "lr_ged_sb_tlag_6",
    # 6 tlag_ged_ns
    "lr_ged_ns_tlag_1",
    "lr_ged_ns_tlag_2",
    "lr_ged_ns_tlag_3",
    "lr_ged_ns_tlag_4",
    "lr_ged_ns_tlag_5",
    "lr_ged_ns_tlag_6",
    # 6 tlag_ged_os
    "lr_ged_os_tlag_1",
    "lr_ged_os_tlag_2",
    "lr_ged_os_tlag_3",
    "lr_ged_os_tlag_4",
    "lr_ged_os_tlag_5",
    "lr_ged_os_tlag_6",
    # 3 tsum_24
    "lr_ged_sb_tsum_24",
    "lr_ged_ns_tsum_24",
    "lr_ged_os_tsum_24",
    # 2 topic_tokens
    "lr_topic_tokens_t1",
    "lr_topic_tokens_t2",
    # 3 topic_ste_theta4_stock
    "lr_topic_ste_theta4_stock_t1",
    "lr_topic_ste_theta4_stock_t2",
    "lr_topic_ste_theta4_stock_t13",
    # 3 topic_ste_theta2_stock
    "lr_topic_ste_theta2_stock_t1",
    "lr_topic_ste_theta2_stock_t2",
    "lr_topic_ste_theta2_stock_t13",
    # 2 topic splag
    "lr_topic_ste_theta4_stock_t1_splag",
    "lr_topic_ste_theta2_stock_t1_splag",
    # 8 wdi features
    "lr_wdi_sm_pop_refg_or",
    "lr_wdi_sm_pop_netm",
    "lr_wdi_dt_oda_odat_pc_zs",
    "lr_wdi_ms_mil_xpnd_gd_zs",
    "lr_wdi_sp_pop_grow",
    "lr_wdi_sp_urb_totl_in_zs",
    "lr_wdi_sp_dyn_imrt_fe_in",
    "lr_wdi_sh_sta_maln_zs",
    # 12 vdem features
    "lr_vdem_v2x_horacc",
    "lr_vdem_v2x_veracc",
    "lr_vdem_v2xnp_client",
    "lr_vdem_v2xnp_regcorr",
    "lr_vdem_v2xpe_exlgeo",
    "lr_vdem_v2xpe_exlsocgr",
    "lr_vdem_v2x_ex_party",
    "lr_vdem_v2x_ex_military",
    "lr_vdem_v2xeg_eqdr",
    "lr_vdem_v2xcl_prpty",
    "lr_vdem_v2xcl_dmove",
    "lr_vdem_v2x_clphy",
]
assert len(_CONFLICT_FEATURES) == 82, (
    f"Expected 82 conflict features, got {len(_CONFLICT_FEATURES)}"
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def build_series_list(
    *,
    n_entities: int = 3,
    n_time: int = 20,
    columns: Sequence[str] = ("a", "b", "c"),
    seed: int | None = None,
    positive_offset: float = 0.0,
) -> list[TimeSeries]:
    """Build a list of per-entity Darts :class:`TimeSeries`.

    Args:
        n_entities: Number of synthetic entities.
        n_time: Timesteps per entity.
        columns: Component (column) names.
        seed: Optional RNG seed for reproducibility.
        positive_offset: Added to every value (use > 0 to keep non-negative).

    Returns:
        A list of ``n_entities`` TimeSeries, each of length ``n_time``.
    """
    rng = np.random.default_rng(seed)
    cols = list(columns)
    series_list: list[TimeSeries] = []
    time_arr = np.arange(1, n_time + 1, dtype=np.int64)
    for entity_id in range(1, n_entities + 1):
        values = (
            rng.standard_normal((n_time, len(cols))).astype(np.float32)
            + np.float32(positive_offset)
        )
        ts = TimeSeries.from_times_and_values(
            times=pd.Index(time_arr),
            values=values.astype(np.float32),
            columns=cols,
            static_covariates=pd.DataFrame({"country_id": [float(entity_id)]}),
            freq=1,
        )
        series_list.append(ts)
    return series_list


def build_probabilistic_series_list(
    *,
    n_entities: int = 2,
    n_time: int = 12,
    n_samples: int = 5,
    columns: Sequence[str] = ("a", "b"),
    seed: int | None = None,
) -> list[TimeSeries]:
    """Build a list of per-entity probabilistic (3-D) Darts TimeSeries."""
    rng = np.random.default_rng(seed)
    cols = list(columns)
    series_list: list[TimeSeries] = []
    time_arr = np.arange(1, n_time + 1, dtype=np.int64)
    for entity_id in range(1, n_entities + 1):
        values = rng.standard_normal(
            (n_time, len(cols), n_samples)
        ).astype(np.float32)
        ts = TimeSeries.from_times_and_values(
            times=pd.Index(time_arr),
            values=values,
            columns=cols,
            static_covariates=pd.DataFrame({"country_id": [float(entity_id)]}),
            freq=1,
        )
        series_list.append(ts)
    return series_list


def extract_values_2d(series_list: list[TimeSeries]) -> np.ndarray:
    """Concatenate the (T, F, 1) values of a list of series to 2-D (N, F)."""
    chunks = [ts.all_values(copy=False)[:, :, 0] for ts in series_list]
    return np.concatenate(chunks, axis=0)


# ----------------------------------------------------------------------
# Config parsing tests
# ----------------------------------------------------------------------


class TestFeatureScalerManagerParsing:
    """Config-format parsing tests."""

    def test_simple_format_parsing(self) -> None:
        """``{scaler_name: [features]}`` is the simple format."""
        mgr = FeatureScalerManager(
            feature_scaler_map={"MaxAbsScaler": ["a", "b"]},
            default_scaler=None,
        )
        mapping = mgr.get_feature_scaler_mapping()
        assert mapping == {"a": "scaler_MaxAbsScaler", "b": "scaler_MaxAbsScaler"}

    def test_named_group_format_parsing(self) -> None:
        """``{group_name: {"scaler": ..., "features": [...]}}`` is the named format."""
        mgr = FeatureScalerManager(
            feature_scaler_map={
                "group_a": {"scaler": "MaxAbsScaler", "features": ["a", "b"]},
                "group_b": {"scaler": "StandardScaler", "features": ["c"]},
            },
            default_scaler=None,
        )
        mapping = mgr.get_feature_scaler_mapping()
        assert mapping == {
            "a": "group_group_a",
            "b": "group_group_a",
            "c": "group_group_b",
        }

    def test_default_scaler_assignment(self) -> None:
        """Unmapped features in ``all_features`` get the default scaler."""
        mgr = FeatureScalerManager(
            feature_scaler_map={"MaxAbsScaler": ["a"]},
            default_scaler="RobustScaler",
            all_features=["a", "b", "c"],
        )
        mapping = mgr.get_feature_scaler_mapping()
        assert mapping["a"] == "scaler_MaxAbsScaler"
        assert mapping["b"] == "default"
        assert mapping["c"] == "default"

    def test_no_default_scaler_when_all_mapped(self) -> None:
        """When every feature is mapped, no default scaler is created."""
        mgr = FeatureScalerManager(
            feature_scaler_map={"MaxAbsScaler": ["a", "b"]},
            default_scaler="RobustScaler",
            all_features=["a", "b"],
        )
        mapping = mgr.get_feature_scaler_mapping()
        assert mapping == {
            "a": "scaler_MaxAbsScaler",
            "b": "scaler_MaxAbsScaler",
        }
        # No default scaler entry.
        assert "default" not in {v for v in mapping.values()}

    def test_duplicate_feature_raises(self) -> None:
        """A feature in two groups raises ``ValueError``."""
        with pytest.raises(ValueError, match="multiple"):
            FeatureScalerManager(
                feature_scaler_map={
                    "group_a": {"scaler": "MaxAbsScaler", "features": ["a"]},
                    "group_b": {"scaler": "StandardScaler", "features": ["a"]},
                },
                default_scaler=None,
            )

    def test_none_scaler_config_raises(self) -> None:
        """A ``None`` scaler config inside a group raises ``ValueError``."""
        with pytest.raises(ValueError, match="cannot be None"):
            FeatureScalerManager(
                feature_scaler_map={
                    "group_a": {"scaler": None, "features": ["a"]},
                },
                default_scaler=None,
            )

    def test_unrecognized_format_raises(self) -> None:
        """An entry whose value is neither list nor dict-with-features raises."""
        with pytest.raises(ValueError, match="Unrecognized"):
            FeatureScalerManager(
                feature_scaler_map={"MaxAbsScaler": 42},  # type: ignore[dict-item]
                default_scaler=None,
            )


# ----------------------------------------------------------------------
# Fit / transform / inverse round-trip tests
# ----------------------------------------------------------------------


class TestFeatureScalerManagerFitTransform:
    """Fit/transform/inverse-transform tests."""

    def test_fit_transform_then_transform(self) -> None:
        """``fit_transform`` then ``transform`` produces identical results."""
        series_list = build_series_list(
            n_entities=3, n_time=20, columns=["a", "b", "c"], seed=0
        )
        mgr = FeatureScalerManager(
            feature_scaler_map={"MaxAbsScaler": ["a", "b", "c"]},
            default_scaler=None,
        )
        transformed_once = mgr.fit_transform(series_list)
        transformed_twice = mgr.transform(series_list)
        assert len(transformed_once) == len(transformed_twice) == 3
        for ts1, ts2 in zip(transformed_once, transformed_twice):
            np.testing.assert_allclose(
                ts1.all_values(copy=False),
                ts2.all_values(copy=False),
                rtol=1e-6,
                atol=1e-6,
            )

    def test_transform_before_fit_raises(self) -> None:
        """``transform`` before ``fit_transform`` raises ``RuntimeError``."""
        series_list = build_series_list(columns=["a"], seed=0)
        mgr = FeatureScalerManager(
            feature_scaler_map={"MaxAbsScaler": ["a"]},
            default_scaler=None,
        )
        with pytest.raises(RuntimeError, match="not fitted"):
            mgr.transform(series_list)

    def test_inverse_transform(self) -> None:
        """Round-trip: fit_transform → inverse_transform recovers values."""
        series_list = build_series_list(
            n_entities=3, n_time=20, columns=["a", "b", "c"], seed=1
        )
        mgr = FeatureScalerManager(
            feature_scaler_map={"MaxAbsScaler": ["a", "b", "c"]},
            default_scaler=None,
        )
        transformed = mgr.fit_transform(series_list)
        recovered = mgr.inverse_transform(transformed)
        original_2d = extract_values_2d(series_list)
        recovered_2d = extract_values_2d(recovered)
        np.testing.assert_allclose(recovered_2d, original_2d, rtol=1e-5, atol=1e-5)

    def test_inverse_transform_before_fit_raises(self) -> None:
        """``inverse_transform`` before ``fit_transform`` raises ``RuntimeError``."""
        series_list = build_series_list(columns=["a"], seed=0)
        mgr = FeatureScalerManager(
            feature_scaler_map={"MaxAbsScaler": ["a"]},
            default_scaler=None,
        )
        with pytest.raises(RuntimeError, match="not fitted"):
            mgr.inverse_transform(series_list)

    def test_is_fitted_property(self) -> None:
        """``is_fitted`` is False before fit, True after."""
        series_list = build_series_list(columns=["a"], seed=0)
        mgr = FeatureScalerManager(
            feature_scaler_map={"MaxAbsScaler": ["a"]},
            default_scaler=None,
        )
        assert mgr.is_fitted is False
        mgr.fit_transform(series_list)
        assert mgr.is_fitted is True

    def test_fit_transform_with_no_scalers(self) -> None:
        """An empty map returns the input series unchanged."""
        series_list = build_series_list(columns=["a"], seed=0)
        mgr = FeatureScalerManager(feature_scaler_map={}, default_scaler=None)
        result = mgr.fit_transform(series_list)
        assert result is series_list  # passthrough — same list object

    def test_get_feature_scaler_mapping_returns_copy(self) -> None:
        """Mutating the returned mapping does not affect the manager."""
        mgr = FeatureScalerManager(
            feature_scaler_map={"MaxAbsScaler": ["a", "b"]},
            default_scaler=None,
        )
        mapping = mgr.get_feature_scaler_mapping()
        mapping["a"] = "mutated"
        mapping2 = mgr.get_feature_scaler_mapping()
        assert mapping2["a"] == "scaler_MaxAbsScaler"

    def test_repr(self) -> None:
        """``__repr__`` includes the scaler-key → feature-count mapping."""
        mgr = FeatureScalerManager(
            feature_scaler_map={"MaxAbsScaler": ["a", "b"]},
            default_scaler=None,
        )
        text = repr(mgr)
        assert "FeatureScalerManager" in text
        assert "2 features" in text
        assert "fitted=False" in text


# ----------------------------------------------------------------------
# Chained scaler tests
# ----------------------------------------------------------------------


class TestFeatureScalerManagerChained:
    """Chained (Pipeline) scaler tests."""

    def test_chained_scaler_simple_format(self) -> None:
        """``AsinhTransform->MaxAbsScaler`` chain via simple format parses."""
        mgr = FeatureScalerManager(
            feature_scaler_map={
                "AsinhTransform->MaxAbsScaler": ["a", "b"]
            },
            default_scaler=None,
        )
        mapping = mgr.get_feature_scaler_mapping()
        assert set(mapping.keys()) == {"a", "b"}

    def test_chained_inverse_transform(self) -> None:
        """Round-trip with ``AsinhTransform->MaxAbsScaler`` chain."""
        # Use positive-offset synthetic data for asinh stability.
        series_list = build_series_list(
            n_entities=3,
            n_time=20,
            columns=["a", "b"],
            seed=2,
            positive_offset=5.0,
        )
        mgr = FeatureScalerManager(
            feature_scaler_map={
                "AsinhTransform->MaxAbsScaler": ["a", "b"]
            },
            default_scaler=None,
        )
        transformed = mgr.fit_transform(series_list)
        recovered = mgr.inverse_transform(transformed)
        original_2d = extract_values_2d(series_list)
        recovered_2d = extract_values_2d(recovered)
        np.testing.assert_allclose(recovered_2d, original_2d, rtol=1e-4, atol=1e-4)


# ----------------------------------------------------------------------
# User's full 82-feature mapping
# ----------------------------------------------------------------------


class TestFeatureScalerManagerUsersMapping:
    """The user's full 82-feature conflict-feature mapping (round-trip)."""

    def test_users_full_mapping(self) -> None:
        """82-feature ``AsinhTransform->MaxAbsScaler`` mapping round-trips."""
        feature_scaler_map = {
            "AsinhTransform->MaxAbsScaler": list(_CONFLICT_FEATURES)
        }
        # 3 synthetic entities, 20 timesteps, 82 features, positive offset.
        series_list = build_series_list(
            n_entities=3,
            n_time=20,
            columns=_CONFLICT_FEATURES,
            seed=42,
            positive_offset=10.0,
        )
        mgr = FeatureScalerManager(
            feature_scaler_map=feature_scaler_map,
            default_scaler=None,
        )
        # All 82 features must be assigned to one scaler key.
        mapping = mgr.get_feature_scaler_mapping()
        assert len(mapping) == 82
        assert all(v == "scaler_AsinhTransform->MaxAbsScaler" for v in mapping.values())

        transformed = mgr.fit_transform(series_list)
        recovered = mgr.inverse_transform(transformed)
        original_2d = extract_values_2d(series_list)
        recovered_2d = extract_values_2d(recovered)
        np.testing.assert_allclose(recovered_2d, original_2d, rtol=1e-4, atol=1e-3)


# ----------------------------------------------------------------------
# Probabilistic (3-D) tests
# ----------------------------------------------------------------------


class TestFeatureScalerManagerProbabilistic:
    """Probabilistic (3-D ``(T, F, S)``) tests."""

    def test_probabilistic_inverse_transform(self) -> None:
        """3-D (T, F, S) with S>1: sample dim preserved; sample-0 round-trips.

        The production ``fit_transform`` path broadcasts the first sample to
        every sample slot (the scaler is deterministic — same legacy
        contract). So after ``fit_transform → inverse_transform``:
            * the shape ``(T, F, S)`` is preserved;
            * every recovered sample equals ``inverse(transform(original[0]))``;
            * ``recovered[:, :, 0]`` matches ``original[:, :, 0]`` within rtol.
        """
        n_samples = 5
        series_list = build_probabilistic_series_list(
            n_entities=2,
            n_time=12,
            n_samples=n_samples,
            columns=["a", "b"],
            seed=3,
        )
        mgr = FeatureScalerManager(
            feature_scaler_map={"MaxAbsScaler": ["a", "b"]},
            default_scaler=None,
        )
        transformed = mgr.fit_transform(series_list)
        recovered = mgr.inverse_transform(transformed)

        assert len(recovered) == 2
        for ts in recovered:
            arr = ts.all_values(copy=False)
            assert arr.ndim == 3
            assert arr.shape[-1] == n_samples, "Sample dim must be preserved"

        # Per-series: sample 0 must round-trip within float32 tolerance.
        for ts_orig, ts_rec in zip(series_list, recovered):
            orig = ts_orig.all_values(copy=False)
            rec = ts_rec.all_values(copy=False)
            np.testing.assert_allclose(
                rec[:, :, 0], orig[:, :, 0], rtol=1e-5, atol=1e-5
            )
            # The broadcast contract: all samples equal sample 0 after the
            # round-trip (forward broadcast → inverse returns identical
            # values per sample).
            for s in range(1, n_samples):
                np.testing.assert_allclose(
                    rec[:, :, s], rec[:, :, 0], rtol=1e-6, atol=1e-6
                )
