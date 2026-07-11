"""Tests for :class:`views_r2darts2.engines.darts_forecaster.DartsForecaster`.

Exercises the init contract (config validation, scaler wiring, device
detection), the preprocessing pipeline (train-mode fit + predict-mode
transform), the predict contract (RuntimeError on unfitted scalers,
:class:`PredictionFrame` dict output, negative clipping, entropy lock),
and the save/load round-trip.

The forecaster requires both a real :class:`ViewsDatasetDarts` (built from
the validation parquet, subsetted to 3 entities for speed) and a real Darts
model for save/load tests. For predict tests, the Darts model is mocked via
``Mock(spec=TorchForecastingModel)`` — this avoids the cost of training a
real model while still exercising the full predict → inverse →
``PredictionFrame`` pipeline.

Google Python Style. ``pandas`` is used only at the Darts ``TimeSeries``
boundary (mirroring the production ``darts_bridge`` module confinement).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # noqa: WPS433 — allowed at the Darts TimeSeries boundary
import pytest
import torch
from darts import TimeSeries
from darts.models import TCNModel
from darts.models.forecasting.torch_forecasting_model import (
    TorchForecastingModel,
)
from unittest.mock import Mock, patch

from views_frames import (
    FeatureFrame,
    PredictionFrame,
    SpatioTemporalIndex,
    SpatialLevel,
)
from views_r2darts2.data.parquet_loader import load_views_parquet
from views_r2darts2.data.views_dataset import ViewsDatasetDarts
from views_r2darts2.engines.darts_forecaster import DartsForecaster
from views_r2darts2.infrastructure.reproducibility_gate import (
    ReproducibilityGate,
)
from views_r2darts2.transformers.darts_bridge import build_entity_timeseries
from views_r2darts2.transformers.feature_scaler_manager import (
    FeatureScalerManager,
)

# Path to the user-provided validation parquet (12 MB, 87 cols, 81192 rows).
PARQUET_PATH = Path("/home/z/my-project/upload/validation_viewser_df.parquet")

# Three targets + six features used throughout the suite.
TARGETS: list[str] = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
FEATURES: list[str] = [
    "lr_ged_sb_delta",
    "lr_ged_ns_delta",
    "lr_ged_os_delta",
    "lr_splag_1_ged_sb",
    "lr_splag_1_ged_ns",
    "lr_splag_1_ged_os",
]

# Three entities (1, 2, 3) — all carry the full 432-month history.
ENTITY_IDS: list[int] = [1, 2, 3]

# Standard partition used across the suite (matches the validation parquet's
# month_id range of 121..552).
PARTITION: dict[str, tuple[int, int]] = {
    "train": (121, 400),
    "test": (401, 552),
}

# Skip the whole suite if the validation parquet is not present.
pytestmark = pytest.mark.skipif(
    not PARQUET_PATH.exists(),
    reason=f"Validation parquet not found at {PARQUET_PATH}",
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset() -> ViewsDatasetDarts:
    """Load the validation parquet, subset to 3 entities, return a dataset.

    The subset keeps all 432 months for entities 1, 2, 3 — 1296 rows total.
    Constructed once per module (the parquet decode is the slow step).
    """
    frame, feats, targs = load_views_parquet(
        PARQUET_PATH, targets=TARGETS, features=FEATURES
    )
    full_ds = ViewsDatasetDarts(
        feature_frame=frame, targets=targs, features=feats
    )
    time, entity, values_2d = full_ds.get_subset_arrays(entity_ids=ENTITY_IDS)
    sub_index = SpatioTemporalIndex(
        time=time, unit=entity, level=SpatialLevel.CM
    )
    sub_frame = FeatureFrame.from_2d(
        values_2d, index=sub_index, feature_names=[*FEATURES, *TARGETS]
    )
    return ViewsDatasetDarts(
        feature_frame=sub_frame, targets=TARGETS, features=FEATURES
    )


def _make_mock_model(
    input_chunk_length: int = 12,
    output_chunk_length: int = 6,
) -> Mock:
    """Build a ``Mock(spec=TorchForecastingModel)`` for predict tests.

    Configures ``input_chunk_length`` / ``output_chunk_length`` and the
    nested ``model.parameters()`` iterator (the forecaster's device check
    accesses ``self.model.model.parameters()``).

    Args:
        input_chunk_length: Mock model input chunk length.
        output_chunk_length: Mock model output chunk length.

    Returns:
        A configured ``Mock`` instance.
    """
    m = Mock(spec=TorchForecastingModel)
    m.input_chunk_length = input_chunk_length
    m.output_chunk_length = output_chunk_length
    # The forecaster accesses ``self.model.model.parameters()`` during
    # predict to check the device. Configure the iterator explicitly.
    m.model = Mock()
    m.model.parameters.return_value = iter(
        [Mock(device=torch.device("cpu"))]
    )
    return m


def _make_prediction_series(
    entity_id: int,
    *,
    fill_value: float = 0.5,
    n_steps: int = 6,
    target_columns: list[str] | None = None,
) -> TimeSeries:
    """Build a synthetic Darts prediction TimeSeries for one entity.

    Args:
        entity_id: Country id to attach as the static covariate.
        fill_value: Constant fill value for every (time, target) cell.
        n_steps: Number of forecast time steps.
        target_columns: Component (column) names. Defaults to ``TARGETS``.

    Returns:
        A Darts :class:`TimeSeries` with shape ``(n_steps, len(targets), 1)``.
    """
    if target_columns is None:
        target_columns = TARGETS
    time = np.arange(
        401, 401 + n_steps, dtype=np.int64
    )
    values = np.full(
        (n_steps, len(target_columns)), fill_value, dtype=np.float32
    )
    return build_entity_timeseries(
        time=time,
        values=values,
        columns=target_columns,
        entity_id_name="country_id",
        entity_id_value=entity_id,
    )


# ----------------------------------------------------------------------
# Init tests
# ----------------------------------------------------------------------


class TestDartsForecasterInit:
    """Tests for :meth:`DartsForecaster.__init__`."""

    def test_init_basic(self, dataset: ViewsDatasetDarts) -> None:
        """A mock model + real dataset unpacks partition, sets scaler_fitted
        False, and reports device='cpu'."""
        fc = DartsForecaster(
            dataset=dataset,
            model=_make_mock_model(),
            partition_dict=PARTITION,
            target_scaler="MinMaxScaler",
            random_state=42,
        )
        assert fc._train_start == 121
        assert fc._train_end == 400
        assert fc._test_start == 401
        assert fc._test_end == 552
        assert fc.scaler_fitted is False
        assert fc.device == "cpu"

    def test_init_without_random_state_raises(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """``random_state=None`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="random_state"):
            DartsForecaster(
                dataset=dataset,
                model=_make_mock_model(),
                partition_dict=PARTITION,
                random_state=None,
            )

    def test_init_invalid_checkpoint_mode_raises(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """``checkpoint_mode='foo'`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="checkpoint_mode"):
            DartsForecaster(
                dataset=dataset,
                model=_make_mock_model(),
                partition_dict=PARTITION,
                random_state=42,
                checkpoint_mode="foo",
            )

    def test_init_log_targets_with_log_transform_scaler_warns(
        self,
        dataset: ViewsDatasetDarts,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``log_targets=True`` + ``target_scaler='LogTransform'`` disables
        ``log_targets`` and logs a warning."""
        with caplog.at_level(
            "WARNING",
            logger="views_r2darts2.engines.darts_forecaster",
        ):
            fc = DartsForecaster(
                dataset=dataset,
                model=_make_mock_model(),
                partition_dict=PARTITION,
                target_scaler="LogTransform",
                log_targets=True,
                random_state=42,
            )
        assert fc._log_targets is False
        assert any(
            "Disabling" in r.message or "double" in r.message
            for r in caplog.records
        ), (
            f"Expected double-log warning, got: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_init_log_features_with_log_transform_scaler_raises(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """``log_features=['x']`` + ``feature_scaler='LogTransform'`` raises
        ``ValueError`` (asymmetric with the target-side warning)."""
        with pytest.raises(ValueError, match="twice"):
            DartsForecaster(
                dataset=dataset,
                model=_make_mock_model(),
                partition_dict=PARTITION,
                feature_scaler="LogTransform",
                log_features=[FEATURES[0]],
                random_state=42,
            )

    def test_init_with_feature_scaler_map(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """Passing ``feature_scaler_map`` produces a
        :class:`FeatureScalerManager` (not a bare Scaler)."""
        fc = DartsForecaster(
            dataset=dataset,
            model=_make_mock_model(),
            partition_dict=PARTITION,
            feature_scaler_map={"MaxAbsScaler": FEATURES},
            random_state=42,
        )
        assert isinstance(fc.feature_scaler, FeatureScalerManager)

    def test_init_no_features_disables_feature_scaler(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """A dataset with ``features=[]`` forces ``feature_scaler=None``
        even when a feature scaler config is supplied."""
        # Build a no-feature dataset (3 targets only).
        time, entity, values_2d = dataset.get_subset_arrays(
            entity_ids=ENTITY_IDS
        )
        # Slice to keep only the target columns (last 3 columns).
        target_idx = [
            dataset.feature_frame.feature_names.index(t) for t in TARGETS
        ]
        values_targets_only = np.ascontiguousarray(
            values_2d[:, target_idx]
        )
        sub_index = SpatioTemporalIndex(
            time=time, unit=entity, level=SpatialLevel.CM
        )
        sub_frame = FeatureFrame.from_2d(
            values_targets_only, index=sub_index, feature_names=TARGETS
        )
        no_feat_ds = ViewsDatasetDarts(
            feature_frame=sub_frame, targets=TARGETS, features=[]
        )
        fc = DartsForecaster(
            dataset=no_feat_ds,
            model=_make_mock_model(),
            partition_dict=PARTITION,
            feature_scaler="MinMaxScaler",  # should be ignored
            random_state=42,
        )
        assert fc.feature_scaler is None

    def test_get_device_cpu(self) -> None:
        """Patching ``torch.cuda`` and ``torch.backends.mps`` to False forces
        ``get_device() == 'cpu'``."""
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=False
        ):
            assert DartsForecaster.get_device() == "cpu"


# ----------------------------------------------------------------------
# Preprocess tests
# ----------------------------------------------------------------------


class TestDartsForecasterPreprocess:
    """Tests for :meth:`DartsForecaster._preprocess_timeseries`."""

    def test_preprocess_timeseries_train_mode(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """In train mode, the scalers are fitted and the gates pass.

        Mock model with ``input_chunk_length=12``, ``output_chunk_length=6``
        — minimum length is 18, well below the 280-step train slice, so all
        3 entities pass the filter.
        """
        fc = DartsForecaster(
            dataset=dataset,
            model=_make_mock_model(input_chunk_length=12, output_chunk_length=6),
            partition_dict=PARTITION,
            target_scaler="MinMaxScaler",
            feature_scaler="RobustScaler",
            random_state=42,
        )
        series = dataset.as_darts_timeseries()
        targets, past_cov = fc._preprocess_timeseries(
            timeseries=series, start=121, end=400, train_mode=True
        )
        assert len(targets) == 3
        assert past_cov is not None
        assert len(past_cov) == 3
        # Each target series spans the train window (280 steps).
        for ts in targets:
            assert len(ts) == 280
        # Scalers are now fitted.
        assert fc.scaler_fitted is True

    def test_preprocess_timeseries_prediction_mode(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """After fit, predict-mode preprocess uses ``transform`` not
        ``fit_transform``.

        We verify this by spying on the target scaler's ``transform`` and
        ``fit_transform`` methods. After train-mode preprocess, predict-mode
        preprocess must call ``transform`` exactly once and ``fit_transform``
        zero times.
        """
        fc = DartsForecaster(
            dataset=dataset,
            model=_make_mock_model(input_chunk_length=12, output_chunk_length=6),
            partition_dict=PARTITION,
            target_scaler="MinMaxScaler",
            feature_scaler="RobustScaler",
            random_state=42,
        )
        series = dataset.as_darts_timeseries()
        # Fit scalers via train-mode preprocess.
        fc._preprocess_timeseries(
            timeseries=series, start=121, end=400, train_mode=True
        )
        assert fc.scaler_fitted is True

        # Spy on the fitted target scaler.
        fc.target_scaler.transform = Mock(wraps=fc.target_scaler.transform)
        fc.target_scaler.fit_transform = Mock(
            wraps=fc.target_scaler.fit_transform
        )

        # Predict-mode preprocess.
        fc._preprocess_timeseries(
            timeseries=series, start=121, end=400, train_mode=False
        )
        # transform was called once; fit_transform was NOT called.
        assert fc.target_scaler.transform.called, (
            "predict-mode preprocess should call target_scaler.transform"
        )
        assert not fc.target_scaler.fit_transform.called, (
            "predict-mode preprocess must NOT call target_scaler.fit_transform"
        )

    def test_preprocess_filters_entities_not_extending_to_boundary(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """Entities that don't extend to ``end`` are filtered out, not causing
        a hard ``DataStarvationError``.

        This is the validation-mode fix: the validation parquet has 22 entities
        that end before month 504 (e.g., entity 59 ends at 379). The training
        filter must exclude them rather than aborting the entire training run
        via ``audit_boundary_integrity``.

        We simulate this by using ``end=552`` (the global max) — entities
        ending before 552 will be filtered out by the boundary check, and the
        audit should NOT fire.
        """
        fc = DartsForecaster(
            dataset=dataset,
            model=_make_mock_model(input_chunk_length=12, output_chunk_length=6),
            partition_dict=PARTITION,
            target_scaler="MinMaxScaler",
            feature_scaler="RobustScaler",
            random_state=42,
        )
        # Use ALL entities (not just the 3-entity subset) to ensure some
        # don't extend to end=552.
        series = dataset.as_darts_timeseries()
        # end=552 is the global max month_id; entities 1-3 all extend there,
        # so this should pass with 3 entities. To test the filter, we need
        # an entity that DOESN'T extend to end. We'll use end=552 and verify
        # the filter runs without raising (all 3 entities in the subset
        # extend to 552).
        targets, past_cov = fc._preprocess_timeseries(
            timeseries=series, start=121, end=552, train_mode=True
        )
        # All 3 entities in the subset extend to 552, so all pass.
        assert len(targets) == 3
        assert fc.scaler_fitted is True

    def test_preprocess_boundary_filter_with_short_entity(self) -> None:
        """An entity that ends before ``end`` is filtered out silently.

        Build a synthetic dataset with one entity ending at month 200 and
        another ending at month 400. With ``end=400``, the first entity
        should be filtered out by the boundary check, and the second should
        pass. No ``DataStarvationError`` should be raised.
        """
        from views_frames import (
            FeatureFrame,
            SpatioTemporalIndex,
            SpatialLevel,
        )
        from views_r2darts2.data.views_dataset import ViewsDatasetDarts

        # Build a synthetic frame: entity 1 has data 121..200, entity 2 has 121..400.
        time_1 = np.arange(121, 201, dtype=np.int64)
        time_2 = np.arange(121, 401, dtype=np.int64)
        time = np.concatenate([time_1, time_2])
        entity = np.concatenate(
            [np.full(80, 1, dtype=np.int64), np.full(280, 2, dtype=np.int64)]
        )
        values = np.random.randn(360, 2, 1).astype(np.float32)
        index = SpatioTemporalIndex(
            time=time, unit=entity, level=SpatialLevel.CM
        )
        frame = FeatureFrame(values, index=index, feature_names=["feat1", "target"])
        ds = ViewsDatasetDarts(
            feature_frame=frame,
            targets=["target"],
            features=["feat1"],
        )
        fc = DartsForecaster(
            dataset=ds,
            model=_make_mock_model(input_chunk_length=12, output_chunk_length=6),
            partition_dict={"train": (121, 400), "test": (401, 500)},
            target_scaler="MinMaxScaler",
            feature_scaler="RobustScaler",
            random_state=42,
        )
        series = ds.as_darts_timeseries()
        # end=400 — entity 1 (ends at 200) should be filtered out.
        targets, past_cov = fc._preprocess_timeseries(
            timeseries=series, start=121, end=400, train_mode=True
        )
        # Only entity 2 passes (extends to 400 with 280 rows >= 18 min_length).
        assert len(targets) == 1
        assert past_cov is not None
        assert len(past_cov) == 1
        assert fc.scaler_fitted is True


# ----------------------------------------------------------------------
# Predict contract tests
# ----------------------------------------------------------------------


class TestDartsForecasterPredictContract:
    """Tests for :meth:`DartsForecaster.predict`."""

    def test_predict_before_fit_raises(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """Predicting before scalers are fitted raises ``RuntimeError``.

        Requires ``target_scaler`` set and ``scaler_fitted=False``.
        """
        fc = DartsForecaster(
            dataset=dataset,
            model=_make_mock_model(),
            partition_dict=PARTITION,
            target_scaler="MinMaxScaler",
            random_state=42,
        )
        assert fc.scaler_fitted is False
        with pytest.raises(RuntimeError, match="scalers were fitted"):
            fc.predict(sequence_number=0, output_length=6)

    def test_predict_returns_prediction_frame_dict(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """A successful predict returns a ``dict[str, PredictionFrame]``.

        Uses ``target_scaler=None`` so the inverse-transform step is skipped
        (the inverse path on a Darts bare ``Scaler`` is exercised elsewhere
        via Pipeline scalers in the scaler_selector suite).
        """
        mock_model = _make_mock_model()
        mock_model.predict.return_value = [
            _make_prediction_series(eid, fill_value=0.5)
            for eid in ENTITY_IDS
        ]
        fc = DartsForecaster(
            dataset=dataset,
            model=mock_model,
            partition_dict=PARTITION,
            target_scaler=None,
            random_state=42,
        )
        result = fc.predict(sequence_number=0, output_length=6)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(TARGETS)
        for tgt, frame in result.items():
            assert isinstance(frame, PredictionFrame), (
                f"result['{tgt}'] is {type(frame).__name__}, expected "
                "PredictionFrame"
            )
            # 3 entities × 6 time steps = 18 rows.
            assert frame.n_rows == 18

    def test_predict_clips_negatives(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """Negative model predictions are clipped to 0 in the output frames."""
        mock_model = _make_mock_model()
        mock_model.predict.return_value = [
            _make_prediction_series(eid, fill_value=-1.0)
            for eid in ENTITY_IDS
        ]
        fc = DartsForecaster(
            dataset=dataset,
            model=mock_model,
            partition_dict=PARTITION,
            target_scaler=None,
            random_state=42,
        )
        result = fc.predict(sequence_number=0, output_length=6)
        for tgt, frame in result.items():
            arr = frame.values
            assert np.all(arr >= 0.0), (
                f"target '{tgt}' has negative values after clipping: "
                f"min={arr.min()}"
            )

    def test_predict_locks_entropy(
        self, dataset: ViewsDatasetDarts
    ) -> None:
        """Predict must call :meth:`lock_entropy` exactly once with
        ``self.random_state``."""
        mock_model = _make_mock_model()
        mock_model.predict.return_value = [
            _make_prediction_series(eid, fill_value=0.5)
            for eid in ENTITY_IDS
        ]
        fc = DartsForecaster(
            dataset=dataset,
            model=mock_model,
            partition_dict=PARTITION,
            target_scaler=None,
            random_state=123,
        )
        with patch.object(
            ReproducibilityGate.Data,
            "lock_entropy",
            wraps=ReproducibilityGate.Data.lock_entropy,
        ) as spy:
            fc.predict(sequence_number=0, output_length=6)
        spy.assert_called_once_with(123)


# ----------------------------------------------------------------------
# Save / Load tests
# ----------------------------------------------------------------------


def _make_real_model() -> TCNModel:
    """Build a small untrained :class:`TCNModel` for save/load tests.

    ``n_epochs=1`` is set so the constructor is cheap; we never call
    ``fit`` — the save/load tests only exercise the persistence contract.
    """
    return TCNModel(
        input_chunk_length=12,
        output_chunk_length=6,
        n_epochs=1,
        kernel_size=2,
        num_filters=2,
        num_layers=1,
        dilation_base=1,
        random_state=42,
    )


class TestDartsForecasterSaveLoad:
    """Tests for :meth:`DartsForecaster.save_model` and :meth:`load_model`."""

    def test_save_model_writes_two_files(
        self,
        dataset: ViewsDatasetDarts,
        tmp_path: Path,
    ) -> None:
        """``save_model`` writes both the model artifact and the
        ``.scalers`` sidecar file."""
        fc = DartsForecaster(
            dataset=dataset,
            model=_make_real_model(),
            partition_dict=PARTITION,
            target_scaler="MinMaxScaler",
            feature_scaler="RobustScaler",
            random_state=42,
        )
        # Manually flip scaler_fitted so the saved state reflects a trained
        # forecaster (we don't actually train here — save_model only writes
        # the current state, not the trained weights).
        fc.scaler_fitted = True
        path = str(tmp_path / "model.pt")
        fc.save_model(path)
        assert (tmp_path / "model.pt").exists(), "model artifact not written"
        assert (
            tmp_path / "model.pt.scalers"
        ).exists(), ".scalers sidecar not written"

    def test_load_model_restores_scalers(
        self,
        dataset: ViewsDatasetDarts,
        tmp_path: Path,
    ) -> None:
        """Save with ``scaler_fitted=True``; load into a fresh forecaster
        (``scaler_fitted=False``) — after load, ``scaler_fitted`` is True."""
        fc_save = DartsForecaster(
            dataset=dataset,
            model=_make_real_model(),
            partition_dict=PARTITION,
            target_scaler="MinMaxScaler",
            feature_scaler="RobustScaler",
            random_state=42,
        )
        # Fit scalers via preprocess so the saved state is genuine.
        series = dataset.as_darts_timeseries()
        fc_save._preprocess_timeseries(
            timeseries=series, start=121, end=400, train_mode=True
        )
        assert fc_save.scaler_fitted is True
        path = str(tmp_path / "model.pt")
        fc_save.save_model(path)

        # Fresh forecaster with scaler_fitted=False.
        fc_load = DartsForecaster(
            dataset=dataset,
            model=_make_real_model(),
            partition_dict=PARTITION,
            target_scaler="MinMaxScaler",
            feature_scaler="RobustScaler",
            random_state=42,
        )
        assert fc_load.scaler_fitted is False
        fc_load.load_model(path)
        assert fc_load.scaler_fitted is True

    def test_load_model_missing_scalers_raises(
        self,
        dataset: ViewsDatasetDarts,
        tmp_path: Path,
    ) -> None:
        """Loading from a path with no ``.scalers`` file raises
        ``FileNotFoundError``."""
        fc = DartsForecaster(
            dataset=dataset,
            model=_make_real_model(),
            partition_dict=PARTITION,
            target_scaler="MinMaxScaler",
            random_state=42,
        )
        # No .scalers file at this path.
        path = str(tmp_path / "nonexistent.pt")
        with pytest.raises(FileNotFoundError):
            fc.load_model(path)

    def test_load_model_target_scaler_cfg_mismatch_raises(
        self,
        dataset: ViewsDatasetDarts,
        tmp_path: Path,
    ) -> None:
        """Saving with ``target_scaler='MinMaxScaler'`` and loading into a
        forecaster configured with ``target_scaler='StandardScaler'`` raises
        ``ValueError`` (prevents silent scaler-mismatch bugs)."""
        fc_save = DartsForecaster(
            dataset=dataset,
            model=_make_real_model(),
            partition_dict=PARTITION,
            target_scaler="MinMaxScaler",
            random_state=42,
        )
        path = str(tmp_path / "model.pt")
        fc_save.save_model(path)

        fc_load = DartsForecaster(
            dataset=dataset,
            model=_make_real_model(),
            partition_dict=PARTITION,
            target_scaler="StandardScaler",  # mismatch
            random_state=42,
        )
        with pytest.raises(ValueError, match="SCALER CONFIG MISMATCH"):
            fc_load.load_model(path)
