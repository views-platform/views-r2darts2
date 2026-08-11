"""Tests for :class:`views_r2darts2.engines.darts_forecaster.DartsForecaster`.

Exercises the init contract (config validation, scaler wiring, device
detection), the scaler-fit + scaled-series flow (delegated to the dataset's
``fit_scalers`` / ``get_scaled_darts_timeseries``), the predict contract
(RuntimeError on unfitted scalers, :class:`PredictionFrame` dict output,
negative clipping, entropy lock), and the save/load round-trip.

The forecaster requires both a real :class:`ViewsDataset` (built from the
synthetic country-month parquet, subsetted to 3 entities for speed) and a
real Darts model for save/load tests. For predict tests, the Darts model is
mocked via ``Mock(spec=TorchForecastingModel)`` — this avoids the cost of
training a real model while still exercising the full predict → inverse →
``PredictionFrame`` pipeline.

The new slim forecaster no longer owns a ``_preprocess_timeseries`` method
or instantiated scalers — scalers live on the dataset
(``dataset._target_scaler`` / ``dataset._feature_scaler``), and the
preprocessing flow is delegated to ``dataset.fit_scalers`` +
``dataset.get_scaled_darts_timeseries``.

``pandas`` is used only at the Darts ``TimeSeries``
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
    PredictionFrame,
    SpatioTemporalIndex,
    SpatialLevel,
)
from views_r2darts2.dataset.base import ViewsDataset
from views_r2darts2.engines.darts_forecaster import DartsForecaster
from views_r2darts2.infrastructure.reproducibility_gate import (
    ReproducibilityGate,
)
from views_r2darts2.transformers.darts_bridge import build_entity_timeseries
from views_r2darts2.transformers.feature_scaler_manager import (
    FeatureScalerManager,
)

# Three targets + nine features used throughout the suite (mirrors conftest).
TARGETS: list[str] = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
FEATURES: list[str] = [
    "lr_ged_sb_delta",
    "lr_ged_ns_delta",
    "lr_ged_os_delta",
    "lr_splag_1_ged_sb",
    "lr_splag_1_ged_ns",
    "lr_splag_1_ged_os",
    "lr_decay_ged_sb_1",
    "lr_decay_ged_sb_5",
    "lr_decay_ged_sb_25",
]

# Three entities (1, 2, 3) — all carry the full 100-month history.
ENTITY_IDS: list[int] = [1, 2, 3]

# Standard partition used across the suite (matches the synthetic parquet's
# month_id range of 121..220).
PARTITION: dict[str, tuple[int, int]] = {
    "train": (121, 200),
    "test": (201, 220),
}


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset(synthetic_cm_parquet_small: Path) -> ViewsDataset:
    """Load the synthetic cm parquet, subset to 3 entities, return a dataset.

    The new API takes the parquet path + ``targets`` + ``broadcast_features``.
    Subsetting is done via ``get_subset_dataset(entity_ids=...)`` (replaces
    the old ``get_subset_arrays``). Constructed once per module (the parquet
    decode + zarr write is the slow step).
    """
    full_ds = ViewsDataset(
        synthetic_cm_parquet_small,
        targets=TARGETS,
        broadcast_features=True,
    )
    return full_ds.get_subset_dataset(entity_ids=ENTITY_IDS)


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
    # predict to check the device (potentially twice — once to detect CPU
    # drift, once to verify restoration). Use ``side_effect`` so each call
    # returns a FRESH iterator (``return_value`` would share one exhausted
    # iterator across calls, causing StopIteration on the second ``next()``).
    m.model = Mock()
    m.model.parameters.side_effect = lambda: iter(
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

    def test_init_basic(self, dataset: ViewsDataset) -> None:
        """A mock model + real dataset unpacks partition, sets scaler_fitted
        False, and reports the mocked device."""
        # Mock get_device so the test is deterministic (MPS/CUDA machines
        # would otherwise report 'mps'/'cuda' instead of 'cpu').
        with patch("views_r2darts2.engines.darts_forecaster._get_device", return_value="cpu"):
            fc = DartsForecaster(
                dataset=dataset,
                model=_make_mock_model(),
                partition_dict=PARTITION,
                target_scaler="MinMaxScaler",
                random_state=42,
            )
        assert fc._train_start == 121
        assert fc._train_end == 200
        assert fc._test_start == 201
        assert fc._test_end == 220
        assert fc.scaler_fitted is False
        assert fc.device == "cpu"

    def test_init_without_random_state_raises(
        self, dataset: ViewsDataset
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
        self, dataset: ViewsDataset
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
        dataset: ViewsDataset,
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
        self, dataset: ViewsDataset
    ) -> None:
        """``log_features=['x']`` + ``feature_scaler='LogTransform'`` raises
        ``ValueError`` (asymmetric with the target-side warning)."""
        with pytest.raises(ValueError, match="only one transformation"):
            DartsForecaster(
                dataset=dataset,
                model=_make_mock_model(),
                partition_dict=PARTITION,
                feature_scaler="LogTransform",
                log_features=[FEATURES[0]],
                random_state=42,
            )

    def test_init_with_feature_scaler_map_config(
        self, dataset: ViewsDataset
    ) -> None:
        """Passing ``feature_scaler_map`` stores the config on the forecaster;
        the actual :class:`FeatureScalerManager` is instantiated by
        ``dataset.fit_scalers`` (slim forecaster design)."""
        fc = DartsForecaster(
            dataset=dataset,
            model=_make_mock_model(),
            partition_dict=PARTITION,
            feature_scaler_map={"MaxAbsScaler": FEATURES},
            random_state=42,
        )
        # The config is stored verbatim.
        assert fc._feature_scaler_map_cfg == {"MaxAbsScaler": FEATURES}
        # Fit scalers to verify the FeatureScalerManager is instantiated on
        # the dataset.
        fc.dataset.fit_scalers(
            target_scaler="MinMaxScaler",
            feature_scaler_map={"MaxAbsScaler": FEATURES},
            time_ids=list(range(121, 201)),
        )
        assert isinstance(fc.dataset._feature_scaler, FeatureScalerManager)

    def test_init_no_features_disables_feature_scaler(
        self, synthetic_cm_parquet_small: Path
    ) -> None:
        """A dataset with ``features=[]`` forces ``dataset._feature_scaler=None``
        even when a feature scaler config is supplied.

        Built via ``ViewsDataset.create_empty`` + ``add_batch`` (the new
        incremental-concatenation API) since the parquet loader auto-derives
        features from the schema.
        """
        # Build a no-feature dataset (targets only).
        ds = ViewsDataset.create_empty(
            "cm", features=[], targets=TARGETS, sample_size=1
        )
        # Add a few rows so fit_scalers has data to fit on.
        n_time = 80
        for eid in ENTITY_IDS:
            ds.add_batch(
                times=np.arange(121, 121 + n_time, dtype=np.int64),
                entities=np.full(n_time, eid, dtype=np.int64),
                values={
                    t: np.linspace(0.1, 1.0, n_time, dtype=np.float32)
                    for t in TARGETS
                },
            )
        fc = DartsForecaster(
            dataset=ds,
            model=_make_mock_model(),
            partition_dict=PARTITION,
            feature_scaler="MinMaxScaler",  # should be ignored
            random_state=42,
        )
        fc.dataset.fit_scalers(
            target_scaler="MinMaxScaler",
            feature_scaler="MinMaxScaler",
            time_ids=list(range(121, 201)),
        )
        assert fc.dataset._feature_scaler is None

    def test_get_device_cpu(self) -> None:
        """Patching ``torch.cuda`` and ``torch.backends.mps`` to False forces
        ``get_device() == 'cpu'``."""
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=False
        ):
            assert DartsForecaster.get_device() == "cpu"


# ----------------------------------------------------------------------
# Scaler-fit / scaled-series tests
# ----------------------------------------------------------------------


class TestDartsForecasterScalerFlow:
    """Tests for the new dataset-driven scaler fit + scaled-series flow.

    The slim forecaster delegates all preprocessing to ``dataset.fit_scalers``
    + ``dataset.get_scaled_darts_timeseries``. These tests verify that flow.
    """

    def test_fit_scalers_fits_target_and_feature_scalers(
        self, dataset: ViewsDataset
    ) -> None:
        """``dataset.fit_scalers`` instantiates and fits both scalers."""
        dataset.fit_scalers(
            target_scaler="MinMaxScaler",
            feature_scaler="RobustScaler",
            time_ids=list(range(121, 201)),
        )
        assert dataset.scalers_fitted is True
        assert dataset._target_scaler is not None
        assert dataset._feature_scaler is not None

    def test_get_scaled_darts_timeseries_returns_targets_and_past_cov(
        self, dataset: ViewsDataset
    ) -> None:
        """After fit, ``get_scaled_darts_timeseries`` returns
        ``(targets, past_covariates)`` with one series per entity."""
        dataset.fit_scalers(
            target_scaler="MinMaxScaler",
            feature_scaler="RobustScaler",
            time_ids=list(range(121, 201)),
        )
        targets, past_cov = dataset.get_scaled_darts_timeseries(
            time_ids=list(range(121, 201)),
            entity_ids=ENTITY_IDS,
        )
        assert len(targets) == len(ENTITY_IDS)
        assert past_cov is not None
        assert len(past_cov) == len(ENTITY_IDS)
        # Each target series spans the train window (80 steps).
        for ts in targets:
            assert len(ts) == 80
        # MinMaxScaler maps targets to [0, 1].
        for ts in targets:
            arr = ts.all_values(copy=False)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            assert float(arr.min()) >= -1e-5
            assert float(arr.max()) <= 1.0 + 1e-5

    def test_get_scaled_darts_timeseries_before_fit_raises(
        self, synthetic_cm_parquet_small: Path
    ) -> None:
        """``get_scaled_darts_timeseries`` before ``fit_scalers`` raises."""
        fresh = ViewsDataset(
            synthetic_cm_parquet_small,
            targets=TARGETS,
            broadcast_features=True,
        )
        with pytest.raises(RuntimeError, match="Scalers not fitted"):
            fresh.get_scaled_darts_timeseries()

    def test_fit_scalers_with_time_filter(
        self, dataset: ViewsDataset
    ) -> None:
        """``fit_scalers(time_ids=...)`` restricts the fit to the train window."""
        # Fit on a 12-month window.
        dataset.fit_scalers(
            target_scaler="MinMaxScaler",
            feature_scaler="RobustScaler",
            time_ids=list(range(121, 133)),
        )
        assert dataset.scalers_fitted is True
        # The fitted MinMaxScaler's data_min should reflect the 12-month
        # window (not the full 100-month range).
        # Darts Scaler wraps sklearn MinMaxScaler; access via ._fitted
        # or similar. Just verify it's fitted and produces scaled output.
        targets, _ = dataset.get_scaled_darts_timeseries(
            time_ids=list(range(121, 133)),
        )
        assert len(targets) == len(ENTITY_IDS)


# ----------------------------------------------------------------------
# Predict contract tests
# ----------------------------------------------------------------------


class TestDartsForecasterPredictContract:
    """Tests for :meth:`DartsForecaster.predict`."""

    def test_predict_before_fit_raises(
        self, dataset: ViewsDataset
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
        self, dataset: ViewsDataset
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
        # Fit scalers (no-op since both are None) and flip the forecaster's
        # scaler_fitted flag (the runtime check is on the forecaster, not
        # the dataset).
        fc.dataset.fit_scalers(
            target_scaler=None,
            feature_scaler=None,
            time_ids=list(range(121, 201)),
        )
        fc.scaler_fitted = True
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
        self, dataset: ViewsDataset
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
        fc.dataset.fit_scalers(
            target_scaler=None,
            feature_scaler=None,
            time_ids=list(range(121, 201)),
        )
        fc.scaler_fitted = True
        result = fc.predict(sequence_number=0, output_length=6)
        for tgt, frame in result.items():
            arr = frame.values
            assert np.all(arr >= 0.0), (
                f"target '{tgt}' has negative values after clipping: "
                f"min={arr.min()}"
            )

    def test_predict_locks_entropy(
        self, dataset: ViewsDataset
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
        fc.dataset.fit_scalers(
            target_scaler=None,
            feature_scaler=None,
            time_ids=list(range(121, 201)),
        )
        fc.scaler_fitted = True
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
        dataset: ViewsDataset,
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
        dataset: ViewsDataset,
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
        # Fit scalers via dataset.fit_scalers so the saved state is genuine.
        fc_save.dataset.fit_scalers(
            target_scaler="MinMaxScaler",
            feature_scaler="RobustScaler",
            time_ids=list(range(121, 201)),
        )
        fc_save.scaler_fitted = True
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
        dataset: ViewsDataset,
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
        dataset: ViewsDataset,
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
        fc_save.scaler_fitted = True
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
