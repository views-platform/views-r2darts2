"""Unit tests for DartsForecastingModelManager helpers.

Covers C-02 from the technical risk register: the rolling-origin sequence count
was previously inlined at three call sites with a silent-failure mode — when
`max(steps) > test_len`, Python's `[None] * -1 == []` and `range(-1) == []`
silently produced an empty prediction batch with no error signal. The helper
`_resolve_total_sequence_number` centralises the formula and raises ValueError
on the invalid configuration instead of silently returning nothing.

Also covers the lazy-import contract (the manager class is importable without
``views_pipeline_core``, but ``__init__`` raises ``ImportError`` if it is
absent), the ``_get_predict_kwargs`` validation, and the ``_build_forecaster``
factory.

Google Python Style. Pandas-free.
"""

from unittest.mock import MagicMock

import pytest

from views_r2darts2.engines.darts_forecaster import DartsForecaster
from views_r2darts2.engines.darts_forecasting_model_manager import (
    DartsForecastingModelManager,
    _PARENT_CLASS,
)


class TestResolveTotalSequenceNumber:
    """C-02 regression: the sequence count helper."""

    def test_standard_case_returns_expected_count(self):
        partition = {"test": (445, 480)}  # test_len = 36
        assert (
            DartsForecastingModelManager._resolve_total_sequence_number(
                partition, max_steps=12
            )
            == 25  # 36 - 12 + 1
        )

    def test_boundary_test_len_equals_max_steps_returns_one(self):
        """Exactly one rolling-origin sequence is valid."""
        partition = {"test": (100, 111)}  # test_len = 12
        assert (
            DartsForecastingModelManager._resolve_total_sequence_number(
                partition, max_steps=12
            )
            == 1
        )

    def test_test_len_smaller_than_max_steps_raises(self):
        """[REGRESSION — C-02 / Copilot finding on PR #10]

        Before the helper existed, this misconfiguration silently produced an
        empty prediction batch via `[None] * -1 == []`, which then propagated
        through `_evaluate_prediction_dataframe` as zero-metric fallthrough.
        Must fail loudly.
        """
        partition = {"test": (100, 110)}  # test_len = 11
        with pytest.raises(ValueError, match="test partition length"):
            DartsForecastingModelManager._resolve_total_sequence_number(
                partition, max_steps=12
            )

    def test_test_len_one_less_than_max_steps_raises(self):
        """The exact boundary that used to produce `total = 0` — empty output."""
        partition = {"test": (100, 110)}  # test_len = 11, max_steps = 12
        with pytest.raises(ValueError):
            DartsForecastingModelManager._resolve_total_sequence_number(
                partition, max_steps=12
            )

    def test_far_below_raises_not_returns_negative(self):
        """The silent-failure mode Copilot flagged: negative totals."""
        partition = {"test": (100, 105)}  # test_len = 6
        with pytest.raises(ValueError):
            DartsForecastingModelManager._resolve_total_sequence_number(
                partition, max_steps=36
            )

    def test_error_message_includes_both_values_for_debuggability(self):
        partition = {"test": (100, 110)}  # test_len = 11
        with pytest.raises(ValueError) as exc_info:
            DartsForecastingModelManager._resolve_total_sequence_number(
                partition, max_steps=12
            )
        msg = str(exc_info.value)
        assert "11" in msg and "12" in msg


# ----------------------------------------------------------------------
# Lazy-import contract tests
# ----------------------------------------------------------------------


class TestManagerLazyImport:
    """The manager class is importable without ``views_pipeline_core``, but
    ``__init__`` raises ``ImportError`` at call time.

    These tests verify the lazy-import contract. When
    ``views_pipeline_core`` IS installed (the user's production environment),
    the tests skip — they only assert behavior in the absent-parent case.
    """

    def test_manager_class_importable_without_views_pipeline_core(self) -> None:
        """Importing :class:`DartsForecastingModelManager` must always succeed.

        When vpc is absent, ``_PARENT_CLASS`` is ``object`` and the class is a
        bare ``type``. When vpc is present, ``_PARENT_CLASS`` is the real
        ``ForecastingModelManager`` and the class inherits from it. Either way,
        the class itself is importable and is a ``type``.
        """
        assert isinstance(DartsForecastingModelManager, type)
        # The _PARENT_CLASS sentinel is either `object` (vpc absent) or the
        # real ForecastingModelManager (vpc present). Both are valid.
        assert _PARENT_CLASS is not None
        # The class must inherit from _PARENT_CLASS (object when vpc absent,
        # ForecastingModelManager when vpc present).
        assert issubclass(DartsForecastingModelManager, _PARENT_CLASS)

    @pytest.mark.skipif(
        _PARENT_CLASS is not object,
        reason="views_pipeline_core is installed — the ImportError path is "
        "only exercised when vpc is absent.",
    )
    def test_manager_init_raises_without_views_pipeline_core(self) -> None:
        """Calling ``DartsForecastingModelManager(model_path=...)`` raises
        ``ImportError`` with a helpful message when vpc is absent.

        Skipped when vpc IS installed (the ``__init__`` succeeds in that case).
        """
        with pytest.raises(ImportError, match="views_pipeline_core"):
            DartsForecastingModelManager(model_path=MagicMock())


# ----------------------------------------------------------------------
# _get_predict_kwargs
# ----------------------------------------------------------------------


class TestGetPredictKwargs:
    """Tests for :meth:`DartsForecastingModelManager._get_predict_kwargs`."""

    @staticmethod
    def _make_manager() -> "DartsForecastingModelManager":
        """Build a manager instance via ``__new__`` (bypasses ``__init__``).

        The parent-class ``__init__`` is what raises ``ImportError`` when
        ``views_pipeline_core`` is absent; the static helpers and instance
        helpers like ``_get_predict_kwargs`` are usable on a bare instance.
        """
        return DartsForecastingModelManager.__new__(DartsForecastingModelManager)

    def test_get_predict_kwargs_complete(self) -> None:
        """A config with ``num_samples=10`` and ``mc_dropout=True`` returns
        both keys."""
        mgr = self._make_manager()
        kwargs = mgr._get_predict_kwargs(
            {"num_samples": 10, "mc_dropout": True}
        )
        assert kwargs == {"num_samples": 10, "mc_dropout": True}

    def test_get_predict_kwargs_missing_num_samples_raises(self) -> None:
        """A config missing ``num_samples`` raises ``ValueError``."""
        mgr = self._make_manager()
        with pytest.raises(ValueError, match="num_samples"):
            mgr._get_predict_kwargs({"mc_dropout": True})

    def test_get_predict_kwargs_missing_mc_dropout_raises(self) -> None:
        """A config missing ``mc_dropout`` raises ``ValueError``."""
        mgr = self._make_manager()
        with pytest.raises(ValueError, match="mc_dropout"):
            mgr._get_predict_kwargs({"num_samples": 10})


# ----------------------------------------------------------------------
# _build_forecaster factory
# ----------------------------------------------------------------------


class TestBuildForecasterFactory:
    """Tests for :meth:`DartsForecastingModelManager._build_forecaster`."""

    def test_build_forecaster_returns_darts_forecaster(self) -> None:
        """``_build_forecaster`` returns a :class:`DartsForecaster` when
        given a (mock) dataset + (mock) model + minimal config.

        The manager is built via ``__new__`` to bypass the vpc-dependent
        ``__init__``; ``_build_forecaster`` itself does not touch the parent
        class, so it is usable on a bare instance.
        """
        mgr = DartsForecastingModelManager.__new__(DartsForecastingModelManager)

        # Mock dataset: returns a non-empty features list so the forecaster
        # wires up a feature scaler (rather than disabling it).
        mock_dataset = MagicMock()
        mock_dataset.features = ["feat_a"]
        mock_model = MagicMock()

        minimal_config: dict = {
            "random_state": 42,
            "feature_scaler": "MinMaxScaler",
            "target_scaler": "MinMaxScaler",
            "log_targets": False,
            "log_features": [],
            "feature_scaler_map": None,
            "use_static_covariates": False,
            "use_cyclic_encoders": False,
        }
        partition = {"train": (121, 400), "test": (401, 552)}

        forecaster = mgr._build_forecaster(
            active_config=minimal_config,
            partition=partition,
            dataset=mock_dataset,
            model_object=mock_model,
        )
        assert isinstance(forecaster, DartsForecaster)
