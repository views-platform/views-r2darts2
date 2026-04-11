"""Unit tests for DartsForecastingModelManager helpers.

Covers C-02 from the technical risk register: the rolling-origin sequence count
was previously inlined at three call sites with a silent-failure mode — when
`max(steps) > test_len`, Python's `[None] * -1 == []` and `range(-1) == []`
silently produced an empty prediction batch with no error signal. The helper
`_resolve_total_sequence_number` centralises the formula and raises ValueError
on the invalid configuration instead of silently returning nothing.
"""

import pytest

from views_r2darts2.engines.darts_forecasting_model_manager import (
    DartsForecastingModelManager,
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
