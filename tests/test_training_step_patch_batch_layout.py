"""The patched training step and ValMetricsCallback must hand
``_produce_train_output`` the tuple shape the installed Darts expects:
five inputs on <=0.45, five inputs + future_target on 0.46+ — and both call
sites must follow the shared flag rather than a hardcoded layout."""

from unittest.mock import MagicMock

import pytest
import torch

from views_r2darts2.infrastructure import callbacks as cb


def _batch():
    inputs = tuple(torch.zeros(2, 4, 1) for _ in range(5))
    sample_weight, target = None, torch.zeros(2, 3, 1)
    return (*inputs, sample_weight, target), target


def _expected(inputs_5, target, takes_target):
    return (*inputs_5, target) if takes_target else tuple(inputs_5)


def test_inputs_helper_five_on_darts_le_045(monkeypatch):
    monkeypatch.setattr(cb, "_PRODUCE_TRAIN_OUTPUT_TAKES_TARGET", False)
    batch, target = _batch()
    assert len(cb._produce_train_output_inputs(batch, target)) == 5


def test_inputs_helper_six_on_darts_ge_046(monkeypatch):
    monkeypatch.setattr(cb, "_PRODUCE_TRAIN_OUTPUT_TAKES_TARGET", True)
    batch, target = _batch()
    out = cb._produce_train_output_inputs(batch, target)
    assert len(out) == 6 and out[-1] is target


@pytest.mark.parametrize("takes_target", [False, True])
def test_patched_training_step_follows_the_flag(monkeypatch, takes_target):
    monkeypatch.setattr(cb, "_PRODUCE_TRAIN_OUTPUT_TAKES_TARGET", takes_target)
    pl_module = MagicMock()
    pl_module._produce_train_output.return_value = torch.zeros(2, 3, 1)
    pl_module._compute_loss.return_value = torch.tensor(0.0)
    batch, target = _batch()
    cb._PatchedTrainingStep(pl_module)(batch, batch_idx=0)
    (passed,), _ = pl_module._produce_train_output.call_args
    expected = _expected(batch[:5], target, takes_target)
    assert len(passed) == len(expected)
    assert all(a is b for a, b in zip(passed, expected))


@pytest.mark.parametrize("takes_target", [False, True])
def test_val_metrics_callback_follows_the_flag(monkeypatch, takes_target):
    monkeypatch.setattr(cb, "_PRODUCE_TRAIN_OUTPUT_TAKES_TARGET", takes_target)
    pl_module = MagicMock()
    pl_module._produce_train_output.return_value = torch.ones(2, 3, 1)
    callback = cb.ValMetricsCallback()
    batch, target = _batch()
    callback.on_validation_batch_end(MagicMock(), pl_module, None, batch, 0)
    (passed,), _ = pl_module._produce_train_output.call_args
    expected = _expected(batch[:5], target, takes_target)
    assert len(passed) == len(expected)
    assert all(a is b for a, b in zip(passed, expected))
    assert len(callback._preds) == 1 and callback._preds[0].shape == (2, 3, 1)


def test_val_metrics_callback_reports_why_a_batch_was_skipped(monkeypatch, caplog):
    # the except clause swallows failures at DEBUG; a test that hits it should see the cause
    pl_module = MagicMock()
    pl_module._produce_train_output.side_effect = RuntimeError("layout mismatch")
    callback = cb.ValMetricsCallback()
    batch, _ = _batch()
    with caplog.at_level("DEBUG", logger=cb.logger.name):
        callback.on_validation_batch_end(MagicMock(), pl_module, None, batch, 0)
    assert callback._preds == [] and "layout mismatch" in caplog.text


def test_detection_reads_the_installed_source_not_just_the_version():
    # the source-inspection path must be the one that decides; the version
    # fallback only runs when source is unavailable
    import inspect
    from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
    src = inspect.getsource(PLForecastingModule._train_val_step)
    call = src.split("_produce_train_output(", 1)[1].split("loss = ", 1)[0]
    assert cb._produce_train_output_takes_target() is ("future_target" in call)


@pytest.mark.parametrize("source_says_target", [True, False])
def test_detection_believes_the_source_over_the_version(monkeypatch, source_says_target):
    # inject a _train_val_step source whose answer may contradict the installed
    # version number; the source must win whenever it is readable
    import inspect
    tail = "future_target if name == 'train' else None," if source_says_target else "static_covariates,"
    fake_src = f"        output = self._produce_train_output((past_target, {tail}))\n        loss = self._compute_loss(output)\n"
    monkeypatch.setattr(inspect, "getsource", lambda *_: fake_src)
    assert cb._produce_train_output_takes_target() is source_says_target


def test_detection_falls_back_to_version_when_source_unavailable(monkeypatch):
    import inspect
    monkeypatch.setattr(inspect, "getsource", lambda *_: (_ for _ in ()).throw(OSError("no source")))
    import darts
    major, minor = (int(x) for x in darts.__version__.split(".")[:2])
    assert cb._produce_train_output_takes_target() is ((major, minor) >= (0, 46))
