"""The patched training step and ValMetricsCallback must hand
``_produce_train_output`` the tuple shape the installed Darts expects:
five inputs on <=0.45, five inputs + future_target on 0.46+."""

from unittest.mock import MagicMock

import torch

from views_r2darts2.infrastructure import callbacks as cb


def _batch():
    inputs = tuple(torch.zeros(2, 4, 1) for _ in range(5))
    sample_weight, target = None, torch.zeros(2, 3, 1)
    return (*inputs, sample_weight, target), target


def test_inputs_helper_five_on_darts_le_045(monkeypatch):
    monkeypatch.setattr(cb, "_PRODUCE_TRAIN_OUTPUT_TAKES_TARGET", False)
    batch, target = _batch()
    assert len(cb._produce_train_output_inputs(batch, target)) == 5


def test_inputs_helper_six_on_darts_ge_046(monkeypatch):
    monkeypatch.setattr(cb, "_PRODUCE_TRAIN_OUTPUT_TAKES_TARGET", True)
    batch, target = _batch()
    out = cb._produce_train_output_inputs(batch, target)
    assert len(out) == 6 and out[-1] is target


def test_patched_training_step_uses_the_helper(monkeypatch):
    monkeypatch.setattr(cb, "_PRODUCE_TRAIN_OUTPUT_TAKES_TARGET", False)
    pl_module = MagicMock()
    pl_module._produce_train_output.return_value = torch.zeros(2, 3, 1)
    pl_module._compute_loss.return_value = torch.tensor(0.0)
    batch, _ = _batch()
    cb._PatchedTrainingStep(pl_module)(batch, batch_idx=0)
    (passed,), _ = pl_module._produce_train_output.call_args
    assert len(passed) == 5


def test_val_metrics_callback_buffers_a_batch(monkeypatch):
    monkeypatch.setattr(cb, "_PRODUCE_TRAIN_OUTPUT_TAKES_TARGET", False)
    pl_module = MagicMock()
    pl_module._produce_train_output.return_value = torch.ones(2, 3, 1)
    callback = cb.ValMetricsCallback()
    batch, _ = _batch()
    callback.on_validation_batch_end(MagicMock(), pl_module, None, batch, 0)
    assert len(callback._preds) == 1 and callback._preds[0].shape == (2, 3, 1)


def test_detection_matches_installed_darts():
    import darts
    major, minor = (int(x) for x in darts.__version__.split(".")[:2])
    assert cb._produce_train_output_takes_target() is ((major, minor) >= (0, 46))
