"""RichLossDiagnosticsCallback must degrade to a warning, not a NameError,
when `rich` is absent — its own import block already tolerates that."""

from unittest.mock import MagicMock

from views_r2darts2.infrastructure import callbacks as cb


def _trainer_and_module(epoch=0):
    trainer = MagicMock()
    trainer.current_epoch = epoch
    crit = MagicMock()
    crit._last_components = {"shape": [1.0], "level": [1.0], "anchor": [1.0]}
    pl_module = MagicMock()
    pl_module.train_criterion = crit
    return trainer, pl_module


def test_epoch_end_without_rich_warns_once_and_returns(monkeypatch, caplog):
    monkeypatch.setattr(cb, "_HAS_RICH", False)
    callback = cb.RichLossDiagnosticsCallback(log_every_n_epochs=1)
    trainer, pl_module = _trainer_and_module()
    with caplog.at_level("WARNING", logger=cb.logger.name):
        callback.on_train_epoch_end(trainer, pl_module)
        callback.on_train_epoch_end(trainer, pl_module)
    assert caplog.text.count("rich` is not installed") == 1


def test_epoch_end_with_rich_builds_the_table(monkeypatch):
    if not cb._HAS_RICH:
        import pytest
        pytest.skip("rich not installed in this environment")
    callback = cb.RichLossDiagnosticsCallback(log_every_n_epochs=1)
    trainer, pl_module = _trainer_and_module()
    printed = []
    monkeypatch.setattr(cb._console, "print", lambda *a, **k: printed.append(a))
    callback.on_train_epoch_end(trainer, pl_module)
    assert printed, "with rich present the diagnostics table should be printed"
