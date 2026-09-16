"""The sweep loop takes its hyperparameters from the ``Run`` that
``WandBModule.initialize_run`` returns, not from a module-level ``import wandb``.

``sys.modules["wandb"] = None`` makes any ``import wandb`` raise ``ImportError``,
so these tests fail if the direct import ever comes back. pipeline-core is
stubbed at the two import sites the method has, so the tests run without it.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from views_r2darts2.engines.darts_forecasting_model_manager import (
    DartsForecastingModelManager,
)

_MANAGER_MODULE = "views_r2darts2.engines.darts_forecasting_model_manager"


def _pipeline_core_stubs() -> dict:
    exceptions = types.ModuleType("views_pipeline_core.exceptions.exceptions")
    exceptions.PipelineException = type("PipelineException", (Exception,), {})
    sniffer = types.ModuleType(
        "views_pipeline_core.modules.validation.core_prediction_sniffer"
    )
    sniffer.CorePredictionSniffer = MagicMock()
    return {
        "wandb": None,
        "views_pipeline_core.exceptions.exceptions": exceptions,
        "views_pipeline_core.modules.validation.core_prediction_sniffer": sniffer,
    }


def _fake_manager(evaluate_sweep_returns):
    fake = MagicMock()
    fake.configs = {"level": "cm", "targets": ["ged_sb"], "steps": [1, 2]}
    fake._evaluate_sweep.return_value = evaluate_sweep_returns
    fake._has_evaluation_metrics.return_value = True
    run = MagicMock(name="wandb_run")
    fake._wandb_module.initialize_run.return_value.__enter__.return_value = run
    return fake, run


@pytest.fixture
def isolated_imports():
    with patch.dict(sys.modules, _pipeline_core_stubs()), patch(
        f"{_MANAGER_MODULE}.ReproducibilityGate"
    ):
        yield


def test_sweep_config_comes_from_the_run_initialize_run_returns(isolated_imports):
    fake, run = _fake_manager(evaluate_sweep_returns=[])

    DartsForecastingModelManager._execute_model_sweeping(fake)

    fake._config_manager.update_for_sweep_run.assert_called_once_with(
        run.config, fake.args, wandb_module=fake._wandb_module
    )
    fake._wandb_module.finish_run.assert_called_once_with()


def test_sweep_converts_prediction_frame_dict_per_sequence(isolated_imports):
    frames = {"ged_sb": ["seq0", "seq1", "seq2"]}
    fake, _ = _fake_manager(evaluate_sweep_returns=frames)

    DartsForecastingModelManager._execute_model_sweeping(fake)

    assert fake._predictions_to_dataframe.call_count == 3
    fake._evaluate_prediction_dataframe.assert_called_once()


def test_sweep_finishes_the_run_even_when_training_raises(isolated_imports):
    fake, _ = _fake_manager(evaluate_sweep_returns=[])
    fake._train_model_artifact.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        DartsForecastingModelManager._execute_model_sweeping(fake)

    fake._wandb_module.finish_run.assert_called_once_with()
