"""The streaming predict path must regenerate darts' ``add_encoders`` at
inference. With ``use_cyclic_encoders=True`` the model trains on
(features + encoder columns); ``_predict_streaming`` drives
``predict_from_dataset`` directly, which skips the regeneration ``predict()``
does, so without the fix darts refuses with "must have equal number of
components". A real (tiny, CPU, one-epoch) fit is the only honest guard."""

import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

from views_r2darts2 import DartsForecaster, ModelCatalog, ViewsDataset
from views_r2darts2.catalogs import model_catalog as mc

warnings.filterwarnings("ignore")


@pytest.fixture
def cpu_catalog(monkeypatch):
    # the catalog hardcodes accelerator="gpu" (register C-21); run this on CPU
    original = mc.ModelCatalog._get_common_pl_trainer_kwargs
    def _cpu(self, extra_callbacks=None):
        from pytorch_lightning.loggers import CSVLogger
        return {**original(self, extra_callbacks), "accelerator": "cpu", "devices": 1,
                "logger": CSVLogger(tempfile.mkdtemp()), "enable_progress_bar": False}
    monkeypatch.setattr(mc.ModelCatalog, "_get_common_pl_trainer_kwargs", _cpu)


def _config(use_cyclic_encoders: bool) -> dict:
    return {
        "name": "probe", "algorithm": "NLinearModel", "run_type": "calibration", "level": "cm",
        "steps": [1, 2, 3], "random_state": 7, "loss_function": "MSELoss", "lr": 1e-3,
        "weight_decay": 0.0, "batch_size": 8, "n_epochs": 1, "optimizer_cls": "Adam",
        "lr_scheduler_cls": "ReduceLROnPlateau", "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 1, "lr_scheduler_min_lr": 1e-6, "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.0, "gradient_clip_val": 1.0, "num_samples": 1,
        "mc_dropout": False, "input_chunk_length": 6, "output_chunk_length": 3,
        "output_chunk_shift": 0, "shared_weights": False, "const_init": True, "normalize": False,
        "use_static_covariates": False, "use_reversible_instance_norm": False,
        "use_cyclic_encoders": use_cyclic_encoders, "target_scaler": "AsinhTransform",
        "feature_scaler": "AsinhTransform", "feature_scaler_map": {}, "targets": ["ged_sb"],
        "prediction_format": "frames",
    }


def _dataset() -> ViewsDataset:
    rng = np.random.default_rng(1)
    T, E = 48, 4
    df = pd.DataFrame({
        "month_id": np.repeat(np.arange(400, 400 + T), E),
        "country_id": np.tile(np.arange(1, E + 1), T),
        "ged_sb": rng.poisson(3, T * E).astype(float),
        "lr_x": rng.normal(size=T * E),
    })
    path = os.path.join(tempfile.mkdtemp(), "cm.parquet")
    df.to_parquet(path, index=False)
    return ViewsDataset(path, targets=["ged_sb"], broadcast_features=True)


@pytest.mark.parametrize("use_cyclic_encoders", [True, False])
def test_train_then_streaming_predict_round_trip(cpu_catalog, use_cyclic_encoders):
    cfg = _config(use_cyclic_encoders)
    ds = _dataset()
    model = ModelCatalog(cfg).get_model("NLinearModel")
    partition = {"train": [400, 400 + 48 - 4], "test": [400 + 48 - 3, 400 + 48 - 1]}
    fc = DartsForecaster(dataset=ds, model=model, partition_dict=partition, random_state=7,
                         target_scaler="AsinhTransform", use_cyclic_encoders=use_cyclic_encoders)
    fc.train()
    preds = fc.predict(0, 3)
    assert list(preds.keys()) == ["ged_sb"]
    if use_cyclic_encoders:
        assert model.encoders is not None and model.encoders.encoding_available
