"""Lifetime of the prediction memmap scratch directory (issue #54).

``DartsForecaster._predict_streaming`` writes one ``.npy`` per target into a
scratch directory and hands back ``PredictionFrame``s whose ``values`` are
``np.memmap``s pointing into it. Nothing used to free that directory: ~4 000 of
them at ~53 GB each filled fimbulthul's 2 TB disk on 2026-09-20 and killed an
unrelated run.

The contract these tests pin is deliberately **not** the ``ZarrStore`` one.
``ZarrStore`` frees its directory on garbage collection, which is safe because
nobody else holds a handle to its bytes. Here the caller holds live memmaps, so
cleanup is tied to two events only — an explicit ``close()`` once the values
have been copied out, and ``atexit`` as the backstop. A test that would pass
under a ``try/finally: rmtree`` in ``_predict_streaming`` would be worthless,
so ``test_frames_stay_readable_until_release`` exists to reject exactly that.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

# views_r2darts2 must be imported before darts: its package root installs the
# tlz/dask compatibility shims that darts' lightgbm import chain needs on
# Python >= 3.11.9. Importing darts first wedges collection when this module is
# run on its own.
from views_r2darts2.dataset.base import ViewsDataset
from views_r2darts2.engines.darts_forecaster import DartsForecaster
from views_r2darts2.transformers.frame_builder import PredictionScratch

import torch  # noqa: E402 — after the shim, see above
from darts.models.forecasting.torch_forecasting_model import (  # noqa: E402
    TorchForecastingModel,
)

# Mirrors tests/test_darts_forecaster.py so the harness is shared, not re-invented.
TARGETS: list[str] = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
ENTITY_IDS: list[int] = [1, 2, 3]
PARTITION: dict[str, tuple[int, int]] = {"train": (121, 200), "test": (201, 220)}


@pytest.fixture(scope="module")
def dataset(synthetic_cm_parquet_small: Path) -> ViewsDataset:
    """The shared synthetic cm dataset, subset to three entities."""
    full_ds = ViewsDataset(
        synthetic_cm_parquet_small,
        targets=TARGETS,
        broadcast_features=True,
    )
    return full_ds.get_subset_dataset(entity_ids=ENTITY_IDS)


def _make_mock_model() -> Mock:
    """A ``Mock(spec=TorchForecastingModel)`` wired for the streaming predict path.

    ``parameters`` uses ``side_effect`` so every call yields a fresh iterator —
    the device check may read it twice.
    """
    m = Mock(spec=TorchForecastingModel)
    m.input_chunk_length = 12
    m.output_chunk_length = 6
    m.model = Mock()
    m.model.parameters.side_effect = lambda: iter([Mock(device=torch.device("cpu"))])

    n_entities, n_time, n_targets, n_samples = len(ENTITY_IDS), 6, len(TARGETS), 1
    m.predict_from_dataset.return_value = (
        np.full((n_entities, n_time, n_targets, n_samples), 0.5, dtype=np.float32),
        [{}] * n_entities,
        list(range(n_entities)),
    )
    m._build_inference_dataset.return_value = Mock()
    return m


def _make_ready_forecaster(dataset: ViewsDataset) -> DartsForecaster:
    """A forecaster with scalers fitted, ready to ``predict``."""
    fc = DartsForecaster(
        dataset=dataset,
        model=_make_mock_model(),
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
    return fc


# ----------------------------------------------------------------------
# PredictionScratch — the owner itself
# ----------------------------------------------------------------------


def test_close_removes_the_directory(tmp_path: Path) -> None:
    """``close()`` deletes the scratch directory and leaves its parent alone."""
    scratch = PredictionScratch(base_dir=tmp_path)
    path = scratch.path
    assert path.exists()
    assert path.parent == tmp_path

    scratch.close()

    assert not path.exists(), f"scratch still exists at {path}"
    assert tmp_path.exists(), "close() must not remove the parent directory"


def test_close_is_idempotent(tmp_path: Path) -> None:
    """A second ``close()`` is a no-op, not an error.

    The manager may release after conversion and ``atexit`` will fire again at
    exit; both must be safe.
    """
    scratch = PredictionScratch(base_dir=tmp_path)
    path = scratch.path

    scratch.close()
    scratch.close()

    assert not path.exists()


def test_atexit_frees_a_scratch_left_open(tmp_path: Path) -> None:
    """A scratch never closed is still removed when the process exits.

    This is the backstop that would have prevented the fimbulthul incident,
    which was accumulation across *completed* runs. Only observable from
    outside the process, so this runs a child (the pattern
    ``tests/test_tlz_compat.py`` uses for interpreter-level behaviour).
    """
    marker = tmp_path / "scratch_path.txt"
    code = (
        "from pathlib import Path\n"
        "from views_r2darts2.transformers.frame_builder import PredictionScratch\n"
        f"s = PredictionScratch(base_dir=Path({str(tmp_path)!r}))\n"
        f"Path({str(marker)!r}).write_text(str(s.path))\n"
        "(s.path / 'payload.bin').write_bytes(b'x' * 1024)\n"
        "# deliberately no close() — exit is the only cleanup signal\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr

    leaked = Path(marker.read_text().strip())
    assert not leaked.exists(), (
        f"scratch survived interpreter exit at {leaked} — atexit backstop missing"
    )


# ----------------------------------------------------------------------
# The forecaster / manager contract
# ----------------------------------------------------------------------


def test_frames_stay_readable_until_release(dataset: ViewsDataset) -> None:
    """Frames must be readable after ``predict`` returns and before release.

    This rejects the tempting wrong fix — a ``try/finally: rmtree`` inside
    ``_predict_streaming`` — which would delete the files while the returned
    memmaps still point at them.
    """
    fc = _make_ready_forecaster(dataset)
    frames = fc.predict(sequence_number=0, output_length=6)

    scratch_paths = [s.path for s in fc._prediction_scratch]
    assert len(scratch_paths) == 1, "one predict() call should own one scratch"
    assert scratch_paths[0].exists()

    # The values must still be reachable through the memmap.
    for target, frame in frames.items():
        assert isinstance(frame.values, np.memmap), f"{target} lost its memmap"
        row = np.asarray(frame.values[0, :])
        assert np.isfinite(row).all()

    freed = fc.release_prediction_scratch()

    assert freed == 1
    assert not scratch_paths[0].exists(), (
        f"scratch still exists after release at {scratch_paths[0]}"
    )


def test_release_leaves_no_scratch_directory_behind(dataset: ViewsDataset) -> None:
    """A predict-then-release cycle adds no surviving ``pred_frames_*`` directory.

    Measured the way the incident was measured — by looking at the temp root —
    so it holds regardless of how the directory is created.
    """
    temp_root = Path(tempfile.gettempdir())
    before = set(temp_root.glob("pred_frames_*"))

    fc = _make_ready_forecaster(dataset)
    fc.predict(sequence_number=0, output_length=6)
    fc.release_prediction_scratch()

    survivors = set(temp_root.glob("pred_frames_*")) - before
    assert not survivors, f"leaked scratch directories: {sorted(survivors)}"


def test_release_is_safe_with_nothing_to_release(dataset: ViewsDataset) -> None:
    """Releasing before any predict returns zero rather than raising."""
    fc = _make_ready_forecaster(dataset)
    assert fc.release_prediction_scratch() == 0


# ----------------------------------------------------------------------
# The manager's format guard — when releasing is safe, and when it is not
# ----------------------------------------------------------------------


def _bare_manager() -> Any:
    """A manager instance with no __init__ run, for testing one method."""
    from views_r2darts2.engines.darts_forecasting_model_manager import (
        DartsForecastingModelManager,
    )

    return DartsForecastingModelManager.__new__(DartsForecastingModelManager)


def test_manager_releases_scratch_when_predictions_became_dataframes() -> None:
    """On the DataFrame path the values are copied, so the scratch is freed."""
    from views_r2darts2.engines.darts_forecasting_model_manager import (
        DartsForecastingModelManager,
    )

    mgr = _bare_manager()
    forecaster = Mock()
    with patch.object(
        DartsForecastingModelManager,
        "_get_prediction_format",
        return_value="dataframe",
    ):
        mgr._release_scratch_if_frames_copied(forecaster)

    forecaster.release_prediction_scratch.assert_called_once_with()


def test_manager_keeps_scratch_when_frames_are_handed_out() -> None:
    """On the prediction_frame path the scratch must NOT be freed.

    The frames leave this package as live memmaps and views-pipeline-core reads
    them afterwards. Releasing here would delete the files underneath them, so
    those directories are left to the atexit backstop.
    """
    from views_r2darts2.engines.darts_forecasting_model_manager import (
        DartsForecastingModelManager,
    )

    mgr = _bare_manager()
    forecaster = Mock()
    with patch.object(
        DartsForecastingModelManager,
        "_get_prediction_format",
        return_value="prediction_frame",
    ):
        mgr._release_scratch_if_frames_copied(forecaster)

    forecaster.release_prediction_scratch.assert_not_called()
