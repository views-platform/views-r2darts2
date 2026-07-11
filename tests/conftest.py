"""Top-level pytest configuration.

Captures the clean ``torch.load`` reference before any monkey-patches
(``apply_all_patches``) overwrite it. The legacy ``conftest`` did the same;
we preserve the invariant because ``DartsForecastingModelManager`` and
``patches.apply_torch_load_patch`` depend on it.

Google Python Style.
"""

from __future__ import annotations

import logging

import torch

# Capture the pristine torch.load BEFORE any test imports a module that calls
# ``apply_all_patches()``. The captured reference is used by the patch itself
# to fall back to the original behavior when the test session is over.
CLEAN_TORCH_LOAD = torch.load


def pytest_sessionstart(session) -> None:  # noqa: ANN001
    """Workspace-integrity check: ensure ``views_r2darts2`` resolves locally."""
    import os
    import sys

    # Find the project root (the directory containing this conftest.py).
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Verify the package is importable from the local source tree (not from a
    # pip-installed location that might be stale).
    import views_r2darts2

    pkg_path = os.path.dirname(os.path.abspath(views_r2darts2.__file__))
    if not pkg_path.startswith(project_root):
        raise SystemExit(
            f"Workspace integrity check FAILED: views_r2darts2 resolves to "
            f"{pkg_path}, expected under {project_root}. "
            "Run pytest from the project root."
        )

    # Quiet down the noisy loggers during tests.
    logging.getLogger("darts").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    print("\n✅ Workspace Integrity Verified (pandas-free)")
