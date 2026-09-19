"""Make the platform's pinned dask stack importable on Python >= 3.11.9.

The VIEWS platform pins ``pandas<2`` (via ``viewser`` and
``views-transformation-library``), which confines dask to <=2024.2 — the last
line whose ``dask.dataframe`` accepts pandas 1 — and ``toolz<0.12``. Both of
those predate two CPython changes that landed in 3.11.9, and each breaks an
import chain this package sits on:

1. ``toolz`` 0.11.x's ``tlz`` lazy-import shim defines a ``ModuleSpec``
   subclass (``TlzSpec``) without the ``_uninitialized_submodules`` attribute
   the importer now reads, so ``import dask`` dies with
   ``AttributeError: 'TlzSpec' object has no attribute '_uninitialized_submodules'``.
   Fixed upstream in toolz 0.12.1 (2024-01-24), which viewser excludes.

2. ``dask.utils.get_named_args`` calls ``inspect.signature`` on pandas
   accessor attributes while building docstrings; on a ``property`` object
   ``inspect.signature`` now raises ``TypeError`` (it raised ``ValueError``
   before 3.11.9), so ``import dask.dataframe`` dies in ``accessor.py``.
   This package no longer imports ``dask.dataframe`` itself, but darts 0.40
   imports ``lightgbm`` when it is installed (it is, via views-stepshifter),
   and lightgbm imports ``dask.dataframe`` at module level — catching only
   ``ImportError``/``ValueError``. Fixed upstream in dask 2024.4, which
   requires pandas>=2.

Each shim applies its one-line fix only when the defect is present, and is a
no-op otherwise. The module is imported at the top of the package
``__init__`` so it runs before any lazy import can reach dask.
"""

from __future__ import annotations


def ensure_tlz_importable() -> bool:
    """Patch ``tlz._build_tlz.TlzSpec`` if it predates the upstream fix.

    Returns ``True`` if a patch was applied, ``False`` if none was needed.
    """
    try:
        from tlz import _build_tlz  # type: ignore[import-not-found]
    except Exception:  # toolz not installed, or a tlz that cannot even import
        return False
    spec_cls = getattr(_build_tlz, "TlzSpec", None)
    if spec_cls is None or hasattr(spec_cls, "_uninitialized_submodules"):
        return False
    spec_cls._uninitialized_submodules = []  # what importlib expects to find
    return True


def ensure_dask_named_args_tolerant() -> bool:
    """Wrap ``dask.utils.get_named_args`` to return ``[]`` on ``TypeError``.

    Only the docstring-decoration path (``derived_from``) calls it with
    ``property`` objects, so an empty list there loses nothing but a note in
    a generated docstring. Returns ``True`` if the wrap was applied, ``False``
    if dask is absent or already tolerant.
    """
    try:
        import dask.utils as dask_utils
    except Exception:
        return False
    original = getattr(dask_utils, "get_named_args", None)
    if original is None or getattr(original, "_views_tolerant", False):
        return False
    try:  # already fine on this interpreter/dask combination?
        original(property(lambda self: None))
        return False
    except TypeError:
        pass
    except Exception:
        return False

    def get_named_args(func):
        try:
            return original(func)
        except TypeError:
            return []

    get_named_args._views_tolerant = True  # type: ignore[attr-defined]
    get_named_args.__wrapped__ = original  # type: ignore[attr-defined]
    dask_utils.get_named_args = get_named_args
    return True


ensure_tlz_importable()
ensure_dask_named_args_tolerant()
