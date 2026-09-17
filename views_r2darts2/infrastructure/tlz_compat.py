"""Make ``toolz<0.12`` importable by dask on Python >= 3.11.9.

The platform's ``viewser`` pin caps ``toolz<0.12``. ``toolz`` 0.11.x ships a
``tlz`` lazy-import shim whose custom ``ModuleSpec`` (``TlzSpec``) lacks the
``_uninitialized_submodules`` attribute that CPython's import machinery reads
from 3.11.9 onward, so ``import dask`` (which does ``from tlz import ...``)
dies with ``AttributeError: 'TlzSpec' object has no attribute
'_uninitialized_submodules'``. toolz fixed this in 0.12.1 (2024-01-24).

This module adds the one missing attribute when — and only when — it is
missing. On toolz >= 0.12.1, or when toolz is not installed, it does nothing.
It is imported at the top of :mod:`views_r2darts2.dataset` so it runs before
any dask import in this package.
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


ensure_tlz_importable()
