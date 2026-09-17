"""``tlz_compat`` patches toolz<0.12's ``TlzSpec`` only when the attribute is
missing, and is a no-op otherwise."""

import types

from views_r2darts2.infrastructure import tlz_compat


def test_noop_when_attribute_present(monkeypatch):
    fake = types.ModuleType("tlz._build_tlz")
    fake.TlzSpec = type("TlzSpec", (), {"_uninitialized_submodules": []})
    monkeypatch.setitem(__import__("sys").modules, "tlz._build_tlz", fake)
    monkeypatch.setitem(__import__("sys").modules, "tlz", types.ModuleType("tlz"))
    assert tlz_compat.ensure_tlz_importable() is False


def test_patches_when_attribute_missing(monkeypatch):
    fake = types.ModuleType("tlz._build_tlz")
    fake.TlzSpec = type("TlzSpec", (), {})
    monkeypatch.setitem(__import__("sys").modules, "tlz._build_tlz", fake)
    monkeypatch.setitem(__import__("sys").modules, "tlz", types.ModuleType("tlz"))
    assert tlz_compat.ensure_tlz_importable() is True
    assert fake.TlzSpec._uninitialized_submodules == []


def test_dask_array_imports_under_the_package():
    # whatever toolz the environment resolved to, importing the dataset package must leave dask usable
    import views_r2darts2.dataset  # noqa: F401
    import dask.array as da
    assert float(da.full((2, 2), 1.0, chunks=(1, 2)).sum().compute()) == 4.0
