"""``tlz_compat`` patches toolz<0.12's ``TlzSpec`` only when the attribute is
missing, leaves an existing attribute alone, is a no-op without toolz — and,
critically, is wired in by ``views_r2darts2.dataset`` itself, not by this file."""

import subprocess
import sys
import textwrap
import types

from views_r2darts2.infrastructure import tlz_compat


def _fake_tlz(monkeypatch, spec_attrs):
    fake = types.ModuleType("tlz._build_tlz")
    fake.TlzSpec = type("TlzSpec", (), spec_attrs)
    monkeypatch.setitem(sys.modules, "tlz", types.ModuleType("tlz"))
    monkeypatch.setitem(sys.modules, "tlz._build_tlz", fake)
    return fake


def test_noop_when_attribute_present_and_leaves_it_untouched(monkeypatch):
    sentinel = ["already-here"]
    fake = _fake_tlz(monkeypatch, {"_uninitialized_submodules": sentinel})
    assert tlz_compat.ensure_tlz_importable() is False
    assert fake.TlzSpec._uninitialized_submodules is sentinel


def test_patches_when_attribute_missing(monkeypatch):
    fake = _fake_tlz(monkeypatch, {})
    assert tlz_compat.ensure_tlz_importable() is True
    assert fake.TlzSpec._uninitialized_submodules == []


def test_noop_and_no_raise_when_tlz_cannot_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "tlz", None)  # makes `import tlz` raise ImportError
    monkeypatch.setitem(sys.modules, "tlz._build_tlz", None)
    assert tlz_compat.ensure_tlz_importable() is False


def test_package_root_wires_the_shims_before_the_manager_path_in_a_fresh_interpreter():
    # In-process, importing this test module already applied the patch, so the
    # wiring can only be observed from a clean interpreter: import the package
    # root, then the manager (whose import chain reaches darts -> lightgbm ->
    # dask.dataframe before it reaches views_r2darts2.dataset), then dask. On toolz>=0.12.1 dask works regardless, so also assert the shim
    # module was actually imported by the package.
    code = textwrap.dedent("""
        import sys
        import views_r2darts2
        assert "views_r2darts2.infrastructure.tlz_compat" in sys.modules, "package root did not import tlz_compat"
        # the manager path imports darts BEFORE views_r2darts2.dataset — the ordering that bit
        from views_r2darts2 import DartsForecastingModelManager  # noqa: F401
        import dask.array as da
        assert float(da.full((2, 2), 1.0, chunks=(1, 2)).sum().compute()) == 4.0
        print("ok")
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stderr[-800:]


def test_dask_named_args_shim_is_idempotent_and_tolerant():
    import dask.utils as du
    first = tlz_compat.ensure_dask_named_args_tolerant()
    assert tlz_compat.ensure_dask_named_args_tolerant() is False  # second call: no-op
    # whether or not a wrap was needed on this interpreter, the call must not raise
    assert du.get_named_args(property(lambda self: None)) == []
    assert du.get_named_args(lambda a, b, *c, **d: None) == ["a", "b"]
    assert first in (True, False)


def test_lightgbm_and_darts_models_import_under_the_package_in_a_fresh_interpreter():
    # lightgbm is optional in this package's environment; when present, darts
    # imports it and it imports dask.dataframe — the chain the second shim guards
    code = textwrap.dedent("""
        import importlib.util, sys
        from views_r2darts2 import DartsForecastingModelManager  # noqa: F401  (darts first, then dataset)
        if importlib.util.find_spec("lightgbm") is not None:
            import lightgbm  # noqa: F401
        import darts.models  # noqa: F401
        print("ok")
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stderr[-800:]
