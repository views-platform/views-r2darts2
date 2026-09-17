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


def test_dataset_package_wires_the_shim_in_a_fresh_interpreter():
    # In-process, importing this test module already applied the patch, so the
    # wiring can only be observed from a clean interpreter: import the dataset
    # package (and nothing else from the test side), then require that dask is
    # usable. On toolz>=0.12.1 dask works regardless, so also assert the shim
    # module was actually imported by the package.
    code = textwrap.dedent("""
        import sys
        import views_r2darts2.dataset
        assert "views_r2darts2.infrastructure.tlz_compat" in sys.modules, "dataset package did not import tlz_compat"
        import dask.array as da
        assert float(da.full((2, 2), 1.0, chunks=(1, 2)).sum().compute()) == 4.0
        print("ok")
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stderr[-800:]
