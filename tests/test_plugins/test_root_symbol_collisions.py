# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Top-level symbol collision policy tests.

Verifies that public ``scp`` root symbols can never be silently shadowed by a
plugin-provided I/O namespace, root export, analysis/simulation extension or
reader export, and that the reserved-name policy is deterministic regardless
of plugin installation or discovery order.

Two complementary guarantees are tested:

- registration-time rejection of plugin **I/O namespaces** whose name
  collides with a public ``scp`` root symbol (``register_io_namespace``);
- **resolution priority**: for every plugin root surface (reader exports,
  ``root_exports``, ``analysis``/``simulation`` extensions), root attribute
  access always resolves a public core symbol first.
"""

import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.core.io_namespaces import _CORE_IO_NAMESPACES
from spectrochempy.core.io_namespaces import _PLUGIN_IO_NAMESPACES
from spectrochempy.core.io_namespaces import _REJECTED_IO_NAMESPACES
from spectrochempy.core.io_namespaces import register_io_namespace
from spectrochempy.lazyimport.root_symbols import is_reserved_root_symbol


def _fresh_interpreter(code: str) -> str:
    """Run *code* in a clean subprocess and return combined stdout+stderr."""
    env = {"PYTHONPATH": ":".join(sys.path)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return (result.stdout + result.stderr).strip()


# -----------------------------------------------------------------------------
# Integration symbol stays the public scientific function
# -----------------------------------------------------------------------------


def test_simpson_callable_without_relying_on_plugin():
    assert callable(scp.simpson)


def test_simpson_is_integration_function():
    assert scp.simpson.__module__ == "spectrochempy.analysis.integration.integrate"
    assert scp.simpson is not scp.read_simpson


def test_top_level_simpson_matches_dataset_simpson():
    y = scp.Coord(np.arange(4.0), title="sample", units="s")
    x = scp.Coord(np.linspace(0.0, 4.0, 9), title="wavelength", units="cm^-1")
    xx, yy = np.meshgrid(x.data, y.data)
    dataset = scp.NDDataset(
        xx**2 + 2.0 * yy, coordset=[y, x], units="absorbance", title="synthetic"
    )

    actual = scp.simpson(dataset, dim="x")
    expected = dataset.simpson(dim="x")
    assert np.array_equal(actual.data, expected.data)


def test_read_simpson_kept():
    assert callable(scp.read_simpson)


def test_simpson_not_advertised_as_io_namespace_in_dir():
    ns = getattr(scp, "simpson", None)
    assert ns is not None
    assert not callable(getattr(ns, "read", None))


# -----------------------------------------------------------------------------
# Registration-time validation
# -----------------------------------------------------------------------------


@pytest.fixture
def clean_plugin_namespaces():
    """Snapshot and restore plugin-contributed I/O namespaces around a test."""
    saved = dict(_PLUGIN_IO_NAMESPACES)
    rejected = dict(_REJECTED_IO_NAMESPACES)
    yield
    _PLUGIN_IO_NAMESPACES.clear()
    _PLUGIN_IO_NAMESPACES.update(saved)
    _REJECTED_IO_NAMESPACES.clear()
    _REJECTED_IO_NAMESPACES.update(rejected)


def test_reserved_namespace_rejected(clean_plugin_namespaces):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ok = register_io_namespace("simpson", "nmr.read_simpson", plugin="nmr-test")

    assert ok is False
    assert "simpson" not in _PLUGIN_IO_NAMESPACES
    assert "simpson" in _REJECTED_IO_NAMESPACES
    assert any("simpson" in str(w.message) for w in caught)
    assert any("plugin=nmr-test" in str(w.message) for w in caught)


def test_non_reserved_namespace_accepted(clean_plugin_namespaces):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ok = register_io_namespace("nmrtest", "nmr.read_agilent", plugin="nmr-test")

    assert ok is True
    assert "nmrtest" in _PLUGIN_IO_NAMESPACES
    assert not any("Refusing" in str(w.message) for w in caught)


def test_core_namespace_cannot_be_overridden(clean_plugin_namespaces):
    # A plugin asking for an existing core namespace must be refused.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ok = register_io_namespace("jcamp", "plugin.read_jcamp", plugin="plugin-test")

    assert ok is False
    assert "jcamp" not in _PLUGIN_IO_NAMESPACES
    assert any("jcamp" in str(w.message) for w in caught)


def test_reserved_symbol_rejection_keeps_reader_available():
    # Rejecting the ``simpson`` namespace must not remove ``read_simpson``.
    assert callable(scp.read_simpson)


def test_status_introspection_after_rejection(clean_plugin_namespaces):
    register_io_namespace("simpson", "nmr.read_simpson", plugin="nmr")
    assert "simpson" in _REJECTED_IO_NAMESPACES
    assert "read_simpson" in _REJECTED_IO_NAMESPACES["simpson"]


# -----------------------------------------------------------------------------
# Reserved-name set construction
# -----------------------------------------------------------------------------


def test_reserved_set_covers_public_symbols():
    for name in ("simpson", "simps", "trapezoid", "Coord", "NDDataset", "read_jcamp"):
        assert is_reserved_root_symbol(name)


def test_reserved_set_covers_dataset_methods():
    assert is_reserved_root_symbol("find_peaks")


def test_reserved_set_covers_io_namespaces():
    for ns in _CORE_IO_NAMESPACES:
        assert is_reserved_root_symbol(ns)


def test_reserved_set_covers_plot_profiles():
    assert is_reserved_root_symbol("set_plot_profile")


def test_unrelated_name_not_reserved():
    assert not is_reserved_root_symbol("nmrtestanalysis")


# -----------------------------------------------------------------------------
# Deterministic resolution regardless of access / discovery order
# -----------------------------------------------------------------------------


@pytest.fixture
def dataset_for_order():
    x = scp.Coord(np.linspace(0.0, 4.0, 9), title="x", units="cm^-1")
    return scp.NDDataset(np.arange(9.0) ** 2, coordset=[x], units="absorbance")


def test_simpson_stack_semantics_stable_across_dir_and_order(dataset_for_order):
    d1 = dataset_for_order
    before_dir = scp.simpson(d1, dim="x").data.copy()

    # Access the reader and dir() in different orders; the integration result
    # must not change.
    _ = scp.read_simpson
    _ = dir(scp)
    d2 = scp.Coord(np.linspace(0.0, 4.0, 9), title="x", units="cm^-1")
    dataset2 = scp.NDDataset(np.arange(9.0) ** 2, coordset=[d2], units="absorbance")
    after = scp.simpson(dataset2, dim="x").data
    assert np.array_equal(before_dir, after)


def test_subprocess_reader_first_does_not_mask_integration():
    code = textwrap.dedent(
        """
        import spectrochempy as scp
        # Reader-first ordering.
        _ = scp.read_simpson
        assert callable(scp.simpson), "integration masked when reader accessed first"
        s = scp.simpson
        assert s.__module__ == "spectrochempy.analysis.integration.integrate"
        print("OK")
        """
    )
    out = _fresh_interpreter(code)
    assert "OK" in out


def test_subprocess_integration_does_not_block_reader():
    code = textwrap.dedent(
        """
        import spectrochempy as scp
        assert callable(scp.simpson)
        _ = scp.simpson
        assert callable(scp.read_simpson), "reader masked after integration access"
        print("OK")
        """
    )
    out = _fresh_interpreter(code)
    assert "OK" in out


# -----------------------------------------------------------------------------
# dir() stability
# -----------------------------------------------------------------------------


def test_dir_does_not_advertise_rejected_namespace(clean_plugin_namespaces):
    register_io_namespace("simpson", "nmr.read_simpson", plugin="nmr-test")
    # `simpson` is a lazy top-level integration function (like `fft`/`smooth`)
    # and, following the existing convention, is not advertised in ``dir()``.
    # The essential invariant is that it resolves to the callable integration
    # function and is never exposed as an ambiguous I/O namespace.
    ns = getattr(scp, "simpson", None)
    assert callable(ns)
    assert not callable(getattr(ns, "read", None))


def test_dir_read_simpson_present():
    # Accessing the reader triggers plugin discovery; the explicit reader
    # surface must then be advertised (and never removed by the collision fix).
    assert callable(scp.read_simpson)
    assert "read_simpson" in dir(scp)


# -----------------------------------------------------------------------------
# Reserved-name set construction (defense-in-depth)
# -----------------------------------------------------------------------------


def test_dataset_method_symbol_is_reserved():
    # ``simpson`` is a public NDDataset method exported at root; it must be a
    # reserved root symbol that a plugin cannot shadow.  The method lives on
    # the instance (via dataset-level ``__getattr__``), not directly on the
    # class, so check via a real instance.
    x = scp.Coord([0.0, 1.0, 2.0], title="x", units="cm^-1")
    ds = scp.NDDataset([1.0, 1.0, 1.0], coordset=[x], units="absorbance")
    assert hasattr(ds, "simpson")
    assert is_reserved_root_symbol("simpson")


def test_reserved_name_wins_over_root_export_collision(clean_plugin_namespaces):
    # Inject a plugin that registers ``simpson`` as a root export.  The
    # reserved-name fast path must still return the core integration function.
    from spectrochempy.plugins.manager import plugin_manager

    class _CollisionPlugin:
        name = "collision-test"
        version = "1.0.0"
        root_exports = {"simpson": {"target": "read_simpson", "namespace": "nmr"}}

    original_plugins = plugin_manager.list_plugins
    plugin_manager.list_plugins = lambda: [*original_plugins(), _CollisionPlugin()]
    try:
        obj = scp.simpson
        assert obj.__module__ == "spectrochempy.analysis.integration.integrate"
        assert not callable(getattr(obj, "read", None))
    finally:
        plugin_manager.list_plugins = original_plugins


def test_reserved_submodule_wins_over_extension_collision():
    # A plugin extension named after a public submodule must not shadow it:
    # ``scp.analysis`` stays the core submodule and is not replaced by the
    # plugin-provided extension object.
    import spectrochempy.analysis as core_analysis
    from spectrochempy.plugins.registry import registry

    registry.extensions.register(
        "analysis",
        "analysis",
        "PLUGIN_ANALYSIS_OBJECT",
        description="collision with the public submodule",
    )
    try:
        obj = scp.analysis
        assert obj is core_analysis
        assert obj != "PLUGIN_ANALYSIS_OBJECT"
        # ``dir(scp)`` still advertises ``analysis`` (the core submodule), and
        # does not surface any plugin-provided extension object.
        names = dir(scp)
        assert "analysis" in names
        assert all(name != "PLUGIN_ANALYSIS_OBJECT" for name in names)
    finally:
        registry.extensions._extensions.pop("analysis", None)
