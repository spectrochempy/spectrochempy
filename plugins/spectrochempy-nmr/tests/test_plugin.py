# ruff: noqa: S101  # assert allowed in tests

"""Tests for spectrochempy-nmr plugin registration and lifecycle."""

import warnings

import pytest
from spectrochempy_nmr import NMRPlugin

import spectrochempy as scp
import spectrochempy.plugins.manager as manager_module
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import PluginState
from spectrochempy.api.plugins import check_plugin_compatibility
from spectrochempy.core.readers.read_topspin import read_topspin as core_read_topspin
from spectrochempy.plugins.deps import MissingPluginError
from spectrochempy.plugins.manager import ENTRY_POINT_GROUP
from spectrochempy.plugins.manager import PluginManager
from spectrochempy.plugins.registry import PluginRegistry
from spectrochempy.testing.plugins import PluginTestHarness


def _require_reader_dependencies() -> None:
    pytest.importorskip("quaternion")


def _isolate_scp_plugins(monkeypatch, scp):
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    monkeypatch.setattr(
        manager_module.importlib.metadata,
        "entry_points",
        lambda group=None: [],
    )
    monkeypatch.setattr(scp, "plugin_manager", pm)
    monkeypatch.setattr(scp, "registry", registry)
    return pm, registry


def test_plugin_metadata():
    """Plugin has required metadata fields."""
    plugin = NMRPlugin()
    assert plugin.name == "nmr"
    assert plugin.version == "0.1.0"
    assert plugin.description
    assert PluginCapability.READER in plugin.capabilities


def test_plugin_compatibility():
    """Plugin passes compatibility checks."""
    plugin = NMRPlugin()
    issues = check_plugin_compatibility(plugin)
    assert not issues, f"Compatibility issues: {issues}"


def test_registration():
    """Plugin registers the topspin reader via declarative hook."""
    _require_reader_dependencies()
    harness = PluginTestHarness()
    harness.register(NMRPlugin())

    reader = harness.get_reader("topspin")
    assert reader is not None
    assert reader["description"]
    assert reader["plugin"] == "nmr"
    assert reader["namespace"] == "nmr"


def test_registration_is_idempotent():
    """Registering the same NMR plugin twice does not duplicate contributions."""
    _require_reader_dependencies()
    harness = PluginTestHarness()
    plugin = NMRPlugin()

    harness.register(plugin)
    first_reader = harness.get_reader("topspin")
    harness.register(plugin)
    second_reader = harness.get_reader("topspin")

    assert harness.get_plugin_state("nmr") == PluginState.ACTIVE
    assert list(harness.available_readers).count("topspin") == 1
    assert second_reader is not None
    assert first_reader is not None
    assert second_reader["func"] is first_reader["func"]


def test_discovery_is_idempotent(monkeypatch):
    """Repeated discovery keeps a single active NMR reader and namespace."""
    _require_reader_dependencies()

    class NMREntryPoint:
        name = "nmr"
        value = "spectrochempy_nmr:NMRPlugin"

        @staticmethod
        def load():
            return NMRPlugin

    def mock_entry_points(group=None):
        if group == ENTRY_POINT_GROUP:
            return [NMREntryPoint()]
        return []

    monkeypatch.setattr(
        manager_module.importlib.metadata,
        "entry_points",
        mock_entry_points,
    )

    pm = PluginManager()
    pm.discover()
    pm.discover()

    assert pm.get_plugin_state("nmr") == PluginState.ACTIVE
    assert list(pm.registry.available_readers).count("topspin") == 1
    assert pm.registry.get_reader("topspin")["namespace"] == "nmr"


def test_lifecycle_state():
    """Plugin transitions to ACTIVE after registration."""
    _require_reader_dependencies()
    harness = PluginTestHarness()
    harness.register(NMRPlugin())
    assert harness.get_plugin_state("nmr") == PluginState.ACTIVE


def test_isolated_harness():
    """Each PluginTestHarness is independent."""
    _require_reader_dependencies()
    h1 = PluginTestHarness()
    h2 = PluginTestHarness()

    h1.register(NMRPlugin())
    assert h1.has_plugin("nmr")
    assert h2.registry.metadata.get_plugin("nmr") is None


def test_package_namespace_exposes_topspin_reader(monkeypatch):
    """scp.nmr.read_topspin exposes the reader while preserving legacy alias."""
    _require_reader_dependencies()

    pm, registry = _isolate_scp_plugins(monkeypatch, scp)
    pm.register(NMRPlugin())

    assert scp.nmr.read_topspin is scp.read_topspin
    assert registry.get_reader("topspin")["namespace"] == "nmr"
    assert scp.read_topspin.__module__ == "spectrochempy_nmr"
    assert scp.nmr.read_topspin.__module__ == "spectrochempy_nmr"


def test_top_level_stub_is_actionable_without_registered_nmr(monkeypatch):
    """scp.read_topspin is a callable MissingPluginError stub without NMR."""
    _isolate_scp_plugins(monkeypatch, scp)

    read_topspin = scp.read_topspin
    assert callable(read_topspin)
    with pytest.raises(MissingPluginError) as excinfo:
        read_topspin("missing")

    message = str(excinfo.value)
    assert "spectrochempy-nmr" in message
    assert "pip install spectrochempy[nmr]" in message


def test_core_stub_is_actionable():
    """The core compatibility stub explains how to install the NMR plugin."""
    try:
        core_read_topspin("missing")
    except MissingPluginError as err:
        message = str(err)
    else:  # pragma: no cover
        raise AssertionError("read_topspin stub should require spectrochempy-nmr")

    assert "spectrochempy-nmr" in message
    assert "pip install spectrochempy[nmr]" in message


def test_read_topspin_no_future_warning():
    """scp.read_topspin and scp.nmr.read_topspin do not emit FutureWarning."""
    _require_reader_dependencies()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", FutureWarning)
        rt = scp.read_topspin
        rt_ns = scp.nmr.read_topspin

    assert callable(rt)
    assert rt is rt_ns
    future_warnings = [w for w in captured if issubclass(w.category, FutureWarning)]
    assert (
        future_warnings == []
    ), f"Expected no FutureWarning from read_topspin, got: {future_warnings}"


def test_nmr_reader_is_not_dataset_accessor_namespace(monkeypatch):
    """Readers are package-level APIs, not dataset accessor methods."""
    _require_reader_dependencies()

    pm, _registry = _isolate_scp_plugins(monkeypatch, scp)
    pm.register(NMRPlugin())

    dataset = scp.NDDataset([1, 2, 3])
    assert not hasattr(dataset, "nmr")
    assert not hasattr(dataset, "read_topspin")
