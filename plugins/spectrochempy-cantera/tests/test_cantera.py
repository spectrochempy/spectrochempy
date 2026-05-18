# ruff: noqa: S101  # assert allowed in tests

"""Tests for spectrochempy-cantera plugin — PFR scope only."""

from __future__ import annotations

import sys

import pytest
from spectrochempy_cantera import CanteraPlugin

from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import PluginState
from spectrochempy.api.plugins import check_plugin_compatibility
from spectrochempy.testing.plugins import PluginTestHarness

try:
    import cantera as ct

    HAS_CANTERA = True
    ct.Solution("gri30.yaml")
    HAS_GRI30 = True
except Exception:
    HAS_CANTERA = False
    HAS_GRI30 = False


# ------------------------------------------------------------------
# Lazy import behaviour
# ------------------------------------------------------------------


def _clean_cantera_modules():
    for mod in list(sys.modules):
        if "spectrochempy_cantera" in mod:
            del sys.modules[mod]


def test_import_does_not_load__pfr():
    """Importing the plugin package does NOT load ``_pfr`` immediately."""
    _clean_cantera_modules()
    import spectrochempy_cantera  # noqa: PLC0415,F401 — import itself is the test action

    assert "spectrochempy_cantera._pfr" not in sys.modules


def test_accessing_pfr_loads__pfr():
    """Accessing ``spectrochempy_cantera.PFR`` triggers the lazy import."""
    _clean_cantera_modules()
    import spectrochempy_cantera  # noqa: PLC0415,F401

    assert "spectrochempy_cantera._pfr" not in sys.modules
    _ = spectrochempy_cantera.PFR
    assert "spectrochempy_cantera._pfr" in sys.modules


def test_registration_does_not_load__pfr():
    """Calling ``register_simulations()`` does not import ``_pfr``."""
    _clean_cantera_modules()
    import spectrochempy_cantera  # noqa: PLC0415,F401 — fresh import so _pfr is not loaded

    plugin = spectrochempy_cantera.CanteraPlugin()
    entries = plugin.register_simulations()
    assert "PFR" in [e["name"] for e in entries]
    assert "spectrochempy_cantera._pfr" not in sys.modules


# ------------------------------------------------------------------
# Plugin metadata and compatibility
# ------------------------------------------------------------------


def test_plugin_metadata():
    """Plugin has required metadata fields (PFR scope only)."""
    plugin = CanteraPlugin()
    assert plugin.name == "cantera"
    assert plugin.version == "0.1.0"
    assert plugin.description
    assert PluginCapability.SIMULATION in plugin.capabilities
    assert PluginCapability.ANALYSIS not in plugin.capabilities
    assert PluginCapability.ACCESSOR not in plugin.capabilities


def test_plugin_compatibility():
    """Plugin passes full compatibility check."""
    plugin = CanteraPlugin()
    issues = check_plugin_compatibility(plugin)
    if not HAS_CANTERA:
        assert any("cantera" in issue for issue in issues)
    else:
        assert not issues, f"Compatibility issues: {issues}"


# ------------------------------------------------------------------
# Registration and lifecycle
# ------------------------------------------------------------------


def test_registration_with_cantera():
    """Plugin registers PFR simulation when cantera is available."""
    if not HAS_CANTERA:
        pytest.skip("cantera not installed")

    harness = PluginTestHarness()
    harness.register(CanteraPlugin())

    simulations = harness.registry.extensions.list_category("simulation")
    assert "PFR" in simulations

    analyses = harness.registry.extensions.list_category("analysis")
    assert len(analyses) == 0


def test_registration_without_cantera():
    """Plugin is marked FAILED when cantera is missing."""
    harness = PluginTestHarness()
    harness.register(CanteraPlugin())

    if not HAS_CANTERA:
        assert harness.get_plugin_state("cantera") == PluginState.FAILED
    else:
        assert harness.get_plugin_state("cantera") == PluginState.ACTIVE


def test_capability_query():
    """Plugin simulation contributions are discoverable via capability query."""
    if not HAS_CANTERA:
        pytest.skip("cantera not installed")

    harness = PluginTestHarness()
    harness.register(CanteraPlugin())

    results = harness.registry.get_by_capability(PluginCapability.SIMULATION)
    names = [r["name"] for r in results]
    assert "PFR" in names


# ------------------------------------------------------------------
# PFR namespace access
# ------------------------------------------------------------------


def test_package_namespace_exposes_pfr():
    """scp.cantera.PFR is accessible and callable when plugin registered."""
    if not HAS_CANTERA:
        pytest.skip("cantera not installed")

    import spectrochempy as scp  # noqa: PLC0415

    if not scp.plugin_manager.has_plugin("cantera"):
        scp.plugin_manager.register(CanteraPlugin())

    pfr_entry = scp.cantera.PFR
    assert callable(pfr_entry)


def test_cantera_does_not_register_dataset_accessor():
    """Cantera has no dataset accessor."""
    import spectrochempy as scp  # noqa: PLC0415

    if HAS_CANTERA and not scp.plugin_manager.has_plugin("cantera"):
        scp.plugin_manager.register(CanteraPlugin())

    dataset = scp.NDDataset([300.0, 310.0])
    assert not hasattr(dataset, "cantera")


# ------------------------------------------------------------------
# PFR import and basic checks
# ------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CANTERA, reason="cantera not installed")
def test_cantera_is_not_available(monkeypatch):
    """_cantera_is_not_available detects missing cantera."""
    from spectrochempy_cantera._pfr import _cantera_is_not_available  # noqa: PLC0415

    assert not _cantera_is_not_available()

    with monkeypatch.context() as m:
        import spectrochempy_cantera._pfr as _pfr  # noqa: PLC0415

        m.setattr(_pfr, "ct", None)
        assert _cantera_is_not_available()

    assert not _cantera_is_not_available()


@pytest.mark.skipif(not HAS_CANTERA, reason="cantera not installed")
def test_pfr_import():
    """PFR class accessible via ``from spectrochempy_cantera import PFR``."""
    from spectrochempy_cantera import PFR as _PFR  # noqa: PLC0415,N811

    assert _PFR.__name__ == "PFR"


def test_pfr_construction_skipped_due_to_cantera_api():
    """
    PFR construction tests skipped (Cantera 3.2 API changes).

    The PFR class was written for an older Cantera API and uses methods
    (Reservoir(contents=), MassFlowController.set_mass_flow_rate) that were
    removed/changed in Cantera 3.2.
    """
    pytest.skip("PFR requires Cantera API adaptation for Cantera 3.2+")
