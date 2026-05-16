# ruff: noqa: S101  # assert allowed in tests

"""Tests for spectrochempy-topspin plugin registration and lifecycle."""

from spectrochempy_topspin import TopSpinPlugin

from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import PluginState
from spectrochempy.api.plugins import check_plugin_compatibility
from spectrochempy.testing.plugins import PluginTestHarness


def test_plugin_metadata():
    """Plugin has required metadata fields."""
    plugin = TopSpinPlugin()
    assert plugin.name == "topspin"
    assert plugin.version == "0.1.0"
    assert plugin.description
    assert PluginCapability.READER in plugin.capabilities


def test_plugin_compatibility():
    """Plugin passes compatibility checks."""
    plugin = TopSpinPlugin()
    issues = check_plugin_compatibility(plugin)
    assert not issues, f"Compatibility issues: {issues}"


def test_registration():
    """Plugin registers the topspin reader via declarative hook."""
    harness = PluginTestHarness()
    harness.register(TopSpinPlugin())

    reader = harness.get_reader("topspin")
    assert reader is not None
    assert reader["description"]


def test_lifecycle_state():
    """Plugin transitions to ACTIVE after registration."""
    harness = PluginTestHarness()
    harness.register(TopSpinPlugin())
    assert harness.get_plugin_state("topspin") == PluginState.ACTIVE


def test_isolated_harness():
    """Each PluginTestHarness is independent."""
    h1 = PluginTestHarness()
    h2 = PluginTestHarness()

    h1.register(TopSpinPlugin())
    assert h1.has_plugin("topspin")
    assert h2.registry.metadata.get_plugin("topspin") is None
