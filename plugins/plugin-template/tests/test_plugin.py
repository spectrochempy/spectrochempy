"""Tests for spectrochempy-myplugin."""

from spectrochempy.api.plugins import (
    PluginCapability,
    PluginState,
    check_plugin_compatibility,
)
from spectrochempy.testing.plugins import PluginTestHarness

from plugin_name import MyPlugin


def test_plugin_metadata():
    """Plugin has required metadata fields."""
    plugin = MyPlugin()
    assert plugin.name == "myplugin"
    assert plugin.version == "0.1.0"
    assert plugin.description


def test_plugin_compatibility():
    """Plugin passes compatibility checks."""
    plugin = MyPlugin()
    issues = check_plugin_compatibility(plugin)
    assert not issues, f"Compatibility issues: {issues}"


def test_plugin_capabilities():
    """Plugin advertises correct capabilities."""
    plugin = MyPlugin()
    assert hasattr(plugin, "capabilities")
    assert PluginCapability.READER in plugin.capabilities


def test_registration():
    """Plugin registers readers and writers successfully."""
    harness = PluginTestHarness()
    harness.register(MyPlugin())

    # Reader should be registered
    reader = harness.get_reader("myformat")
    assert reader is not None
    assert reader["description"] == "Read MyFormat files"
    assert reader["extensions"] == [".myf", ".myformat"]

    # Writer should be registered
    writer = harness.get_writer("myformat")
    assert writer is not None
    assert writer["description"] == "Write MyFormat files"


def test_lifecycle_state():
    """Plugin transitions to ACTIVE after registration."""
    harness = PluginTestHarness()
    harness.register(MyPlugin())
    assert harness.get_plugin_state("myplugin") == PluginState.ACTIVE


def test_declarative_hooks():
    """Plugin contributions can be inspected via sub-registries."""
    harness = PluginTestHarness()
    harness.register(MyPlugin())

    assert harness.registry.io.get_reader("myformat") is not None
    assert harness.registry.io.get_writer("myformat") is not None


def test_isolated_harness():
    """Each PluginTestHarness is independent."""
    h1 = PluginTestHarness()
    h2 = PluginTestHarness()

    h1.register(MyPlugin())
    assert h1.has_plugin("myplugin")
    assert not h2.has_plugin("myplugin")
