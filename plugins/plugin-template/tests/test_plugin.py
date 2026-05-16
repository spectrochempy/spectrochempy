# ruff: noqa: S101  # assert allowed in tests

"""Tests for spectrochempy-myplugin."""

from plugin_name import MyPlugin

from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import PluginState
from spectrochempy.api.plugins import check_plugin_compatibility
from spectrochempy.testing.plugins import PluginTestHarness


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
    assert PluginCapability.ACCESSOR in plugin.capabilities


def test_registration():
    """Plugin registers readers, writers, analyses, and accessors."""
    harness = PluginTestHarness()
    harness.register(MyPlugin())

    reader = harness.get_reader("myformat")
    assert reader is not None
    assert reader["description"] == "Read MyFormat files"
    assert reader["extensions"] == [".myf", ".myformat"]

    writer = harness.get_writer("myformat")
    assert writer is not None

    # Analysis contributions via ExtensionRegistry
    analyses = harness.registry.extensions.list_category("analysis")
    assert "my_analysis" in analyses

    accessors = harness.registry.extensions.list_category("accessor")
    assert "myplugin.analysis" in accessors
    assert "my_analysis" in accessors


def test_lifecycle_state():
    """Plugin transitions to ACTIVE after registration."""
    harness = PluginTestHarness()
    harness.register(MyPlugin())
    assert harness.get_plugin_state("myplugin") == PluginState.ACTIVE


def test_analysis_contribution():
    """Plugin analysis can be discovered via capability query."""
    harness = PluginTestHarness()
    harness.register(MyPlugin())

    results = harness.registry.get_by_capability(PluginCapability.ANALYSIS)
    names = [r["name"] for r in results]
    assert "my_analysis" in names


def test_accessor_contribution():
    """Plugin accessor can be discovered via capability query."""
    harness = PluginTestHarness()
    harness.register(MyPlugin())

    results = harness.registry.get_by_capability(PluginCapability.ACCESSOR)
    names = [r["name"] for r in results]
    assert "myplugin.analysis" in names


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
    assert h2.registry.extensions.list_category("analysis") == {}
