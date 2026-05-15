# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Plugin testing utilities.

Provides :class:`PluginTestHarness` for writing isolated plugin tests
without touching the global plugin registry or manager.
"""

from __future__ import annotations

from typing import Any

from spectrochempy.plugins.lifecycle import PluginDescriptor
from spectrochempy.plugins.lifecycle import PluginState
from spectrochempy.plugins.manager import PluginManager
from spectrochempy.plugins.registry import PluginRegistry


class PluginTestHarness:
    """
    Isolated test harness for plugin development.

    Creates a fresh :class:`~spectrochempy.plugins.registry.PluginRegistry`
    and :class:`~spectrochempy.plugins.manager.PluginManager` for every
    test, preventing state leakage between tests.

    Typical usage::

        from spectrochempy.testing.plugins import PluginTestHarness


        def test_my_plugin():
            harness = PluginTestHarness()
            harness.register(MyPlugin())

            # Assert contributions are registered
            assert harness.get_reader("myformat") is not None
            assert harness.registry.io.get_reader("myformat") is not None

            # Inspect lifecycle state
            assert harness.get_plugin_state("myplugin") == PluginState.ACTIVE

            # List failed plugins
            assert harness.get_failed_plugins() == {}

    The harness also works as a context manager::

        with PluginTestHarness() as h:
            h.register(MyPlugin())
            ...
    """

    def __init__(self) -> None:
        self.registry = PluginRegistry()
        self.manager = PluginManager(registry=self.registry)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, plugin: Any) -> None:
        """Register *plugin* into the isolated manager."""
        self.manager.register(plugin)

    # ------------------------------------------------------------------
    # Delegation: registry contributions
    # ------------------------------------------------------------------

    def get_reader(self, name: str) -> dict[str, Any] | None:
        """Look up a reader by name."""
        return self.registry.get_reader(name)

    def get_writer(self, name: str) -> dict[str, Any] | None:
        """Look up a writer by name."""
        return self.registry.get_writer(name)

    def get_processor(self, name: str) -> dict[str, Any] | None:
        """Look up a processor by name."""
        return self.registry.get_processor(name)

    def get_visualizer(self, name: str) -> dict[str, Any] | None:
        """Look up a visualizer by name."""
        return self.registry.visualization.get_visualizer(name)

    @property
    def available_readers(self) -> dict[str, dict[str, Any]]:
        """All registered readers."""
        return self.registry.available_readers

    @property
    def available_writers(self) -> dict[str, dict[str, Any]]:
        """All registered writers."""
        return self.registry.available_writers

    @property
    def available_processors(self) -> dict[str, dict[str, Any]]:
        """All registered processors."""
        return self.registry.available_processors

    # ------------------------------------------------------------------
    # Delegation: lifecycle introspection
    # ------------------------------------------------------------------

    def get_plugin_state(self, name: str) -> PluginState | None:
        """Return the current :class:`PluginState` of *name*."""
        return self.manager.get_plugin_state(name)

    def get_plugin_descriptor(self, name: str) -> PluginDescriptor | None:
        """Return a :class:`PluginDescriptor` for *name*."""
        return self.manager.get_plugin_descriptor(name)

    def get_active_plugins(self) -> list[str]:
        """Return names of all ACTIVE plugins."""
        return self.manager.get_active_plugins()

    def get_failed_plugins(self) -> dict[str, str]:
        """Return ``{name: error}`` for all FAILED plugins."""
        return self.manager.get_failed_plugins()

    def has_plugin(self, name: str) -> bool:
        """Check if *name* is registered."""
        return self.manager.has_plugin(name)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> PluginTestHarness:
        return self

    def __exit__(self, *args: object) -> None:
        self.registry.clear()
