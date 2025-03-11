# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

from abc import ABC
from abc import abstractmethod
from importlib.metadata import entry_points

from spectrochempy.application.application import debug_
from spectrochempy.application.application import warning_


# Define the base class for SpectroChemPy plugins
class SpectroChemPyPlugin(ABC):
    """Base class for SpectroChemPy plugins."""

    _is_initialized = False
    _auto_initialize = False

    @abstractmethod
    def initialize(self, manager) -> bool:
        """Initialize the plugin."""
        # Check if dependencies are installed
        import importlib

        for dep, pkg in self.require_dependencies.items():
            try:
                importlib.import_module(pkg)
            except ImportError:
                debug_(
                    f"Required dependency {dep} not found. Install with: pip install {dep}"
                )
                return False

        # check if plugins dependencies are installed
        for dep in self.require_plugins:
            plugin = manager.get_plugin(dep)
            if not plugin:
                debug_(
                    f"`{self.name}` plugin requires the `{dep}` plugin. "
                    f"Install with: pip install spectrochempy[{dep}]"
                )
                return False

        return True

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

    @property
    def require_dependencies(self) -> dict[str:str]:
        """Dict of required package dependencies with eventual renaming."""
        return {}

    @property
    def require_plugins(self) -> list[str]:
        """List of required plugins."""
        return []

    @property
    def enabled(self) -> bool:
        """Check if the plugin is enabled."""
        if not self._is_initialized:
            warning_(
                f"Plugin `{self.name}` is not installed or not initialized. "
                f"If not installed : install with ``pip install spectrochempy[{self.name}]``"
            )
        return self._is_initialized

    @property
    def auto_initialize(self) -> bool:
        """Check if the plugin should be automatically initialized."""
        return self._auto_initialize


# Define the plugin manager
class PluginManager:
    """Manages SpectroChemPy plugins."""

    def __init__(self):
        self._plugins: dict[str, SpectroChemPyPlugin] = {}

        # # Discover all existing plugins
        # self.discover_plugins()

        # # Initialize all discovered plugins (if enabled in preferences)
        # self.initialize_plugins()

    def discover_plugins(self):
        """Discover and load all available plugins."""
        discovered_plugins = entry_points(group="spectrochempy.plugins")

        for entry_point in discovered_plugins:
            try:
                plugin_class: type[SpectroChemPyPlugin] = entry_point.load()
                plugin = plugin_class()
                self._plugins[plugin.name] = plugin
            except Exception as e:
                warning_(f"Failed to load plugin {entry_point.name}: {e}")

    def get_plugin(self, name: str) -> SpectroChemPyPlugin:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def __getattr__(self, item):
        return self._plugins.get(item)

    @property
    def available_plugins(self) -> list[SpectroChemPyPlugin]:
        """Dictionary of available plugins."""
        return self._plugins
