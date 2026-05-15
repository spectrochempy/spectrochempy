# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

from __future__ import annotations

import importlib.metadata
import logging
from typing import Any

import pluggy

from spectrochempy.api.plugins.validation import validate_plugin_compatibility
from spectrochempy.plugins.hooks import SpectroChemPyHookSpec
from spectrochempy.plugins.lifecycle import PluginDescriptor
from spectrochempy.plugins.lifecycle import PluginState
from spectrochempy.plugins.registry import registry as _default_registry

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "spectrochempy.plugins"


class PluginManager:
    """
    Orchestrates plugin discovery, validation, registration, and lifecycle.

    Parameters
    ----------
    registry : PluginRegistry or None
        Registry instance to use.  When ``None`` (default), the module-
        level singleton is used for backward compatibility.

    The manager tracks every plugin through an explicit lifecycle
    (:class:`~spectrochempy.plugins.lifecycle.PluginState`).  Errors
    during loading or registration are isolated: a single failing
    plugin never crashes discovery or other plugins.
    """

    def __init__(self, registry: Any = None) -> None:
        self._pm = pluggy.PluginManager("spectrochempy")
        self._pm.add_hookspecs(SpectroChemPyHookSpec)
        self._discovered = False

        if registry is None:
            registry = _default_registry
        self.registry = registry

        # --- Lifecycle tracking ---
        self._plugin_states: dict[str, PluginState] = {}
        self._plugin_errors: dict[str, Exception] = {}
        self._plugin_entry_points: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self) -> None:
        if self._discovered:
            return
        for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
            if self._pm.get_plugin(ep.name) is not None:
                continue
            self._plugin_states[ep.name] = PluginState.DISCOVERED
            self._plugin_entry_points[ep.name] = ep.value

            try:
                cls = ep.load()
                plugin = cls() if isinstance(cls, type) else cls
                self.register(plugin)
            except Exception as exc:
                logger.warning(
                    "Plugin '%s' failed during discovery: %s", ep.name, exc
                )
                self._plugin_states[ep.name] = PluginState.FAILED
                self._plugin_errors[ep.name] = exc
        self._discovered = True

    # ------------------------------------------------------------------
    # Registration (with lifecycle tracking)
    # ------------------------------------------------------------------

    def register(self, plugin: Any) -> None:
        name = getattr(plugin, "name", plugin.__class__.__name__.lower())

        # Respect explicit disable
        if self._plugin_states.get(name) == PluginState.DISABLED:
            logger.info("Plugin '%s' is disabled — skipping registration.", name)
            return

        self._plugin_states.setdefault(name, PluginState.LOADED)

        # --- Compatibility validation ---
        is_compatible, _errors = validate_plugin_compatibility(plugin)
        if not is_compatible:
            logger.warning("Plugin '%s' is incompatible and will be skipped.", name)
            self._plugin_states[name] = PluginState.FAILED
            return

        self._pm.register(plugin, name)

        # --- Imperative registration (backward compatible) ---
        if hasattr(plugin, "register") and callable(plugin.register):
            try:
                plugin.register(self.registry)
            except Exception as exc:
                logger.exception(
                    "Plugin '%s' raised an error during register().", name
                )
                self._plugin_states[name] = PluginState.FAILED
                self._plugin_errors[name] = exc
                return

        # --- Declarative hook collection ---
        self._collect_declarative_hooks(plugin)

        self.registry.register_plugin(name, plugin)
        self._plugin_states[name] = PluginState.ACTIVE

    # ------------------------------------------------------------------
    # Declarative hook collection
    #
    # Each hook type targets a specialised sub-registry:
    #
    #   register_readers    → self.registry.io
    #   register_writers    → self.registry.io
    #   register_processors → self.registry.processing
    # ------------------------------------------------------------------

    def _collect_declarative_hooks(self, plugin: Any) -> dict[str, list[str]]:
        contributions: dict[str, list[str]] = {}

        self._collect_readers(plugin, contributions)
        self._collect_writers(plugin, contributions)
        self._collect_processors(plugin, contributions)

        return contributions

    def _collect_readers(
        self, plugin: Any, contributions: dict[str, list[str]]
    ) -> None:
        if not (
            hasattr(plugin, "register_readers") and callable(plugin.register_readers)
        ):
            return
        try:
            readers = plugin.register_readers()
            if not isinstance(readers, list):
                return
            for reader in readers:
                if not isinstance(reader, dict):
                    continue
                name = reader.get("name")
                func = reader.get("func")
                if name and func:
                    self.registry.io.register_reader(
                        name,
                        func,
                        description=reader.get("description", ""),
                        extensions=reader.get("extensions"),
                    )
                    contributions.setdefault("readers", []).append(name)
        except Exception:
            logger.exception(
                "Failed to collect readers from plugin '%s'",
                getattr(plugin, "name", "unknown"),
            )

    def _collect_writers(
        self, plugin: Any, contributions: dict[str, list[str]]
    ) -> None:
        if not (
            hasattr(plugin, "register_writers") and callable(plugin.register_writers)
        ):
            return
        try:
            writers = plugin.register_writers()
            if not isinstance(writers, list):
                return
            for writer in writers:
                if not isinstance(writer, dict):
                    continue
                name = writer.get("name")
                func = writer.get("func")
                if name and func:
                    self.registry.io.register_writer(
                        name,
                        func,
                        description=writer.get("description", ""),
                    )
                    contributions.setdefault("writers", []).append(name)
        except Exception:
            logger.exception(
                "Failed to collect writers from plugin '%s'",
                getattr(plugin, "name", "unknown"),
            )

    def _collect_processors(
        self, plugin: Any, contributions: dict[str, list[str]]
    ) -> None:
        if not (
            hasattr(plugin, "register_processors")
            and callable(plugin.register_processors)
        ):
            return
        try:
            procs = plugin.register_processors()
            if not isinstance(procs, list):
                return
            for proc in procs:
                if not isinstance(proc, dict):
                    continue
                name = proc.get("name")
                func = proc.get("func")
                if name and func:
                    self.registry.processing.register_processor(
                        name,
                        func,
                        description=proc.get("description", ""),
                    )
                    contributions.setdefault("processors", []).append(name)
        except Exception:
            logger.exception(
                "Failed to collect processors from plugin '%s'",
                getattr(plugin, "name", "unknown"),
            )

    # ------------------------------------------------------------------
    # Plugin loading
    # ------------------------------------------------------------------

    def load_plugin(self, name: str) -> Any | None:
        self.discover()
        plugin = self._pm.get_plugin(name)
        if plugin is None:
            for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
                if ep.name == name:
                    self._plugin_states[ep.name] = PluginState.DISCOVERED
                    self._plugin_entry_points[ep.name] = ep.value
                    try:
                        cls = ep.load()
                        plugin = cls() if isinstance(cls, type) else cls
                        self.register(plugin)
                    except Exception as exc:
                        logger.warning(
                            "Plugin '%s' failed to load: %s", name, exc
                        )
                        self._plugin_states[name] = PluginState.FAILED
                        self._plugin_errors[name] = exc
                    break
        return plugin

    # ------------------------------------------------------------------
    # Activation / deactivation
    # ------------------------------------------------------------------

    def activate_plugin(self, name: str) -> bool:
        """
        Activate a previously deactivated plugin.

        Only meaningful for plugins in ``DISABLED`` state (reactivates
        without re-registering).  Returns ``True`` on success.
        """
        if self._plugin_states.get(name) == PluginState.DISABLED:
            self._plugin_states[name] = PluginState.ACTIVE
            logger.info("Plugin '%s' activated.", name)
            return True
        logger.debug(
            "Cannot activate plugin '%s' (state: %s).",
            name,
            self._plugin_states.get(name),
        )
        return False

    def deactivate_plugin(self, name: str) -> bool:
        """
        Deactivate a plugin without unloading it.

        Marks the plugin as ``DISABLED``.  Its contributions remain in
        the registry but the plugin is skipped on future discovery
        cycles and its state is visible via introspection.

        Returns ``True`` on success.
        """
        if self._plugin_states.get(name) == PluginState.ACTIVE:
            self._plugin_states[name] = PluginState.DISABLED
            logger.info("Plugin '%s' deactivated.", name)
            return True
        logger.debug(
            "Cannot deactivate plugin '%s' (state: %s).",
            name,
            self._plugin_states.get(name),
        )
        return False

    # ------------------------------------------------------------------
    # Introspection API
    # ------------------------------------------------------------------

    def get_plugin_state(self, name: str) -> PluginState | None:
        """Return the current :class:`PluginState` of *name*, or ``None``."""
        self.discover()
        return self._plugin_states.get(name)

    def get_plugin_descriptor(self, name: str) -> PluginDescriptor | None:
        """Return a :class:`PluginDescriptor` snapshot for *name*, or ``None``."""
        self.discover()
        state = self._plugin_states.get(name)
        if state is None:
            return None
        plugin = self._pm.get_plugin(name)
        version = getattr(plugin, "version", "") if plugin else ""
        return PluginDescriptor(
            name=name,
            version=version,
            state=state,
            error=str(self._plugin_errors.get(name, "")),
            entry_point=self._plugin_entry_points.get(name),
        )

    def get_failed_plugins(self) -> dict[str, str]:
        """Return ``{name: error_message}`` for every plugin in FAILED state."""
        self.discover()
        return {
            name: str(self._plugin_errors.get(name, "unknown error"))
            for name, state in self._plugin_states.items()
            if state == PluginState.FAILED
        }

    def get_active_plugins(self) -> list[str]:
        """Return names of all plugins currently in ACTIVE state."""
        self.discover()
        return [
            name
            for name, state in self._plugin_states.items()
            if state == PluginState.ACTIVE
        ]

    # ------------------------------------------------------------------
    # Plugin query API (legacy, preserved)
    # ------------------------------------------------------------------

    def get_plugin(self, name: str) -> Any | None:
        self.discover()
        return self._pm.get_plugin(name)

    @property
    def available_plugins(self) -> dict[str, Any]:
        self.discover()
        return {name: self._pm.get_plugin(name) for name in self._pm.get_plugins()}

    def list_plugins(self) -> list[Any]:
        self.discover()
        return list(self._pm.get_plugins())

    def has_plugin(self, name: str) -> bool:
        self.discover()
        return self._pm.get_plugin(name) is not None

    def hook(self) -> Any:
        self.discover()
        return self._pm.hook


plugin_manager = PluginManager()
