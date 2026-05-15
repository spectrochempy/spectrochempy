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
from spectrochempy.plugins.registry import registry as _default_registry

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "spectrochempy.plugins"


class PluginManager:
    """
    Orchestrates plugin discovery, validation, and registration.

    Parameters
    ----------
    registry : PluginRegistry or None
        Registry instance to use.  When ``None`` (default), the module-
        level singleton ``spectrochempy.plugins.registry.registry`` is
        used for backward compatibility.

    The manager is the main orchestration layer: it calls
    ``plugin.register(registry)`` for backward-compatible imperative
    registration **and** collects declarative hook contributions
    (``register_readers``, ``register_writers``,
    ``register_processors``) so that both patterns coexist.
    """

    def __init__(self, registry: Any = None) -> None:
        self._pm = pluggy.PluginManager("spectrochempy")
        self._pm.add_hookspecs(SpectroChemPyHookSpec)
        self._discovered = False

        if registry is None:
            registry = _default_registry
        self.registry = registry

    def discover(self) -> None:
        if self._discovered:
            return
        for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
            if self._pm.get_plugin(ep.name) is not None:
                continue
            cls = ep.load()
            plugin = cls() if isinstance(cls, type) else cls
            self.register(plugin)
        self._discovered = True

    def register(self, plugin: Any) -> None:
        name = getattr(plugin, "name", plugin.__class__.__name__.lower())

        is_compatible, _errors = validate_plugin_compatibility(plugin)
        if not is_compatible:
            logger.warning("Plugin '%s' is incompatible and will be skipped.", name)
            return

        self._pm.register(plugin, name)

        # --- Imperative registration (backward compatible) ---
        if hasattr(plugin, "register") and callable(plugin.register):
            plugin.register(self.registry)

        # --- Declarative hook collection (new style) ---
        self._collect_declarative_hooks(plugin)

        self.registry.register_plugin(name, plugin)

    # ------------------------------------------------------------------
    # Declarative hook collection
    #
    # Each hook type targets a specialised sub-registry:
    #
    #   register_readers    → self.registry.io
    #   register_writers    → self.registry.io
    #   register_processors → self.registry.processing
    #
    # Backward-compatible forwarding on PluginRegistry means that
    # ``self.registry.register_reader(...)`` would also work, but
    # routing directly to sub-registries is more explicit.
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
    # Plugin query API
    # ------------------------------------------------------------------

    def load_plugin(self, name: str) -> Any | None:
        self.discover()
        plugin = self._pm.get_plugin(name)
        if plugin is None:
            for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
                if ep.name == name:
                    cls = ep.load()
                    plugin = cls() if isinstance(cls, type) else cls
                    self.register(plugin)
                    break
        return plugin

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
