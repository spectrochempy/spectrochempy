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

from spectrochempy.plugins.hooks import SpectroChemPyHookSpec
from spectrochempy.plugins.registry import registry

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "spectrochempy.plugins"


class PluginManager:
    def __init__(self) -> None:
        self._pm = pluggy.PluginManager("spectrochempy")
        self._pm.add_hookspecs(SpectroChemPyHookSpec)
        self._discovered = False

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
        self._pm.register(plugin, name)
        if hasattr(plugin, "register") and callable(plugin.register):
            plugin.register(registry)
        registry.register_plugin(name, plugin)

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
