from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any


def _matches_namespace(info: dict[str, Any], namespace: str) -> bool:
    metadata = info.get("metadata", {})
    return (
        info.get("plugin") == namespace
        or info.get("namespace") == namespace
        or metadata.get("plugin") == namespace
        or metadata.get("namespace") == namespace
    )


def _method_name(key: str, namespace: str) -> str:
    prefix = f"{namespace}."
    if key.startswith(prefix):
        return key[len(prefix) :]
    return key


class PluginNamespace:
    """Package-level namespace for plugin APIs such as ``scp.nmr``."""

    def __init__(self, namespace: str, manager: Any, registry: Any) -> None:
        self._namespace = namespace
        self._manager = manager
        self._registry = registry

    def __dir__(self) -> list[str]:
        names = set(super().__dir__())
        names.update(self._reader_names())
        names.update(self._extension_names())
        plugin = self._manager.get_plugin(self._namespace)
        if plugin is not None:
            names.update(name for name in dir(plugin) if not name.startswith("_"))
            module = self._plugin_module(plugin)
            if module is not None:
                names.update(name for name in dir(module) if not name.startswith("_"))
        return sorted(names)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("read_"):
            reader = name[len("read_") :]
            info = self._registry.get_reader(reader)
            if info and _matches_namespace(info, self._namespace):
                return info["func"]

        for category in ("analysis", "simulation"):
            info = self._registry.extensions.get(category, name)
            if info and _matches_namespace(info, self._namespace):
                return info["obj"]

        plugin = self._manager.get_plugin(self._namespace)
        if plugin is not None and hasattr(plugin, name):
            return getattr(plugin, name)

        if plugin is not None:
            module = self._plugin_module(plugin)
            if module is not None:
                try:
                    return getattr(module, name)
                except AttributeError:
                    pass

        raise AttributeError(
            f"plugin namespace '{self._namespace}' has no attribute '{name}'"
        )

    @staticmethod
    def _plugin_module(plugin: Any) -> ModuleType | None:
        module_name = getattr(plugin.__class__, "__module__", None)
        if not module_name:
            return None
        return importlib.import_module(module_name)

    def _reader_names(self) -> set[str]:
        return {
            f"read_{name}"
            for name, info in self._registry.available_readers.items()
            if _matches_namespace(info, self._namespace)
        }

    def _extension_names(self) -> set[str]:
        names: set[str] = set()
        for category in ("analysis", "simulation"):
            for name, info in self._registry.extensions.list_category(category).items():
                if _matches_namespace(info, self._namespace):
                    names.add(_method_name(name, self._namespace))
        return names


class DatasetPluginAccessor:
    """
    Dataset-bound namespace for plugin operations such as ``nd.iris``.

    These accessors are reserved for callables that operate on the parent
    dataset. Plugin I/O and object creation should remain package-level APIs
    exposed through :class:`PluginNamespace`, for example ``scp.nmr.read_topspin``.
    """

    def __init__(self, dataset: Any, namespace: str, registry: Any) -> None:
        self._dataset = dataset
        self._namespace = namespace
        self._registry = registry

    def __dir__(self) -> list[str]:
        names = set(super().__dir__())
        names.update(self._accessor_names())
        return sorted(names)

    def __getattr__(self, name: str) -> Callable:
        info = self._registry.get_accessor(f"{self._namespace}.{name}")
        if info and _matches_namespace(info, self._namespace):
            func = info["obj"]
            return lambda *args, **kwargs: func(self._dataset, *args, **kwargs)

        raise AttributeError(
            f"dataset plugin accessor '{self._namespace}' has no attribute '{name}'"
        )

    def _accessor_names(self) -> set[str]:
        return {
            _method_name(name, self._namespace)
            for name, info in self._registry.available_accessors.items()
            if name.startswith(f"{self._namespace}.")
            and _matches_namespace(info, self._namespace)
        }


def has_namespace(registry: Any, namespace: str) -> bool:
    """Return ``True`` when any registered contribution belongs to *namespace*."""
    if registry.metadata.get_plugin(namespace) is not None:
        return True

    if any(
        _matches_namespace(info, namespace)
        for info in registry.available_readers.values()
    ):
        return True

    for category in ("analysis", "simulation", "accessor"):
        if any(
            _matches_namespace(info, namespace)
            for info in registry.extensions.list_category(category).values()
        ):
            return True

    return False


def has_dataset_namespace(registry: Any, namespace: str) -> bool:
    """Return ``True`` when dataset accessors exist for *namespace*."""
    return any(
        _matches_namespace(info, namespace)
        for info in registry.available_accessors.values()
    )


class PluginNamespaceModule(ModuleType):
    """
    A module-like wrapper that makes ``from spectrochempy.<ns> import X`` possible.

    Instances are inserted into ``sys.modules`` under keys such as
    ``spectrochempy.iris`` and delegate attribute access to the underlying
    :class:`PluginNamespace`, preserving lazy loading of the actual plugin.
    """

    def __init__(self, namespace: str, manager: Any, registry: Any) -> None:
        self._namespace_obj = PluginNamespace(namespace, manager, registry)
        name = f"spectrochempy.{namespace}"
        super().__init__(name)
        self.__package__ = "spectrochempy"
        self.__path__: list[str] = []
        self.__file__: str | None = None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._namespace_obj, name)

    def __dir__(self) -> list[str]:
        return dir(self._namespace_obj)


def register_namespace_modules() -> None:
    """Insert ``PluginNamespaceModule`` instances into ``sys.modules``."""
    from spectrochempy.plugins.features import KNOWN_PLUGIN_NAMESPACES
    from spectrochempy.plugins.manager import plugin_manager
    from spectrochempy.plugins.registry import registry

    for ns in KNOWN_PLUGIN_NAMESPACES:
        key = f"spectrochempy.{ns}"
        if key not in sys.modules:
            sys.modules[key] = PluginNamespaceModule(ns, plugin_manager, registry)
