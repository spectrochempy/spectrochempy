from __future__ import annotations

from collections.abc import Callable
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

        raise AttributeError(
            f"plugin namespace '{self._namespace}' has no attribute '{name}'"
        )

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
    """Dataset-bound namespace for plugin operations such as ``nd.iris``."""

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
