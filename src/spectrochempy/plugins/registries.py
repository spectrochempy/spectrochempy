# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Specialised registries for each plugin contribution domain.

Each registry owns its own state and exposes a focused API.
They are composed by :class:`~spectrochempy.plugins.registry.PluginRegistry`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class IORegistry:
    """
    Manages reader, writer, and filetype contributions.

    This registry is responsible for all I/O-related plugin
    contributions: file readers, file writers, and file-format
    metadata.
    """

    def __init__(self) -> None:
        self._readers: dict[str, dict[str, Any]] = {}
        self._writers: dict[str, dict[str, Any]] = {}
        self._filetypes: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    def register_reader(
        self,
        name: str,
        func: Callable,
        *,
        description: str = "",
        extensions: list[str] | None = None,
        plugin: str | None = None,
        namespace: str | None = None,
    ) -> None:
        self._readers[name] = {
            "func": func,
            "description": description,
            "extensions": extensions or [],
            "plugin": plugin,
            "namespace": namespace or plugin,
        }

    def get_reader(self, name: str) -> dict[str, Any] | None:
        return self._readers.get(name)

    @property
    def available_readers(self) -> dict[str, dict[str, Any]]:
        return dict(self._readers)

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def register_writer(
        self, name: str, func: Callable, *, description: str = ""
    ) -> None:
        self._writers[name] = {"func": func, "description": description}

    def get_writer(self, name: str) -> dict[str, Any] | None:
        return self._writers.get(name)

    @property
    def available_writers(self) -> dict[str, dict[str, Any]]:
        return dict(self._writers)

    # ------------------------------------------------------------------
    # Filetypes
    # ------------------------------------------------------------------

    def register_filetype(self, ext: str, info: dict[str, Any]) -> None:
        self._filetypes[ext] = info

    def get_filetype(self, ext: str) -> dict[str, Any] | None:
        return self._filetypes.get(ext)

    @property
    def available_filetypes(self) -> dict[str, dict[str, Any]]:
        return dict(self._filetypes)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._readers.clear()
        self._writers.clear()
        self._filetypes.clear()


class ProcessingRegistry:
    """
    Manages processor, unit-context, and dtype-handler contributions.

    This registry holds data-processing functions, unit system
    setup callables, and numpy-dtype handler mappings.
    """

    def __init__(self) -> None:
        self._processors: dict[str, dict[str, Any]] = {}
        self._unit_contexts: dict[str, dict[str, Any]] = {}
        self._dtype_handlers: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Processors
    # ------------------------------------------------------------------

    def register_processor(
        self, name: str, func: Callable, *, description: str = ""
    ) -> None:
        self._processors[name] = {"func": func, "description": description}

    def get_processor(self, name: str) -> dict[str, Any] | None:
        return self._processors.get(name)

    @property
    def available_processors(self) -> dict[str, dict[str, Any]]:
        return dict(self._processors)

    # ------------------------------------------------------------------
    # Unit contexts
    # ------------------------------------------------------------------

    def register_unit_context(
        self,
        name: str,
        setup_func: Callable,
        *,
        predicate: Callable[[Any], bool] | None = None,
        argument_extractor: Callable[[Any], Any] | None = None,
        description: str = "",
    ) -> None:
        self._unit_contexts[name] = {
            "name": name,
            "func": setup_func,
            "predicate": predicate,
            "argument_extractor": argument_extractor,
            "description": description,
        }

    def get_unit_context(self, name: str) -> Callable | None:
        info = self._unit_contexts.get(name)
        if info is None:
            return None
        return info["func"]

    def get_unit_context_info(self, name: str) -> dict[str, Any] | None:
        info = self._unit_contexts.get(name)
        if info is None:
            return None
        return dict(info)

    def get_applicable_unit_context(self, obj: Any) -> dict[str, Any] | None:
        for info in self._unit_contexts.values():
            predicate = info.get("predicate")
            if predicate is None:
                continue
            if predicate(obj):
                return dict(info)
        return None

    # ------------------------------------------------------------------
    # Dtype handlers
    # ------------------------------------------------------------------

    def register_dtype_handler(self, dtype: str, handler: Any) -> None:
        self._dtype_handlers[dtype] = handler

    def has_dtype_handler(self, dtype: str) -> bool:
        return dtype in self._dtype_handlers

    def get_dtype_handler(self, dtype: str) -> Any | None:
        return self._dtype_handlers.get(dtype)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._processors.clear()
        self._unit_contexts.clear()
        self._dtype_handlers.clear()


class VisualizationRegistry:
    """
    Manages visualizer contributions (future).

    Currently a placeholder for future plotting / visualisation
    contributions from plugins.
    """

    def __init__(self) -> None:
        self._visualizers: dict[str, dict[str, Any]] = {}

    def register_visualizer(
        self, name: str, func: Callable, *, description: str = ""
    ) -> None:
        self._visualizers[name] = {"func": func, "description": description}

    def get_visualizer(self, name: str) -> dict[str, Any] | None:
        return self._visualizers.get(name)

    @property
    def available_visualizers(self) -> dict[str, dict[str, Any]]:
        return dict(self._visualizers)

    def clear(self) -> None:
        self._visualizers.clear()


class ExtensionRegistry:
    """
    Generic registry for arbitrary named extensions.

    Allows plugins to register any named object under a category,
    providing a flexible escape hatch for contributions that don't
    fit the specialised sub-registries (e.g. fit models, simulation
    engines, domain metadata schemas, validation rules).

    Categories are free-form strings (e.g. ``"fit_model"``,
    ``"reaction_mechanism"``, ``"thermodynamic_package"``).
    """

    def __init__(self) -> None:
        self._extensions: dict[str, dict[str, Any]] = {}

    def register(
        self,
        category: str,
        name: str,
        obj: Any,
        *,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if category not in self._extensions:
            self._extensions[category] = {}
        self._extensions[category][name] = {
            "obj": obj,
            "description": description,
            "metadata": metadata or {},
        }

    def get(self, category: str, name: str) -> dict[str, Any] | None:
        cat = self._extensions.get(category)
        if cat is None:
            return None
        return cat.get(name)

    def list_category(self, category: str) -> dict[str, dict[str, Any]]:
        return dict(self._extensions.get(category, {}))

    @property
    def categories(self) -> list[str]:
        return list(self._extensions)

    def clear(self) -> None:
        self._extensions.clear()


class MetadataRegistry:
    """
    Manages plugin descriptor metadata.

    Tracks which plugins are registered and their descriptors.
    Does **not** store the plugin instances themselves (those live
    in the pluggy ``PluginManager``).
    """

    def __init__(self) -> None:
        self._plugins: dict[str, Any] = {}

    def register_plugin(self, name: str, plugin: Any) -> None:
        self._plugins[name] = plugin

    def get_plugin(self, name: str) -> Any | None:
        return self._plugins.get(name)

    @property
    def available_plugins(self) -> dict[str, Any]:
        return dict(self._plugins)

    def clear(self) -> None:
        self._plugins.clear()


class HandlerRegistry:
    """
    Manages generic handler overrides registered by plugins.

    Each handler is a callable associated with a named extension point
    (e.g. ``"coord.reversed"``).  The core dispatches to handlers when
    present, falling back to the default behaviour when a handler
    returns ``None`` or is not registered.

    Multiple plugins may register handlers for the same extension point.
    They are called in registration order (first-registered wins on a
    non-``None`` result).  This makes ``importer.infer_filetype_key`` and
    similar discovery hooks composable across plugins.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable]] = {}

    def register_handler(self, name: str, func: Callable) -> None:
        """
        Register a callable for the given extension point.

        If another plugin has already registered a handler for the same
        name, both are kept and called in order (first non-``None`` wins).
        """
        self._handlers.setdefault(name, []).append(func)

    def get_handler(self, name: str) -> Callable | None:
        """
        Return a chained handler for *name*, or ``None`` if not registered.

        The returned callable iterates over every registered handler and
        returns the first non-``None`` result.  If every handler returns
        ``None``, the chain itself returns ``None``.
        """
        handlers = self._handlers.get(name)
        if not handlers:
            return None

        def _chain(*args, **kwargs):
            for func in handlers:
                result = func(*args, **kwargs)
                if result is not None:
                    return result
            return None

        return _chain

    @property
    def available_handlers(self) -> dict[str, list[Callable]]:
        """Return a snapshot of all registered handlers."""
        return {name: list(funcs) for name, funcs in self._handlers.items()}

    def clear(self) -> None:
        self._handlers.clear()
