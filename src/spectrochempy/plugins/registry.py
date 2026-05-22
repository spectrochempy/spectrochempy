# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from spectrochempy.plugins.capabilities import PluginCapability
from spectrochempy.plugins.registries import ExtensionRegistry
from spectrochempy.plugins.registries import IORegistry
from spectrochempy.plugins.registries import MetadataRegistry
from spectrochempy.plugins.registries import ProcessingRegistry
from spectrochempy.plugins.registries import VisualizationRegistry

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Composition root for all specialised registries.

    This is a stable facade that delegates to domain-specific
    sub-registries.  All methods of the previous monolithic
    ``PluginRegistry`` are preserved for backward compatibility.

    New code can access sub-registries directly::

        registry.io.register_reader(...)
        registry.processing.register_processor(...)
        registry.metadata.register_plugin(...)

    Legacy code using top-level methods continues to work::

        registry.register_reader(...)     # forwards to registry.io
        registry.register_processor(...)  # forwards to registry.processing

    .. versionchanged:: 1.0
       Responsibilities decomposed into specialised registries.
    """

    def __init__(self) -> None:
        self.io = IORegistry()
        self.processing = ProcessingRegistry()
        self.visualization = VisualizationRegistry()
        self.metadata = MetadataRegistry()
        self.extensions = ExtensionRegistry()

    def clear(self) -> None:
        """Remove all entries from every sub-registry."""
        self.io.clear()
        self.processing.clear()
        self.visualization.clear()
        self.metadata.clear()
        self.extensions.clear()

    def merge_from(self, other: PluginRegistry) -> None:
        """Merge contributions from another registry into this registry."""
        self.io._readers.update(other.io._readers)
        self.io._writers.update(other.io._writers)
        self.io._filetypes.update(other.io._filetypes)
        self.processing._processors.update(other.processing._processors)
        self.processing._unit_contexts.update(other.processing._unit_contexts)
        self.processing._dtype_handlers.update(other.processing._dtype_handlers)
        self.visualization._visualizers.update(other.visualization._visualizers)
        for category, entries in other.extensions._extensions.items():
            self.extensions._extensions.setdefault(category, {}).update(entries)

    # ------------------------------------------------------------------
    # Capability-based query
    # ------------------------------------------------------------------

    def get_by_capability(self, capability: PluginCapability) -> list[dict[str, Any]]:
        """
        Return all plugin contributions matching a given capability.

        This provides a uniform way to discover what the system can do
        regardless of which sub-registry holds the contributions.

        Example::

            registry.get_by_capability(PluginCapability.ANALYSIS)
            # → [{"plugin": "iris", "name": "pca", ...}, ...]
        """
        results: list[dict[str, Any]] = []
        if capability == PluginCapability.READER:
            for name, info in self.io.available_readers.items():
                results.append({"capability": "reader", "name": name, **info})
        elif capability == PluginCapability.WRITER:
            for name, info in self.io.available_writers.items():
                results.append({"capability": "writer", "name": name, **info})
        elif capability == PluginCapability.PROCESSOR:
            for name, info in self.processing.available_processors.items():
                results.append({"capability": "processor", "name": name, **info})
        elif capability == PluginCapability.VISUALIZER:
            for name, info in self.visualization.available_visualizers.items():
                results.append({"capability": "visualizer", "name": name, **info})
        elif capability in (
            PluginCapability.ANALYSIS,
            PluginCapability.SIMULATION,
            PluginCapability.ACCESSOR,
        ):
            for name, info in self.extensions.list_category(capability.value).items():
                results.append({"capability": capability.value, "name": name, **info})
        return results

    # ------------------------------------------------------------------
    # Backward-compatible forwarding — IORegistry
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
        self.io.register_reader(
            name,
            func,
            description=description,
            extensions=extensions,
            plugin=plugin,
            namespace=namespace,
        )

    def get_reader(self, name: str) -> dict[str, Any] | None:
        return self.io.get_reader(name)

    @property
    def available_readers(self) -> dict[str, dict[str, Any]]:
        return self.io.available_readers

    def register_writer(
        self, name: str, func: Callable, *, description: str = ""
    ) -> None:
        self.io.register_writer(name, func, description=description)

    def get_writer(self, name: str) -> dict[str, Any] | None:
        return self.io.get_writer(name)

    @property
    def available_writers(self) -> dict[str, dict[str, Any]]:
        return self.io.available_writers

    def register_filetype(self, ext: str, info: dict[str, Any]) -> None:
        self.io.register_filetype(ext, info)

    def get_filetype(self, ext: str) -> dict[str, Any] | None:
        return self.io.get_filetype(ext)

    @property
    def available_filetypes(self) -> dict[str, dict[str, Any]]:
        return self.io.available_filetypes

    # ------------------------------------------------------------------
    # Backward-compatible forwarding — ProcessingRegistry
    # ------------------------------------------------------------------

    def register_processor(
        self, name: str, func: Callable, *, description: str = ""
    ) -> None:
        self.processing.register_processor(name, func, description=description)

    def get_processor(self, name: str) -> dict[str, Any] | None:
        return self.processing.get_processor(name)

    @property
    def available_processors(self) -> dict[str, dict[str, Any]]:
        return self.processing.available_processors

    # ------------------------------------------------------------------
    # Backward-compatible forwarding — ExtensionRegistry
    # ------------------------------------------------------------------

    def register_analysis(
        self, name: str, func: Callable, *, description: str = ""
    ) -> None:
        self.extensions.register("analysis", name, func, description=description)

    def get_analysis(self, name: str) -> dict[str, Any] | None:
        return self.extensions.get("analysis", name)

    @property
    def available_analyses(self) -> dict[str, dict[str, Any]]:
        return self.extensions.list_category("analysis")

    def register_simulation(
        self, name: str, func: Callable, *, description: str = ""
    ) -> None:
        self.extensions.register("simulation", name, func, description=description)

    def get_simulation(self, name: str) -> dict[str, Any] | None:
        return self.extensions.get("simulation", name)

    @property
    def available_simulations(self) -> dict[str, dict[str, Any]]:
        return self.extensions.list_category("simulation")

    def register_accessor(
        self,
        name: str,
        func: Callable,
        *,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.extensions.register(
            "accessor", name, func, description=description, metadata=metadata
        )

    def get_accessor(self, name: str) -> dict[str, Any] | None:
        return self.extensions.get("accessor", name)

    @property
    def available_accessors(self) -> dict[str, dict[str, Any]]:
        return self.extensions.list_category("accessor")

    def register_unit_context(
        self,
        name: str,
        setup_func: Callable,
        *,
        predicate: Callable[[Any], bool] | None = None,
        argument_extractor: Callable[[Any], Any] | None = None,
        description: str = "",
    ) -> None:
        self.processing.register_unit_context(
            name,
            setup_func,
            predicate=predicate,
            argument_extractor=argument_extractor,
            description=description,
        )

    def get_unit_context(self, name: str) -> Callable | None:
        return self.processing.get_unit_context(name)

    def get_unit_context_info(self, name: str) -> dict[str, Any] | None:
        return self.processing.get_unit_context_info(name)

    def get_applicable_unit_context(self, obj: Any) -> dict[str, Any] | None:
        return self.processing.get_applicable_unit_context(obj)

    def register_dtype_handler(self, dtype: str, handler: Any) -> None:
        self.processing.register_dtype_handler(dtype, handler)

    def has_dtype_handler(self, dtype: str) -> bool:
        return self.processing.has_dtype_handler(dtype)

    def get_dtype_handler(self, dtype: str) -> Any | None:
        return self.processing.get_dtype_handler(dtype)

    # ------------------------------------------------------------------
    # Backward-compatible forwarding — MetadataRegistry
    # ------------------------------------------------------------------

    def register_plugin(self, name: str, plugin: Any) -> None:
        self.metadata.register_plugin(name, plugin)

    def get_plugin(self, name: str) -> Any | None:
        return self.metadata.get_plugin(name)

    @property
    def available_plugins(self) -> dict[str, Any]:
        return self.metadata.available_plugins


registry = PluginRegistry()
