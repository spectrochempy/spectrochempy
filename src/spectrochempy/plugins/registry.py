# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

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

    def clear(self) -> None:
        """Remove all entries from every sub-registry."""
        self.io.clear()
        self.processing.clear()
        self.visualization.clear()
        self.metadata.clear()

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
    ) -> None:
        self.io.register_reader(
            name, func, description=description, extensions=extensions
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

    def register_unit_context(self, name: str, setup_func: Callable) -> None:
        self.processing.register_unit_context(name, setup_func)

    def get_unit_context(self, name: str) -> Callable | None:
        return self.processing.get_unit_context(name)

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
