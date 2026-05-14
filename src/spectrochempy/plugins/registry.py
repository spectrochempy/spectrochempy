# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class PluginRegistry:
    _instance: PluginRegistry | None = None

    def __new__(cls) -> PluginRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._readers: dict[str, dict[str, Any]] = {}
            cls._instance._writers: dict[str, dict[str, Any]] = {}
            cls._instance._processors: dict[str, dict[str, Any]] = {}
            cls._instance._unit_contexts: dict[str, Callable] = {}
            cls._instance._dtype_handlers: dict[str, Any] = {}
            cls._instance._plugins: dict[str, Any] = {}
            cls._instance._filetypes: dict[str, dict[str, Any]] = {}
        return cls._instance

    @classmethod
    def _reset(cls) -> None:
        cls._instance = None

    def register_plugin(self, name: str, plugin: Any) -> None:
        self._plugins[name] = plugin

    def get_plugin(self, name: str) -> Any | None:
        return self._plugins.get(name)

    @property
    def available_plugins(self) -> dict[str, Any]:
        return dict(self._plugins)

    def register_reader(
        self,
        name: str,
        func: Callable,
        *,
        description: str = "",
        extensions: list[str] | None = None,
    ) -> None:
        self._readers[name] = {
            "func": func,
            "description": description,
            "extensions": extensions or [],
        }

    def get_reader(self, name: str) -> dict[str, Any] | None:
        return self._readers.get(name)

    @property
    def available_readers(self) -> dict[str, dict[str, Any]]:
        return dict(self._readers)

    def register_writer(
        self, name: str, func: Callable, *, description: str = ""
    ) -> None:
        self._writers[name] = {"func": func, "description": description}

    def get_writer(self, name: str) -> dict[str, Any] | None:
        return self._writers.get(name)

    @property
    def available_writers(self) -> dict[str, dict[str, Any]]:
        return dict(self._writers)

    def register_processor(
        self, name: str, func: Callable, *, description: str = ""
    ) -> None:
        self._processors[name] = {"func": func, "description": description}

    def get_processor(self, name: str) -> dict[str, Any] | None:
        return self._processors.get(name)

    def register_unit_context(self, name: str, setup_func: Callable) -> None:
        self._unit_contexts[name] = setup_func

    def get_unit_context(self, name: str) -> Callable | None:
        return self._unit_contexts.get(name)

    def register_dtype_handler(self, dtype: str, handler: Any) -> None:
        self._dtype_handlers[dtype] = handler

    def has_dtype_handler(self, dtype: str) -> bool:
        return dtype in self._dtype_handlers

    def get_dtype_handler(self, dtype: str) -> Any | None:
        return self._dtype_handlers.get(dtype)

    def register_filetype(self, ext: str, info: dict[str, Any]) -> None:
        self._filetypes[ext] = info

    def get_filetype(self, ext: str) -> dict[str, Any] | None:
        return self._filetypes.get(ext)

    @property
    def available_filetypes(self) -> dict[str, dict[str, Any]]:
        return dict(self._filetypes)


registry = PluginRegistry()
