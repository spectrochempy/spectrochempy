"""
Core I/O namespace wrappers for the namespace-based API.

This module provides lightweight namespace objects that expose
``scp.<domain>.read(...)`` and ``scp.<domain>.write(...)`` for core
I/O operations.
"""

from __future__ import annotations

from typing import Any

# Single source of truth for core I/O namespace configuration.
#
# To add a new namespace, append an entry here:
#   "<domain>": ("read_<domain>", "write_<domain>" or None)
#
# The namespace object will automatically delegate ``scp.<domain>.read()``
# and ``scp.<domain>.write()`` to the corresponding top-level functions.
# No other code changes are required.
_CORE_IO_NAMESPACES: dict[str, tuple[str | None, str | None]] = {
    "jcamp": ("read_jcamp", "write_jcamp"),
    "csv": ("read_csv", "write_csv"),
    "matlab": ("read_matlab", "write_matlab"),
    "omnic": ("read_omnic", None),
    "opus": ("read_opus", None),
    "quadera": ("read_quadera", None),
    "soc": ("read_soc", None),
    "spc": ("read_spc", None),
    "wire": ("read_wire", None),
    "labspec": ("read_labspec", None),
}

# Plugin-contributed I/O namespaces.
#
# Plugins register namespaces such as ``topspin`` or ``agilent`` that map
# ``scp.<domain>.read()`` to a dotted attribute path on the ``spectrochempy``
# package (for example ``nmr.read_topspin``).  This keeps the core package
# decoupled from plugin-specific reader names while exposing the same
# namespace-style API as core I/O domains.
_PLUGIN_IO_NAMESPACES: dict[str, tuple[str | None, str | None]] = {}


def register_io_namespace(
    name: str,
    read_path: str | None = None,
    write_path: str | None = None,
) -> None:
    """
    Register a plugin-contributed I/O namespace.

    Parameters
    ----------
    name : str
        Namespace name exposed as ``scp.<name>``.
    read_path : str or None
        Dotted attribute path resolving to the read function, relative to
        the ``spectrochempy`` package (for example ``nmr.read_topspin``).
    write_path : str or None
        Dotted attribute path resolving to the write function, if any.
    """
    _PLUGIN_IO_NAMESPACES[name] = (read_path, write_path)


def unregister_io_namespace(name: str) -> None:
    """Remove a plugin-contributed I/O namespace (mainly for tests)."""
    _PLUGIN_IO_NAMESPACES.pop(name, None)


def _resolve_func(func_name: str) -> Any:
    """Lazy-resolve a top-level function by name."""
    import spectrochempy as scp

    return getattr(scp, func_name)


def _resolve_attr(attr_path: str) -> Any:
    """Lazy-resolve a dotted attribute path relative to ``spectrochempy``."""
    import spectrochempy as scp

    obj: Any = scp
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


class _IONamespace:
    """
    Lightweight namespace for core I/O operations.

    Instances are returned for names such as ``scp.jcamp`` and delegate
    ``read()`` / ``write()`` to the existing public ``read_*`` / ``write_*``
    functions (core namespaces) or to plugin namespaced APIs
    (plugin-contributed namespaces).
    """

    def __init__(self, name: str) -> None:
        self._name = name
        if name in _CORE_IO_NAMESPACES:
            self._read_ref, self._write_ref = _CORE_IO_NAMESPACES[name]
            self._plugin = False
        else:
            self._read_ref, self._write_ref = _PLUGIN_IO_NAMESPACES[name]
            self._plugin = True

    def __dir__(self) -> list[str]:
        names = []
        if self._read_ref:
            names.append("read")
        if self._write_ref:
            names.append("write")
        return names

    def __getattr__(self, name: str) -> Any:
        if name == "read" and self._read_ref:
            if self._plugin:
                return _resolve_attr(self._read_ref)
            return _resolve_func(self._read_ref)
        if name == "write" and self._write_ref:
            if self._plugin:
                return _resolve_attr(self._write_ref)
            return _resolve_func(self._write_ref)
        raise AttributeError(f"namespace '{self._name}' has no attribute '{name}'")

    def __repr__(self) -> str:
        ops = []
        if self._read_ref:
            ops.append("read")
        if self._write_ref:
            ops.append("write")
        return f"<IONamespace '{self._name}' ({', '.join(ops)})>"


def _is_io_namespace(name: str) -> bool:
    return name in _CORE_IO_NAMESPACES or name in _PLUGIN_IO_NAMESPACES
