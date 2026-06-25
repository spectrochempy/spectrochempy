"""Core I/O namespace wrappers for the namespace-based API.

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


def _resolve_func(func_name: str) -> Any:
    """Lazy-resolve a top-level function by name."""
    import spectrochempy as scp

    return getattr(scp, func_name)


class _IONamespace:
    """Lightweight namespace for core I/O operations.

    Instances are returned for names such as ``scp.jcamp`` and delegate
    ``read()`` / ``write()`` to the existing public ``read_*`` / ``write_*``
    functions.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._read_name, self._write_name = _CORE_IO_NAMESPACES[name]

    def __dir__(self) -> list[str]:
        names = ["read"] if self._read_name else []
        if self._write_name:
            names.append("write")
        return names

    def __getattr__(self, name: str) -> Any:
        if name == "read" and self._read_name:
            return _resolve_func(self._read_name)
        if name == "write" and self._write_name:
            return _resolve_func(self._write_name)
        raise AttributeError(
            f"namespace '{self._name}' has no attribute '{name}'"
        )

    def __repr__(self) -> str:
        ops = []
        if self._read_name:
            ops.append("read")
        if self._write_name:
            ops.append("write")
        return f"<IONamespace '{self._name}' ({', '.join(ops)})>"


def _is_io_namespace(name: str) -> bool:
    return name in _CORE_IO_NAMESPACES
