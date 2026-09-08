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

# Namespace requests rejected because they collide with a public root symbol.
# Maps ``name -> "plugin=<plugin> read=<read_path> write=<write_path>"`` and is
# available for introspection/diagnostics.  The plugin itself remains fully
# functional; only the conflicting short namespace is refused.
_REJECTED_IO_NAMESPACES: dict[str, str] = {}


def _namespace_owner(name: str) -> str | None:
    """Return the plugin an I/O namespace was contributed by, if known."""
    read_path = None
    for _name, (_read, _write) in _PLUGIN_IO_NAMESPACES.items():
        if _name == name:
            read_path = _read
            break
    if not read_path or "." not in read_path:
        return None
    return read_path.split(".", 1)[0]


def register_io_namespace(
    name: str,
    read_path: str | None = None,
    write_path: str | None = None,
    plugin: str | None = None,
) -> bool:
    """
    Register a plugin-contributed I/O namespace.

    Names that collide with a public ``scp`` root symbol are rejected with a
    controlled warning instead of silently shadowing the core API.  The
    reader behind ``read_<name>`` remains available through its explicit
    top-level function.

    Parameters
    ----------
    name : str
        Namespace name exposed as ``scp.<name>``.
    read_path : str or None
        Dotted attribute path resolving to the read function, relative to
        the ``spectrochempy`` package (for example ``nmr.read_topspin``).
    write_path : str or None
        Dotted attribute path resolving to the write function, if any.
    plugin : str or None
        Name of the contributing plugin, used in diagnostics.

    Returns
    -------
    bool
        ``True`` when the namespace was registered, ``False`` when it was
        rejected because of a reserved-name collision (or because a core
        namespace already owns the name).
    """
    import warnings

    from spectrochempy.lazyimport.root_symbols import is_reserved_root_symbol

    if name in _CORE_IO_NAMESPACES:
        _REJECTED_IO_NAMESPACES[
            name
        ] = f"plugin={plugin or 'unknown'} core namespace '{name}'"
        warnings.warn(
            f"Refusing plugin I/O namespace '{name}' (plugin={plugin or 'unknown'}): "
            f"'{name}' is already a core I/O namespace.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    if is_reserved_root_symbol(name):
        _REJECTED_IO_NAMESPACES[
            name
        ] = f"plugin={plugin or 'unknown'} read={read_path} write={write_path}"
        if read_path and "." in read_path:
            reader_surface = f"scp.{read_path.rsplit('.', 1)[-1]}"
        else:
            reader_surface = f"scp.read_{name}"
        warnings.warn(
            f"Refusing plugin I/O namespace '{name}' (plugin={plugin or 'unknown'}): "
            f"'{name}' is a reserved public SpectroChemPy symbol. Use "
            f"'{reader_surface}' instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    if name in _PLUGIN_IO_NAMESPACES:
        warnings.warn(
            f"I/O namespace '{name}' (plugin={plugin or 'unknown'}) is already "
            f"registered by plugin '{_namespace_owner(name) or 'unknown'}'; "
            "keeping the first registration.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    _PLUGIN_IO_NAMESPACES[name] = (read_path, write_path)
    return True


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
