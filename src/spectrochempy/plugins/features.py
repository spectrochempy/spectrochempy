# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Known optional plugin features used for user-facing installation hints.

This registry is intentionally static and limited to official optional
SpectroChemPy plugins. It is only used to produce clear errors when an optional
plugin is not installed. Installed plugins must declare their own contributions
through the plugin system; third-party plugins must not be listed here.
"""

from __future__ import annotations

from spectrochempy.plugins.deps import MissingPluginError

KNOWN_PLUGIN_READERS = {
    "topspin": ("nmr", "spectrochempy-nmr", "spectrochempy[nmr]"),
}

KNOWN_PLUGIN_NAMESPACES = {
    "nmr": ("spectrochempy-nmr", "spectrochempy[nmr]"),
    "iris": ("spectrochempy-iris", "spectrochempy[iris]"),
    "cantera": ("spectrochempy-cantera", "spectrochempy[cantera]"),
}

OFFICIAL_PLUGINS = {
    "iris": {
        "title": "IRIS plugin",
        "package": "spectrochempy-iris",
        "extra": "spectrochempy[iris]",
        "namespace": "iris",
    },
    "nmr": {
        "title": "NMR plugin",
        "package": "spectrochempy-nmr",
        "extra": "spectrochempy[nmr]",
        "namespace": "nmr",
    },
    "cantera": {
        "title": "Cantera plugin",
        "package": "spectrochempy-cantera",
        "extra": "spectrochempy[cantera]",
        "namespace": "cantera",
    },
}


def plugin_reader_install_hint(reader_name: str) -> str | None:
    """Return an install hint if *reader_name* belongs to an optional plugin."""
    info = KNOWN_PLUGIN_READERS.get(reader_name)
    if info is None:
        return None
    _ns, plugin_name, extra = info
    return (
        f"The 'read_{reader_name}' feature requires the optional plugin "
        f"'{plugin_name}'. Install it with: pip install {extra}"
    )


def plugin_reader_missing_stub(reader_name: str):
    """Return a callable stub for a missing official optional plugin reader."""
    info = KNOWN_PLUGIN_READERS.get(reader_name)
    if info is None:
        return None
    _ns, plugin_name, extra = info
    feature = f"read_{reader_name}"

    def _missing_plugin_reader(*args, **kwargs):
        raise MissingPluginError(
            feature,
            plugin_name=plugin_name,
            install_hint=f"pip install {extra}",
        )

    _missing_plugin_reader.__name__ = feature
    _missing_plugin_reader.__qualname__ = feature
    _missing_plugin_reader.__doc__ = plugin_reader_install_hint(reader_name)
    return _missing_plugin_reader


def plugin_namespace_install_hint(namespace: str) -> str | None:
    """Return an install hint if *namespace* belongs to an optional plugin."""
    info = KNOWN_PLUGIN_NAMESPACES.get(namespace)
    if info is None:
        return None
    plugin_name, extra = info
    return (
        f"The '{namespace}' namespace requires the optional plugin "
        f"'{plugin_name}'. Install it with: pip install {extra}"
    )
