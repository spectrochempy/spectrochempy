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
    "carroucell": ("spectrochempy-carroucell", "spectrochempy-carroucell"),
}

EXPERIMENTAL_PLUGIN_NAMESPACES = {
    "cantera": ("spectrochempy-cantera", "spectrochempy-cantera"),
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
    "hypercomplex": {
        "title": "Hypercomplex plugin",
        "package": "spectrochempy-hypercomplex",
        "extra": "spectrochempy-hypercomplex",
        "namespace": "",
    },
    "carroucell": {
        "title": "Carroucell plugin",
        "package": "spectrochempy-carroucell",
        "extra": "spectrochempy-carroucell",
        "namespace": "carroucell",
    },
}

PLUGIN_SYMBOL_HINTS = {
    "IRIS": {
        "plugin_title": "IRIS plugin",
        "plugin_package": "spectrochempy-iris",
        "namespace": "scp.iris.IRIS",
    },
    "IrisKernel": {
        "plugin_title": "IRIS plugin",
        "plugin_package": "spectrochempy-iris",
        "namespace": "scp.iris.IrisKernel",
    },
    "batch_iris": {
        "plugin_title": "IRIS plugin",
        "plugin_package": "spectrochempy-iris",
        "namespace": "scp.iris.batch_iris",
    },
    "compare_kernels": {
        "plugin_title": "IRIS plugin",
        "plugin_package": "spectrochempy-iris",
        "namespace": "scp.iris.compare_kernels",
    },
    "iris_report": {
        "plugin_title": "IRIS plugin",
        "plugin_package": "spectrochempy-iris",
        "namespace": "scp.iris.iris_report",
    },
}


def plugin_symbol_install_hint(symbol: str) -> str | None:
    """Return an AttributeError message for a missing official plugin symbol."""
    info = PLUGIN_SYMBOL_HINTS.get(symbol)
    if info is None:
        return None

    return (
        f"module 'spectrochempy' has no attribute '{symbol}'.\n\n"
        "Did you mean:\n"
        f"    {info['namespace']}\n\n"
        f"The official {info['plugin_title']} is not installed.\n\n"
        "Install it with:\n"
        f"    pip install {info['plugin_package']}\n\n"
        "or:\n"
        "    pip install spectrochempy[plugins]"
    )


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
    if info is not None:
        plugin_name, extra = info
        return (
            f"The '{namespace}' namespace requires the optional plugin "
            f"'{plugin_name}'. Install it with: pip install {extra}"
        )
    info = EXPERIMENTAL_PLUGIN_NAMESPACES.get(namespace)
    if info is not None:
        plugin_name, extra = info
        return (
            f"The '{namespace}' namespace requires the experimental plugin "
            f"'{plugin_name}'. It is not officially supported, the API may change "
            f"without notice, and manual installation is required. "
            f"Install it with: pip install {extra}"
        )
    return None
