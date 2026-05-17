# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""Known optional plugin features used for user-facing installation hints."""

from __future__ import annotations

KNOWN_PLUGIN_READERS = {
    "topspin": ("nmr", "spectrochempy-nmr", "spectrochempy[nmr]"),
}

KNOWN_PLUGIN_NAMESPACES = {
    "nmr": ("spectrochempy-nmr", "spectrochempy[nmr]"),
    "iris": ("spectrochempy-iris", "spectrochempy[iris]"),
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
