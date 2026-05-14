# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Bruker TopSpin reader - moved to external plugin.

This reader has been moved to the external package ``spectrochempy-topspin``.
Install it with::

    pip install spectrochempy-topspin

The plugin auto-registers via the entry point ``spectrochempy.plugins``
so that ``scp.read_topspin(...)`` works transparently after installation.
"""

from spectrochempy.plugins.deps import MissingPluginError


def read_topspin(*paths, **kwargs):
    raise MissingPluginError(
        "read_topspin",
        plugin_name="spectrochempy-topspin",
        install_hint="pip install spectrochempy-topspin",
    )
