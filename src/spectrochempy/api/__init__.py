# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

# lazy_stub: skip

"""
SpectroChemPy Public API.

Stable public API surface.  Internal modules (everything outside
:mod:`spectrochempy.api`) may change without notice.

External plugins should import only from this namespace and from
:mod:`spectrochempy.api.plugins`.
"""

from spectrochempy.api.plugins import CORE_PLUGIN_API_VERSION
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import SpectroChemPyPlugin

__all__ = [
    "CORE_PLUGIN_API_VERSION",
    "PluginCapability",
    "SpectroChemPyPlugin",
]
