# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

# lazy_stub: skip

"""
Public plugin API.

This is the **stable** import target for external plugins.
Internal modules may change without notice.
"""

from spectrochempy.api.plugins.base import SpectroChemPyPlugin
from spectrochempy.api.plugins.constants import CORE_PLUGIN_API_VERSION
from spectrochempy.api.plugins.hooks import hookimpl
from spectrochempy.api.plugins.hooks import hookspec
from spectrochempy.api.plugins.validation import validate_plugin_compatibility

__all__ = [
    "CORE_PLUGIN_API_VERSION",
    "SpectroChemPyPlugin",
    "hookimpl",
    "hookspec",
    "validate_plugin_compatibility",
]
