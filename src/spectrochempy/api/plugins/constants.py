# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Plugin API compatibility constants.

Central location for API versioning used during plugin validation.
"""

CORE_PLUGIN_API_VERSION = "1.0"
"""The current SpectroChemPy plugin API version.

Plugins declaring a matching ``plugin_api_version`` (major version)
are considered compatible.  A mismatch causes the plugin to be
skipped with a warning.
"""
