# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Public hook markers for SpectroChemPy plugins.

Re-exports the internal ``hookspec`` / ``hookimpl`` markers so that
external plugins can import them from the stable API namespace::

    from spectrochempy.api.plugins import hookspec, hookimpl
"""

from spectrochempy.plugins.hooks import hookimpl  # noqa: F401
from spectrochempy.plugins.hooks import hookspec  # noqa: F401
