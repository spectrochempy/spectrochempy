# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import lazy_loader as _lazy_loader

from spectrochempy.utils.decorators import warn_deprecated

# --------------------------------------------------------------------------------------
# Lazy loading of sub-packages
# --------------------------------------------------------------------------------------
__getattr__, __dir__, __all__ = _lazy_loader.attach_stub(__name__, __file__)

warn_deprecated(
    "spectrochempy.processing.alignement",
    kind="import path",
    replace="spectrochempy.processing.alignment",
    policy=True,
    action="is deprecated",
    stacklevel=2,
)
