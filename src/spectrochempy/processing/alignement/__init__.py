# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import warnings

import lazy_loader as _lazy_loader

# --------------------------------------------------------------------------------------
# Lazy loading of sub-packages
# --------------------------------------------------------------------------------------
__getattr__, __dir__, __all__ = _lazy_loader.attach_stub(__name__, __file__)

warnings.warn(
    "Importing from `spectrochempy.processing.alignement` is deprecated and "
    "will be removed in 0.12.0. Use `spectrochempy.processing.alignment` instead.",
    DeprecationWarning,
    stacklevel=2,
)
