# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import lazy_loader as _lazy_loader

# --------------------------------------------------------------------------------------
# Lazy loading of sub-packages
# --------------------------------------------------------------------------------------
__getattr__, __dir__, __all__ = _lazy_loader.attach_stub(__name__, __file__)
