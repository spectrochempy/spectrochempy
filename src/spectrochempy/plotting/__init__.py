# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
SpectroChemPy Plot Package.

This package provides plotting functionality for SpectroChemPy datasets,
with support for multiple backends (matplotlib, plotly).

Modules
-------
dispatcher : Main plotting dispatcher with backend routing
backends : Backend implementations (matplotlib, plotly)
plot1d : 1D plotting functions
plot2d : 2D plotting functions
plot3d : 3D plotting functions
multiplot : Multi-panel plotting
plot_setup : Matplotlib lazy initialization
mpl_assets : Style and font management
"""

import lazy_loader as _lazy_loader

# --------------------------------------------------------------------------------------
# Lazy loading of sub-packages
# --------------------------------------------------------------------------------------
__getattr__, __dir__, __all__ = _lazy_loader.attach_stub(__name__, __file__)
