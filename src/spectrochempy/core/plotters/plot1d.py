## ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Module containing 1D plotting function(s).

DEPRECATED: This module is now a re-export wrapper.
The implementation has moved to spectrochempy.plotting.plot1d.
"""

__all__ = [
    "plot_1D",
    "plot_pen",
    "plot_scatter",
    "plot_bar",
    "plot_multiple",
    "plot_scatter_pen",
]

__dataset_methods__ = __all__

# Re-export from new location for backward compatibility
from spectrochempy.plotting.plot1d import plot_1D
from spectrochempy.plotting.plot1d import plot_bar
from spectrochempy.plotting.plot1d import plot_multiple
from spectrochempy.plotting.plot1d import plot_pen
from spectrochempy.plotting.plot1d import plot_scatter
from spectrochempy.plotting.plot1d import plot_scatter_pen
