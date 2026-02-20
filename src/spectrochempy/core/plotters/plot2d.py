# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
2D plotting functions.

DEPRECATED: This module is now a re-export wrapper.
The implementation has moved to spectrochempy.plotting.plot2d.
"""

__all__ = [
    "plot_2D",
    "plot_contour",
    "plot_contourf",
    "plot_image",
    "plot_lines",
    "plot_map",
    "plot_stack",
]

__dataset_methods__ = __all__

# Re-export from new location for backward compatibility
from spectrochempy.plotting.plot2d import plot_2D
from spectrochempy.plotting.plot2d import plot_contour
from spectrochempy.plotting.plot2d import plot_contourf
from spectrochempy.plotting.plot2d import plot_image
from spectrochempy.plotting.plot2d import plot_lines
from spectrochempy.plotting.plot2d import plot_map
from spectrochempy.plotting.plot2d import plot_stack

plot_2D = plot_2D
plot_contour = plot_contour
plot_contourf = plot_contourf
plot_image = plot_image
plot_lines = plot_lines
plot_map = plot_map
plot_stack = plot_stack
