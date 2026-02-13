# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
2D plotting functions.

DEPRECATED: This module is now a re-export wrapper.
The implementation has moved to spectrochempy.plot.plot2d.
"""

__all__ = [
    "plot_2D",
    "plot_map",
    "plot_stack",
    "plot_image",
]

__dataset_methods__ = __all__

# Re-export from new location for backward compatibility
from spectrochempy.plot.plot2d import (
    plot_2D,
    plot_map,
    plot_stack,
    plot_image,
)

plot_2D = plot_2D
plot_map = plot_map
plot_stack = plot_stack
plot_image = plot_image
