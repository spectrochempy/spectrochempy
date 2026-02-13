# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Multiplot functions.

DEPRECATED: This module is now a re-export wrapper.
The implementation has moved to spectrochempy.plot.multiplot.
"""

__all__ = [
    "multiplot",
    "multiplot_map",
    "multiplot_stack",
    "multiplot_image",
    "multiplot_lines",
    "multiplot_scatter",
    "multiplot_with_transposed",
    "plot_with_transposed",
]

__dataset_methods__ = __all__

# Re-export from new location for backward compatibility
from spectrochempy.plot.multiplot import (
    multiplot,
    multiplot_map,
    multiplot_stack,
    multiplot_image,
    multiplot_lines,
    multiplot_scatter,
    multiplot_with_transposed,
    plot_with_transposed,
)

multiplot = multiplot
multiplot_map = multiplot_map
multiplot_stack = multiplot_stack
multiplot_image = multiplot_image
multiplot_lines = multiplot_lines
multiplot_scatter = multiplot_scatter
multiplot_with_transposed = multiplot_with_transposed
plot_with_transposed = plot_with_transposed
