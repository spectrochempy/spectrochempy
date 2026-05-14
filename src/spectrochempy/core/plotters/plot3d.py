# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
3D plotting functions.

DEPRECATED: This module is now a re-export wrapper.
The implementation has moved to spectrochempy.plotting.plot3d.
"""

__all__ = ["plot_3D", "plot_surface", "plot_waterfall"]

__dataset_methods__ = __all__

# Re-export from new location for backward compatibility
from spectrochempy.plotting.plot3d import plot_3D
from spectrochempy.plotting.plot3d import plot_surface
from spectrochempy.plotting.plot3d import plot_waterfall
