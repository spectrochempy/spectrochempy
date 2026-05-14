# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Matplotlib plotting setup and state restoration utilities.

This module re-exports from spectrochempy.core.plotters.plot_setup
for backward compatibility with the new plot package structure.
"""

# Re-export everything from the original location
from spectrochempy.core.plotters.plot_setup import _is_mpl_initialized
from spectrochempy.core.plotters.plot_setup import _set_mpl_state
from spectrochempy.core.plotters.plot_setup import ensure_spectrochempy_plot_style
from spectrochempy.core.plotters.plot_setup import get_import_time_rcparams
from spectrochempy.core.plotters.plot_setup import lazy_ensure_mpl_config
from spectrochempy.core.plotters.plot_setup import restore_rcparams

__all__ = [
    "_is_mpl_initialized",
    "_set_mpl_state",
    "ensure_spectrochempy_plot_style",
    "get_import_time_rcparams",
    "lazy_ensure_mpl_config",
    "restore_rcparams",
]
