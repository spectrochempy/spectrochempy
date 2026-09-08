# ======================================================================================
# Copyright (©) 2015-2026 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to provide a framework
# for processing, analysing and modelling *Spectro*scopic
# data for *Chem*istry with *Py*thon (SpectroChemPy).
#
# This software is governed by the CeCILL-B license under French law.
# ======================================================================================

"""
Plot profile API functions exposed at the ``spectrochempy`` root.

This is the single source of truth for the plot profile convenience
functions.  It is used both by the root ``__getattr__``/``__dir__``
implementation and by the reserved-name policy protecting public root
symbols from plugin collisions.
"""

_PLOT_PROFILE_FUNCTIONS = {
    "set_plot_profile": "spectrochempy.plotting.profile",
    "get_plot_profile": "spectrochempy.plotting.profile",
    "list_plot_profiles": "spectrochempy.plotting.profile",
    "save_plot_profile": "spectrochempy.plotting.profile",
    "delete_plot_profile": "spectrochempy.plotting.profile",
}
