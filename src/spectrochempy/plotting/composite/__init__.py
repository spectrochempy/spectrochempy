## ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Composite plotting module."""

__all__ = [
    "plotmerit",
    "plot_iris_lcurve",
    "plot_iris_distribution",
    "plot_iris_merit",
    "plot_scree",
    "plot_score",
]

from spectrochempy.plotting.composite.iris import plot_iris_distribution
from spectrochempy.plotting.composite.iris import plot_iris_lcurve
from spectrochempy.plotting.composite.iris import plot_iris_merit
from spectrochempy.plotting.composite.plotmerit import plotmerit
from spectrochempy.plotting.composite.plotscree import plot_scree
from spectrochempy.plotting.composite.plotscore import plot_score
