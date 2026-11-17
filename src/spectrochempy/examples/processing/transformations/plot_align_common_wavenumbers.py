# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Compare average spectra over shared wavenumbers
===============================================

Compare two groups from an ammonium zeolite activation series. Their spectral
windows overlap in the OH stretching region. Keep the measured wavenumbers
shared by both groups, then plot the mean spectra and their difference.
See the :doc:`alignment tutorial </userguide/processing/alignment>` for
how joins handle matching coordinates and masks.
"""

import spectrochempy as scp

series = scp.read_omnic("irdata/nh4y-activation.spg")
earlier = series[15:25, 3800.0:3490.0]
later = series[35:45, 3720.0:3430.0]

earlier, later = scp.align(earlier, later, dim="x", method="inner")
earlier_mean = earlier.mean(dim="y")
later_mean = later.mean(dim="y")

_ = scp.plot_multiple(
    [earlier_mean, later_mean, later_mean - earlier_mean],
    method="pen",
    labels=["Earlier mean", "Later mean", "Later − earlier"],
    legend="best",
)
