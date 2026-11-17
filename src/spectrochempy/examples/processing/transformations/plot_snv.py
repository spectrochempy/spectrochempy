# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Standardize each spectrum with SNV
==================================

Apply SNV to an IR series and compare the measured and standardized
spectra. Each spectrum uses its own mean and standard deviation. See the
:doc:`transformations tutorial </userguide/processing/transformations>`
for how the chosen wavenumber window affects the result.
"""

import spectrochempy as scp

series = scp.read_omnic("irdata/nh4y-activation.spg")
spectra = series[::10, 4000.0:2000.0]
standardized = spectra.snv()
standardized.title = "standardized intensity"
standardized.units = None

_ = scp.multiplot(
    [spectra, standardized],
    method="lines",
    ncol=2,
    labels=["Measured spectra", "After SNV"],
    x_reverse=True,
    figsize=(10, 4),
)
