# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Subtract the mean IR spectrum
=============================

Center an activation series along the acquisition axis to show how
each spectrum differs from the mean at each wavenumber. See the
:doc:`transformations tutorial </userguide/processing/transformations>`
for per-spectrum centering and autoscaling.
"""

import spectrochempy as scp

series = scp.read_omnic("irdata/nh4y-activation.spg")
spectra = series[:, 4000.0:2000.0]
centered = spectra.center(dim="y")

_ = centered[::10].plot(
    method="lines",
    title="Deviations from the mean spectrum",
    x_reverse=True,
)
