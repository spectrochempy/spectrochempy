# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Track the area of an IR band over time
======================================

Correct the baseline of an IR region and plot one integrated area per
spectrum. See the :doc:`peak integration tutorial </userguide/analysis/peak_integration>`
for the choice of baseline windows and a comparison with Simpson's rule.
"""

import spectrochempy as scp

spectra = scp.read_omnic("irdata/nh4y-activation.spg")
region = spectra[:20, 1250.0:1800.0]
region.y -= region.y[0]
region.y.ito("min")
region.y.title = "elapsed time"

# %%
# Inspect the spectral region used to calculate each area.

_ = region[::4].plot(method="lines", title="Measured IR region before correction")

# %%
# Remove the baseline before integrating each spectrum.

baseline = scp.Baseline(
    model="polynomial",
    order=3,
    ranges=[[1740.0, 1800.0], [1550.0, 1570.0], [1250.0, 1300.0]],
)
_ = baseline.fit(region)
area = baseline.corrected.trapezoid(dim="x")
_ = area.plot(scatter=True)
