# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Correct an IR baseline using reference windows
==============================================

Fit a quadratic baseline to three windows between and around two synthetic
IR bands. The corrected trace should return close to zero away from the bands.
See the :doc:`baseline correction tutorial </userguide/processing/baseline>`
for choosing support windows and comparing correction methods.
"""

import spectrochempy as scp

x = scp.Coord.linspace(1800.0, 1400.0, 401, title="wavenumber", units="cm^-1")
band_a = scp.gaussian(x.data, pos=1660.0, width=38.0, normalized=False)
band_b = scp.gaussian(x.data, ampl=0.65, pos=1510.0, width=48.0, normalized=False)
background = 0.12 + 0.0004 * (x.data - 1400.0) + 0.18 * ((x.data - 1600.0) / 200.0) ** 2
spectrum = scp.NDDataset(band_a + band_b + background, coordset=[x], title="absorbance")

baseline = scp.Baseline(
    model="polynomial",
    order=2,
    ranges=[[1760.0, 1800.0], [1580.0, 1600.0], [1400.0, 1440.0]],
)
_ = baseline.fit(spectrum)
_ = baseline.plot(show_regions=True)
