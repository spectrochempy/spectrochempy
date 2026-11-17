# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Mark detected peaks on an IR spectrum
=====================================

Find peaks in the last spectrum of a CO adsorption series and mark their
positions. See the :doc:`peak finding tutorial </userguide/analysis/peak_finding>`
for the detection criteria and an example of tracking a band over time.
"""

import spectrochempy as scp

spectra = scp.read("irdata/CO@Mo_Al2O3.SPG")
spectrum = spectra[-1, 2300.0:1900.0]
peaks, _ = spectrum.find_peaks(distance="5 cm^-1", prominence=0.02)

ax = spectrum.plot_pen()
_ = peaks.plot_scatter(
    ax=ax,
    clear=False,
    data_only=True,
    marker="v",
    color="black",
)
