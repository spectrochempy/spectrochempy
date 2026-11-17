# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Remove a spike along the acquisition axis
=========================================

A narrow Raman band persists across this synthetic series. One acquisition
has a positive spike at the band maximum. Remove that spike by comparing
the neighboring acquisitions at each Raman shift. See the :doc:`denoising
tutorial </userguide/processing/denoising>` for when to use this method.
"""

import spectrochempy as scp

shift = scp.Coord.linspace(980.0, 1040.0, 121, title="Raman shift", units="cm^-1")
band = scp.gaussian(shift.data, pos=1010.0, width=5.0, normalized=False)
reference = scp.NDDataset(band, coordset=[shift], title="intensity")
series = scp.stack(*([reference] * 9))
series.y = scp.Coord.linspace(0.0, 8.0, 9, title="time", units="s")
series[4, 1010.0] = 2.5

cleaned = series.despike(method="neighbors", dim="y")

_ = scp.plot_multiple(
    [series[4], cleaned[4]],
    method="pen",
    labels=["Acquisition with spike", "Corrected acquisition"],
    legend="best",
)
