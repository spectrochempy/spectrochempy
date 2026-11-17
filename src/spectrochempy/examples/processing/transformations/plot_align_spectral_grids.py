# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Bring two spectra onto a common grid
====================================

Two synthetic spectra sample the same band on different wavenumber grids.
Interpolate the second spectrum onto the first grid before subtraction.
The difference then reflects the lower band intensity in the second
spectrum. See the :doc:`alignment tutorial </userguide/processing/alignment>`
for custom grids, masked samples, and joins on existing coordinates.
"""

import spectrochempy as scp

x = scp.Coord.linspace(1100.0, 1300.0, 101, title="wavenumber", units="cm^-1")
other_x = scp.Coord.linspace(1099.0, 1303.0, 69, title=x.title, units=x.units)
band = scp.gaussian(x.data, pos=1200.0, width=30.0, normalized=False)
other_band = scp.gaussian(
    other_x.data, pos=1200.0, width=30.0, ampl=0.8, normalized=False
)
first = scp.NDDataset(band, coordset=[x], title="intensity")
second = scp.NDDataset(other_band, coordset=[other_x], title="intensity")

first, second = scp.align(first, second, dim="x", method="interpolate")
_ = scp.plot_multiple(
    [first, second, first - second],
    method="pen",
    labels=["First spectrum", "Second on first grid", "Difference"],
    legend="best",
)
