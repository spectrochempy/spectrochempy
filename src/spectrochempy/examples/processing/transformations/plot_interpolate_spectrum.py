# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Interpolate a spectrum onto a denser grid
=========================================

Evaluate a sampled band on a finer wavenumber grid with PCHIP. The
markers show the measurements and the curve shows the interpolated values.
See the :doc:`alignment tutorial </userguide/processing/alignment>` for
a comparison with linear interpolation and the options for points outside
the measured range.
"""

import spectrochempy as scp

x = scp.Coord.linspace(1140.0, 1260.0, 13, title="wavenumber", units="cm^-1")
band = scp.gaussian(x.data, pos=1197.0, width=32.0, normalized=False)
spectrum = scp.NDDataset(band, coordset=[x], title="intensity")

target = scp.Coord.linspace(1140.0, 1260.0, 121, title=x.title, units=x.units)
resampled = spectrum.interpolate(dim="x", coord=target, method="pchip")
_ = scp.plot_multiple(
    [resampled, spectrum],
    method="pen",
    labels=["PCHIP interpolation", "Input samples"],
    marker=["None", "o"],
    ls=["-", "None"],
    legend="best",
)
