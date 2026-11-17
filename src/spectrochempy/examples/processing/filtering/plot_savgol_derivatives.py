# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Differentiate a band with Savitzky–Golay
========================================

Calculate the first derivative of a noisy band on a descending wavenumber
axis. The derivative crosses zero near the band maximum. See the
:doc:`smoothing tutorial </userguide/processing/smoothing>` for choosing the
window and understanding the signed spacing and derivative units.
"""

import spectrochempy as scp

x = scp.Coord.linspace(2400.0, 1600.0, 401, title="wavenumber", units="cm^-1")
band = scp.gaussian(x.data, pos=2000.0, width=120.0, normalized=False)
noise = scp.normal(scale=0.005, size=x.size, seed=42)
spectrum = scp.NDDataset(band + noise.data, coordset=[x], units="absorbance")

derivative = spectrum.differentiate(derivative_order=1, size=21, polynomial_order=3)
_ = scp.multiplot(
    [spectrum, derivative],
    method="lines",
    ncol=2,
    labels=["Noisy band", "First derivative"],
    figsize=(9, 3.5),
    x_reverse=True,
)
