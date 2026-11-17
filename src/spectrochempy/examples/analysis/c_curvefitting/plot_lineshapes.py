# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Compare common spectral line shapes
===================================

Overlay profiles with a common center and unit peak height to compare
their tails and symmetry. See the :doc:`fitting tutorial </userguide/analysis/fitting>`
for the width, area, and shape parameters.
"""

import spectrochempy as scp

x = scp.Coord.linspace(1540.0, 1660.0, 241, title="wavenumber", units="cm^-1")
params = {"pos": 1600.0, "width": 20.0, "normalized": False}
profiles = [
    scp.gaussian(x.data, **params),
    scp.lorentzian(x.data, **params),
    scp.voigt(x.data, ratio=0.5, **params),
    scp.asymmetricvoigt(x.data, ratio=0.5, asym=0.8, **params),
]
labels = ["Gaussian", "Lorentzian", "Voigt", "Asymmetric Voigt"]
shapes = scp.NDDataset(
    profiles, coordset=[scp.Coord(labels=labels), x], title="relative intensity"
)
_ = shapes.plot_pen(legend=labels, legend_loc="upper right")
