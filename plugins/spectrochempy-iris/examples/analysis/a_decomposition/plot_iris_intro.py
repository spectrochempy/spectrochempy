# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415

"""
Plot the IRIS distribution for CO adsorption
============================================

Fit a Langmuir kernel to CO adsorption spectra and plot the recovered
affinity distribution. This example requires the
``spectrochempy-iris`` plugin.

See the :doc:`IRIS tutorial </userguide/analysis/iris>` for data preparation,
regularization selection, and interpretation.
"""

import spectrochempy as scp

# %%
# Read the CO stretching region and set the pressure of each spectrum in torr.
spectra = scp.read_omnic("irdata/CO@Mo_Al2O3.SPG")[:, 2250.0:1950.0]
pressures = [
    0.003,
    0.004,
    0.009,
    0.014,
    0.021,
    0.026,
    0.036,
    0.051,
    0.093,
    0.150,
    0.203,
    0.300,
    0.404,
    0.503,
    0.602,
    0.702,
    0.801,
    0.905,
    1.004,
]
spectra.y = scp.Coord(pressures, title="pressure", units="torr")
_ = spectra.plot(colorbar=True)

# %%
# Fit once at a fixed regularization strength of :math:`10^{-4}`.
kernel = spectra.iris.kernel_matrix(kernel_type="langmuir", q=[-8, -1, 50])
analysis = scp.iris.IRIS(reg_par=[-4, -4, 1])
_ = analysis.fit(spectra, kernel)
_ = analysis.result.f[0].plot_contour()
