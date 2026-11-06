# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Calculate 2D-COS maps from an IR activation series
==================================================

This example interpolates a missing acquisition onto a uniform time grid and
calculates synchronous and asynchronous 2D correlation maps with
``correlation_2d``.
"""

import spectrochempy as scp

# %%
# Load the local IR series and regularize acquisition times
# ---------------------------------------------------------
spectra = scp.read("irdata/nh4y-activation.spg")
spectra.y.title = "acquisition time"
uniform_times = scp.Coord.linspace(
    float(spectra.y.data[0]),
    float(spectra.y.data[-1]),
    spectra.shape[0] + 1,
    title=spectra.y.title,
    units=spectra.y.units,
)
spectra = spectra.interpolate(dim="y", coord=uniform_times, method="linear")

# Restrict the spectral range and downsample for a compact correlation map.
region = spectra[:, 1800.0:1000.0][:, ::4]
_ = region.plot()

# %%
# Calculate and inspect synchronous and asynchronous maps
# ---------------------------------------------------------
result = scp.correlation_2d(region, perturbation_dim="y")
_ = result.synchronous.plot()
_ = result.asynchronous.plot()

# %%
# Both maps carry the same wavenumber axes; the result also records the
# perturbation dimension used in the calculation.
result.synchronous.shape, result.asynchronous.shape, result.perturbation_dim
