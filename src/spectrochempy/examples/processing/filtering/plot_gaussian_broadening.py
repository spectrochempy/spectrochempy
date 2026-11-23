# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Gaussian broadening in coordinate units
=======================================

This example broadens narrow synthetic bands to a given instrumental
resolution with ``Filter(method="gaussian", sigma_units="coordinate")``.
With ``sigma_units="coordinate"``, ``sigma`` is expressed in the units of the
coordinate (here cm⁻¹), so the broadening does not depend on the sampling step.
"""

import numpy as np

import spectrochempy as scp


# %%
# Narrow bands sampled on two grids
# ---------------------------------
# The same two bands (FWHM 2 cm⁻¹) are sampled every 0.25 cm⁻¹ and every
# 0.5 cm⁻¹. The ``sigma_units="coordinate"`` option requires a uniformly
# spaced coordinate.
def narrow_spectrum(x):
    spectrum = scp.gaussian(x, ampl=1.0, pos=1080.0, width=2.0, normalized=False)
    spectrum += scp.gaussian(x, ampl=0.6, pos=1100.0, width=2.0, normalized=False)
    return spectrum


fine = narrow_spectrum(
    scp.Coord.linspace(1000.0, 1200.0, 801, units="cm^-1", title="wavenumber")
)
coarse = narrow_spectrum(
    scp.Coord.linspace(1000.0, 1200.0, 401, units="cm^-1", title="wavenumber")
)

# %%
# Broadening to an 8 cm⁻¹ resolution
# ----------------------------------
# For a Gaussian, the full width at half maximum (FWHM) is
# :math:`\sqrt{8 \ln 2}\,\sigma`.
resolution = 8.0
sigma = resolution / np.sqrt(8 * np.log(2))
broadening = scp.Filter(method="gaussian", sigma=sigma, sigma_units="coordinate")
fine_broad = broadening(fine)
coarse_broad = broadening(coarse)

_ = scp.plot_multiple(
    [fine, fine_broad, coarse_broad],
    method="pen",
    labels=["narrow bands", "broadened (0.25 cm⁻¹ grid)", "broadened (0.5 cm⁻¹ grid)"],
    ls=["-", "-", "--"],
    legend="best",
    title=f"Gaussian broadening, FWHM = {resolution} cm⁻¹",
)


# %%
# Measured widths
# ---------------
# The convolution of two Gaussians is a Gaussian whose FWHM is the quadratic
# sum of both widths. The result is the same on both grids. With the default
# ``sigma_units="points"``, the same ``sigma`` is a number of points, and the
# broadening depends on the sampling step.
def fwhm(spectrum):
    band = spectrum[1070.0:1090.0]
    return band.characterize_peaks_1d(band.argmax()).rows[0]["fwhm"].magnitude


in_points = scp.Filter(method="gaussian", sigma=sigma)
print(f"expected FWHM: {np.hypot(2.0, resolution):.2f} cm⁻¹")
print(f"coordinate units, 0.25 cm⁻¹ grid: {fwhm(fine_broad):.2f} cm⁻¹")
print(f"coordinate units, 0.5 cm⁻¹ grid: {fwhm(coarse_broad):.2f} cm⁻¹")
print(f"points, 0.25 cm⁻¹ grid: {fwhm(in_points(fine)):.2f} cm⁻¹")
print(f"points, 0.5 cm⁻¹ grid: {fwhm(in_points(coarse)):.2f} cm⁻¹")

# %%
# Uncomment the following line to display all figures when running the script
# directly with Python.

# %%

# scp.show()
