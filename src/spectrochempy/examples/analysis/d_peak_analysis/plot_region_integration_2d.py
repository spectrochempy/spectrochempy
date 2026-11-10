# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415, T201
"""
Integrate a 2D peak and estimate its uncertainty
================================================

Integrate a peak within physical bounds and estimate the noise from a separate
background region. The spectrum is synthetic so that the assumption of
independent noise with constant variance is known.
"""

import spectrochempy as scp

# %%
# Make a 2D spectrum with reproducible noise
# ------------------------------------------
x = scp.Coord.linspace(1950, 2150, 101, units="cm^-1", title="pump wavenumber")
y = scp.Coord.linspace(1950, 2150, 81, units="cm^-1", title="probe wavenumber")
grid = scp.zeros((y.size, x.size), coordset=[y, x], units="absorbance")
peak = {
    "height": 1.0,
    "axes": {
        "y": {"model": "gaussian", "center": 2040, "width": 20},
        "x": {"model": "gaussian", "center": 2070, "width": 25},
    },
}
signal = scp.simulate_peaks(grid, [peak])
noise = scp.normal(scale=0.015, size=grid.shape, seed=42, units=grid.units)
spectrum = signal + noise
spectrum.title = "Synthetic 2D spectrum"

# %%
# Choose signal and background boxes
# ----------------------------------
# Bounds use the coordinate units. They can fall between sampled points.
# The background samples must be separate from every sample used for the
# signal integral, including samples used to interpolate the signal bounds.

bounds = {"x": (2037, 2103), "y": (2012, 2069)}
noise_bounds = {"x": (1960, 2000), "y": (2080, 2130)}

ax = spectrum.plot_image(figsize=(7, 5), cmap="viridis", colorbar=True)
for box, color, label in (
    (bounds, "white", "Signal"),
    (noise_bounds, "orange", "Background"),
):
    left, right = box["x"]
    bottom, top = box["y"]
    _ = ax.plot(
        [left, right, right, left, left],
        [bottom, bottom, top, top, bottom],
        color=color,
        linewidth=1.5,
        label=label,
    )
_ = ax.legend()

# %%
# Integrate the signal and propagate the noise
# --------------------------------------------
# The result gives the signed integral and its standard uncertainty in the
# same units. The background estimates noise only. Its mean is not subtracted
# from the signal, and the uncertainty does not account for baseline drift
# or correlated noise.

result = spectrum.integrate_region(bounds, noise_bounds=noise_bounds)
if result.valid:
    print(f"Integral: {result.value:.3f} {result.unit:~P}")
    if result.uncertainty_status == "available":
        print(f"Standard uncertainty: {result.uncertainty:.3f} {result.unit:~P}")
    else:
        print(f"Uncertainty unavailable: {result.uncertainty_status}")
else:
    print(f"Integral unavailable: {result.reason}")
