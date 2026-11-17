# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Peak Integration
#
# This tutorial integrates a baseline-corrected infrared region for each
# spectrum in a time series. For the preparation steps, see
# [Import IR](../importexport/importIR.rst),
# [Slicing](../processing/slicing.rst), and
# [Baseline Correction](../processing/baseline.rst) tutorials.

# %% [markdown]
# First, let's import the SpectroChemPy API.

# %%
import spectrochempy as scp

# %% [markdown]
# Now, import some 2D data into an NDDataset object.

# %%
ds = scp.read_omnic("irdata/nh4y-activation.spg")
ds

# %% [markdown]
# It's a series of 55 spectra.
#
# For the demonstration, select only the first 20 on a limited region from 1250 to
# 1800 cm$^{-1}$ (Do not forget to
# use floating numbers for slicing).

# %%
X = ds[:20, 1250.0:1800.0]

# %% [markdown]
# We can also eventually remove the offset on the acquisition time dimension (y).

# %%
X.y -= X.y[0]
X.y.ito("min")
X.y.title = "acquisition time"

# %% [markdown]
# We set some plotting preferences and then plot the raw data.

# %%
prefs = scp.preferences
prefs.figure.figsize = (6, 3)
prefs.colormap = "Dark2"
prefs.colorbar = True
_ = X.plot()

# %% [markdown]
# Fit a cubic polynomial to support windows near the edges of the region
# and between bands. Choose windows that follow the background and inspect
# the corrected spectra before integrating. A baseline that runs through
# a band changes its measured area. The same windows are used for all rows.

# %%
blc = scp.Baseline()
blc.ranges = (
    [1740.0, 1800.0],
    [1550.0, 1570.0],
    [1250.0, 1300.0],
)
blc.model = "polynomial"
blc.order = 3

_ = blc.fit(X)  # fit the baseline

Xcorr = blc.corrected  # get the corrected dataset
_ = Xcorr.plot()

# %% [markdown]
# Integrate along the wavenumber axis (`dim="x"`) to obtain one area per
# acquisition time. The methods use the wavenumber coordinates, so the
# area includes the spacing between points and carries the signal unit
# multiplied by the wavenumber unit. SpectroChemPy adjusts for a descending
# axis. The area is signed, so signal below the fitted baseline subtracts
# from signal above it. If a row contains a masked point inside the
# integration window, its area is masked.

# %%
inttrapz = Xcorr.trapezoid(dim="x")
intsimps = Xcorr.simpson(dim="x")

# %% [markdown]
# `dataset.simpson(dim="x")` and the
# equivalent top-level `scp.simpson(dataset, dim="x")` produce the same result.
# Do not confuse this integration function with `scp.read_simpson(...)`, which
# reads SIMPSON NMR simulation data files — the two are unrelated operations
# that happen to share the "simpson" name.

# %% [markdown]
# As you can see, both methods give almost the same results in this case.

# %%
_ = scp.plot_multiple(
    method="scatter",
    ms=5,
    datasets=[inttrapz, intsimps],
    labels=["trapezoidal rule", "Simpson's rule"],
    legend="best",
)

# %% [markdown]
# The difference between the trapezoidal and Simpson integration methods is visualized
# below as a percentage of the Simpson area. This comparison is meaningful
# when the reference area is sufficiently far from zero.

# %%
diff = (inttrapz - intsimps) * 100.0 / intsimps
diff.title = "difference"
diff.units = "percent"
_ = diff.plot(scatter=True, ms=5)


# %% [markdown]
# The [short gallery example](../../gettingstarted/examples/gallery/auto_examples_analysis/d_peak_analysis/plot_peak_integration.rst)
# shows the baseline correction and trapezoidal area in a single workflow.


# %% [markdown]
# ## Integrate a rectangular region in 2D
#
# Use `integrate_region()` to obtain one signed volume over a box defined
# in physical coordinates. Specify lower and upper bounds for both native
# dimensions. Bare numbers use the coordinate units, and compatible
# quantities are converted. Bounds between samples are handled by linear
# interpolation along each axis. Both axes must be strictly monotone.
# Masked or non-finite samples with nonzero integration weight invalidate
# the result. The volume unit is the signal unit multiplied by both axis
# units.
#
# The [short 2D gallery example](../../gettingstarted/examples/gallery/auto_examples_analysis/d_peak_analysis/plot_region_integration_2d.rst)
# integrates the synthetic peak without noise. Here we add reproducible noise
# and estimate its effect on the integral.
#
# %% [markdown]
# ### Make a 2D spectrum with reproducible noise

# %%
x = scp.Coord.linspace(1950, 2150, 101, units="cm^-1", title="pump wavenumber")
y = scp.Coord.linspace(1950, 2150, 81, units="cm^-1", title="probe wavenumber")
probe = scp.gaussian(y.data, pos=2040, width=20, normalized=False)
pump = scp.gaussian(x.data, pos=2070, width=25, normalized=False)
signal = scp.NDDataset(probe[:, None] * pump, coordset=[y, x], units="absorbance")
noise = scp.normal(scale=0.015, size=signal.shape, seed=42, units=signal.units)
spectrum = signal + noise
spectrum.title = "Synthetic 2D spectrum"

# %% [markdown]
# ### Choose signal and background boxes
#
# Center the signal box on the peak at 2070 cm⁻¹ (pump) and 2040 cm⁻¹
# (probe). The bounds extend roughly 1.3 to 1.5 peak widths from the
# center along each axis, so they include most of the Gaussian tails.
# The background box is in a flat part of the map, well away from the peak.
# It shares no samples with the signal integral, including the neighbors
# used to interpolate the bounds.
#
# Enter both pairs of bounds from low to high. The plot displays pump
# wavenumbers decreasing to the right and probe wavenumbers increasing
# downward, which places the background box in the lower right.

# %%
bounds = {"x": (2037, 2103), "y": (2012, 2069)}
noise_bounds = {"x": (1960, 2000), "y": (2080, 2130)}

ax = spectrum.plot_image(figsize=(7, 5), cmap="viridis", colorbar=True)
for box, color, label in (
    (bounds, "white", "Signal"),
    (noise_bounds, "orange", "Background"),
):
    x_min, x_max = box["x"]
    y_min, y_max = box["y"]
    _ = ax.plot(
        [x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min],
        color=color,
        linewidth=1.5,
        label=label,
    )
_ = ax.legend()

# %% [markdown]
# ### Integrate the signal and propagate the noise
#
# The result gives the signed integral and its standard uncertainty in the
# same units. The propagation assumes additive, independent noise with
# constant variance. Supply `noise_std` instead if the sample noise is known.
# Use only one of the two arguments. The calculation leaves the signal
# baseline as supplied, and the uncertainty covers independent noise only.

# %%
result = spectrum.integrate_region(bounds, noise_bounds=noise_bounds)
if result.valid:
    print(f"Integral: {result.value:.3f} {result.unit:~P}")
    if result.uncertainty_status == "available":
        print(f"Standard uncertainty: {result.uncertainty:.3f} {result.unit:~P}")
    else:
        print(f"Uncertainty unavailable: {result.uncertainty_status}")
else:
    print(f"Integral unavailable: {result.reason}")
