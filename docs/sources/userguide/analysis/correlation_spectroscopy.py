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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Two-dimensional correlation spectroscopy (2D-COS)
#
# Use 2D-COS to examine how a series of spectra changes with a controlled
# perturbation such as time or temperature. Both axes of the resulting
# maps are spectral coordinates. The calculation follows the order of
# the spectra along the perturbation axis.
#
# The method needs a 2D dataset with finite real intensities and a numeric,
# monotonic, uniformly spaced perturbation coordinate. Resolve masked or
# missing measurements before calculating the maps. Any
# interpolation during preparation changes the data used by the transform.
#
# ```{=rst}
# See :cite:t:`noda:1993` for the generalized 2D correlation method.
#
# In zeolite catalysis, :cite:t:`thibault-starzyk:2000` used 2D-COS to track
# xylene isomers during reaction over H-MFI, and :cite:t:`moulin:2008`
# applied 2D correlation to probe Brønsted acid sites. In a study of
# acid-site speciation in hierarchical ZSM-5, :cite:t:`fernandez:2010`
# used operando IR correlation maps to compare changes in hydroxyl
# and aromatic bands during o-xylene isomerization.
# ```

# %% [markdown]
# ## Put the perturbation on a uniform grid
#
# This example follows an NH₄Y activation series recorded over time.
# The series has one missing acquisition. A uniform time axis requires
# interpolating the spectra themselves. Assigning new time coordinates
# to the original spectra would leave the gap in the measurements.
# Here linear interpolation estimates the missing spectrum. Document any
# such step when interpreting the correlations.

# %%
import logging

import spectrochempy as scp

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

# %% [markdown]
# ## Exclude saturation and estimate the baseline
#
# Restrict the range to 1250–4000 cm⁻¹ to exclude saturation before fitting
# the baseline. Fit a quadratic polynomial in three background windows.
# The multivariate SVD model fits the baseline using the first three
# components of the support data and reconstructs a baseline for every
# spectrum. The transform subtracts those baselines from the original
# spectra.
#
# Choose the support windows and component count for the dataset at hand.
# Inspect the before and after comparison for changes in broad bands and
# for extrapolation beyond the support windows.

# %%
region = spectra[:, 4000.0:1250.0]
baseline = scp.Baseline(
    model="polynomial",
    order=2,
    multivariate="svd",
    n_components=3,
    ranges=[[3900.0, 3980.0], [2100.0, 2200.0], [1800.0, 1900.0]],
    include_limits=False,
)
previous_log_threshold = logging.root.manager.disable
logging.disable(max(previous_log_threshold, logging.INFO))
try:
    _ = baseline.fit(region)
    corrected = baseline.transform()
finally:
    logging.disable(previous_log_threshold)

_ = scp.multiplot(
    [region[::5], corrected[::5]],
    method="lines",
    ncol=2,
    labels=["Before baseline correction", "After polynomial SVD correction"],
    cmap="viridis",
    figsize=(11, 3.5),
    x_reverse=True,
)

# %% [markdown]
# ## Calculate and interpret the maps
#
# The calculation subtracts the mean spectrum to form the dynamic
# spectra $A$. The synchronous map is $A^{T} A/(n-1)$, where $n$ is the
# number of spectra. Positive cross-peaks indicate intensities varying
# in the same direction, and negative ones indicate opposite variation.
# Its diagonal gives the variance at each wavenumber.
#
# The asynchronous map is $A^{T} N A/(n-1)$, where $N$ is the discrete
# Hilbert–Noda matrix. This map is antisymmetric with a zero diagonal.
# Proportional variations give no asynchronous signal. Reversing the
# perturbation order reverses the asynchronous signs and preserves the
# synchronous map.
#
# To interpret the order of changes, consider the synchronous and
# asynchronous signs at the same ordered pair of wavenumbers and keep
# the perturbation direction consistent. Evidence for a reaction mechanism
# also requires chemical assignments and independent context.
#
# Thin the spectral grid after baseline fitting to limit the size of the
# maps. Keep every point along the uniform time grid.

# %%
result = scp.correlation_spectroscopy(corrected[:, ::4], perturbation_dim="y")

# %% [markdown]
# ## Plot the maps with reference spectra
#
# Divide each display map by its own maximum absolute intensity to make
# weaker asynchronous features visible. Compare absolute amplitudes
# using the original maps, which retain squared signal units. Red solid
# contours are positive and blue dashed contours are negative.
#
# The top panels show spectra from the start, middle, and end of the
# corrected series. Pass these spectra as external references with
# `projection_data` so that their bands can be compared to cross-peaks
# on the spectral axis. See the [plotting guide](../plotting/index.rst)
# for calculated projections and layout options.

# %%
maps = [result.synchronous, result.asynchronous]
maxima = [float(abs(dataset.data).max()) for dataset in maps]
display_maps = []
for dataset, maximum in zip(maps, maxima, strict=True):
    display = dataset.copy()
    display.data = dataset.data / maximum if maximum else dataset.data.copy()
    display.units = None
    display_maps.append(display)

fractions = [0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9]
levels = [-value for value in fractions[::-1]] + fractions
indices = [0, corrected.shape[0] // 2, corrected.shape[0] - 1]
references = [corrected[index].squeeze() for index in indices]
reference_style = {
    "labels": ["Start", "Middle", "End"],
    "color": ["0.5", "#a08010", "0.1"],
    "intensity_label": "Corrected\nabsorbance",
}
panel_options = [
    {
        "projection_kwargs": {
            "x": {
                **reference_style,
                "title": f"{title}: max |C| = {maximum:.3g} {dataset.units:~P}",
                "legend": "upper center" if index == 0 else False,
            }
        }
    }
    for index, (title, maximum, dataset) in enumerate(
        zip(["Synchronous", "Asynchronous"], maxima, maps, strict=True)
    )
]

_ = scp.multiplot(
    display_maps,
    method="contour",
    ncol=2,
    levels=levels,
    positive_color="#b2182b",
    negative_color="#2166ac",
    positive_linestyle="-",
    negative_linestyle="--",
    projections=("x",),
    projection_data={"x": references},
    projection_size=0.25,
    panel_kwargs=panel_options,
    equal_aspect=True,
    diagonal=True,
    xlim=(4000, 1250),
    ylim=(4000, 1250),
    xlabel="Wavenumber / cm$^{-1}$",
    ylabel="Wavenumber / cm$^{-1}$",
    colorbar=False,
    figsize=(12, 8),
    suptitle=(
        "NH4Y activation: 2D-COS\n"
        "Red solid: positive; blue dashed: negative — levels scaled separately"
    ),
    return_dict=True,
    show=False,
)

# %% [markdown]
# The [short gallery example](../../gettingstarted/examples/gallery/auto_examples_analysis/d_peak_analysis/plot_2d_cos_ir.rst)
# uses two simulated bands on an already uniform time grid. See the
# [API reference](../../reference/generated/spectrochempy.correlation_spectroscopy.rst)
# for the input requirements and sign convention.
