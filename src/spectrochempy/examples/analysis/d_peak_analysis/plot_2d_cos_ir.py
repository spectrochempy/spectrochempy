# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Explore an IR activation series with Noda's 2D-COS
==================================================

Generalized two-dimensional correlation spectroscopy (2D-COS) compares the
changes of spectral intensities along an external perturbation, here acquisition
time. Both axes of the resulting maps are wavenumbers; time is the ordered
sequence over which correlations are calculated.

The method follows I. Noda, *Generalized Two-Dimensional Correlation Method
Applicable to Infrared, Raman, and Other Types of Spectroscopy*, Applied
Spectroscopy **47**, 1329–1336 (1993),
`doi:10.1366/0003702934067694 <https://doi.org/10.1366/0003702934067694>`_.

We use an NH4Y activation series, remove its saturated low-wavenumber region,
and illustrate a multivariate polynomial baseline correction before calculating
synchronous and asynchronous maps with ``correlation_spectroscopy``.
"""

# sphinx_gallery_thumbnail_number = 2

import spectrochempy as scp

# %%
# Regularize the perturbation coordinate
# --------------------------------------
# The series has one missing acquisition. The discrete Hilbert–Noda transform
# requires equally spaced perturbation points: merely relabeling the time axis
# would not supply the missing spectrum. Linear interpolation estimates it on
# a uniform grid, without creating a new independent measurement. This choice
# can affect the correlations and should be reported in a real analysis.
# ``Coord`` keeps its standard 0.1 percent linearization tolerance.
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

# %%
# Exclude saturation and estimate the baseline
# --------------------------------------------
# Retain 1250–4000 cm-1: the saturated region below 1250 cm-1 is excluded
# before baseline fitting and correlation. Saturated intensities cannot be
# recovered by baseline correction.
region = spectra[:, 4000.0:1250.0]

# A quadratic polynomial is fitted in three selected background windows.
# With ``multivariate="svd"``, Baseline fits the first three SVD components
# of the support data and reconstructs a baseline for every spectrum. This is
# the existing multivariate SVD approach, without a separate centered PCA.
# Only the reconstructed baseline is subtracted; the spectral series itself
# is not replaced by a three-component reconstruction.
#
# These windows and the component count are illustrative choices for this
# dataset. Inspect the before/after comparison, especially broad absorption
# bands and extrapolation beyond the outer support windows. For another
# sample, select background windows again rather than copying these values.
baseline = scp.Baseline(
    model="polynomial",
    order=2,
    multivariate="svd",
    n_components=3,
    ranges=[[3900.0, 3980.0], [2100.0, 2200.0], [1800.0, 1900.0]],
    include_limits=False,
)
baseline.fit(region)
corrected = baseline.transform()

_ = scp.multiplot(
    [region[::5], corrected[::5]],
    method="lines",
    ncol=2,
    labels=["Before baseline correction", "After polynomial SVD correction"],
    cmap="viridis",
    figsize=(11, 3.5),
    x_reverse=True,
)

# %%
# Calculate the two correlation maps
# ----------------------------------
# The function subtracts the mean spectrum along time to form dynamic spectra
# A. The synchronous map is A.T @ A / (n - 1), a covariance map: a positive
# cross-peak indicates same-direction variations, and a negative one indicates
# opposite-direction variations. Its diagonal contains the spectral variances.
#
# The asynchronous map uses A.T @ N @ A / (n - 1), where N is the discrete
# Hilbert–Noda matrix. It is antisymmetric with a zero diagonal and can reveal
# differences in the evolution of overlapping bands. Proportional variations
# give no asynchronous signal. Reversing the perturbation order reverses its
# sign while leaving the synchronous map unchanged.
#
# Noda's interpretation combines synchronous and asynchronous cross-peak signs
# to discuss the sequence of spectral changes. Such interpretation requires
# consistent axis and perturbation ordering and suitable spectral trajectories;
# an asynchronous peak alone does not establish a reaction mechanism.
# Downsample only the spectral axis, after baseline fitting, to keep the maps
# compact. The temporal grid used for the transform is unchanged.
result = scp.correlation_spectroscopy(corrected[:, ::4], perturbation_dim="y")

# %%
# Contours with reference spectra above
# -------------------------------------
# Each display map is divided by its own maximum absolute intensity. This
# keeps weak asynchronous features visible without suggesting that their
# amplitudes equal synchronous amplitudes. The original result maps retain
# their squared signal units; the panel titles report their absolute maxima.
# Contours are at ±2.5, 5, 10, 20, 35, 50, 65, 80 and 90 percent of each
# maximum. Zero is omitted; red solid contours are positive and blue dashed
# contours are negative. Explicit sign colors keep weak contours legible on
# white; all rendering and panel composition use the scp API.
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

# ``projections=("x",)`` selects the retained spectral dimension. Supplying
# ``projection_data`` puts the corrected reference spectra in its top panel;
# these curves are not sums of the correlation maps. Without external data,
# ``projection_reducer`` selects a sum, mean or maximum over the other axis.
# The plotting API owns alignment, signed contour styles and the complete
# figure lifecycle. Use ``output="correlation.png"`` to save the whole grid.
panels = scp.multiplot(
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

# %%
# Uncomment the following line to display all figures when running the script
# directly with Python.

# %%

# scp.show()
