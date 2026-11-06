# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Find peaks in an experimental 2D-IR spectrum
============================================

This example loads a pump-probe 2D-IR spectrum, finds positive cross-peaks and
negative diagonal bleach peaks, and locates them on a two-dimensional map.

The experimental data describe D-silicalite at 210 °C, measured with parallel
polarization at a waiting time of 0 ps. They accompany Donaldson, Hawkins and
Howe, *Chemical Science* (2025), `DOI: 10.1039/D4SC08093A
<https://doi.org/10.1039/D4SC08093A>`_, and originate from `STFC eData
<https://doi.org/10.5286/edata/938>`_ under the `CC BY 4.0 license
<https://creativecommons.org/licenses/by/4.0/>`_.

The native ``.scp`` file preserves the cropped CSV matrix, its pump and probe
coordinates, and difference-absorbance units. Preparation retained the 128
parallel-detector probe rows and the pump range 2176–2831 cm⁻¹, removing the
mirrored half. No baseline correction, smoothing or normalization was applied.
The diagonal bleach is negative. See the `dataset README
<https://github.com/spectrochempy/spectrochempy_data/blob/master/testdata/irdata/2dir/README.md>`_
for the original filename and preparation details.
"""

# sphinx_gallery_thumbnail_number = 2

import spectrochempy as scp

# %%
# Download and read the 2D matrix
# --------------------------------
dataset = scp.read("irdata/2dir/silicalite_210C_0ps_2DIR.scp")
_ = dataset.plot(method="lines", figsize=(7, 5), colorbar=True)

# %%
# Detect positive cross-peaks and negative diagonal bleach
# ---------------------------------------------------------
distance = {"y": scp.ur("3 cm^-1"), "x": scp.ur("5 cm^-1")}
cross_peaks = dataset.find_peaks(
    dims=("y", "x"), height=0.01, distance=distance, as_result=True
)
bleach_peaks = (-dataset).find_peaks(
    dims=("y", "x"), height=0.02, distance=distance, as_result=True
)

cross_peaks.table.top(5, by="height")

# %%
# Peak coordinates are reported in the dataset's probe (y) and pump (x) axes.
# Bleach heights are positive amplitudes measured on the negated spectrum.
bleach_peaks.table.top(5, by="height")

# %%
# Locate the detected peaks on the 2D-IR map
# ------------------------------------------
ax = dataset.plot_image(figsize=(7, 5), cmap="RdBu_r", colorbar=True)
for result, marker, color, label in (
    (cross_peaks, "o", "black", "Positive peaks"),
    (bleach_peaks, "s", "limegreen", "Negative bleach peaks"),
):
    rows = result.to_dict()
    _ = ax.scatter(
        [row["x_position"].magnitude for row in rows],
        [row["y_position"].magnitude for row in rows],
        marker=marker,
        s=65,
        facecolors="none",
        edgecolors=color,
        linewidths=1.5,
        label=label,
    )
_ = ax.legend()

# %%
# Uncomment the following line to display all figures when running the script
# directly with Python.

# %%

# scp.show()
