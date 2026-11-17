# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Find peaks in an experimental 2D-IR spectrum
============================================

Mark positive peaks and negative bleach minima on the same map.
The D-silicalite spectrum is from `Donaldson, Hawkins and Howe (2025)
<https://doi.org/10.1039/D4SC08093A>`_ and is available from `STFC eData
<https://doi.org/10.5286/edata/938>`_ under `CC BY 4.0
<https://creativecommons.org/licenses/by/4.0/>`_.
See the :doc:`peak finding tutorial </userguide/analysis/peak_finding>`
for data preparation, peak tables, and connected regions.
"""

import spectrochempy as scp

dataset = scp.read("irdata/2dir/silicalite_210C_0ps_2DIR.scp")
distance = {"y": scp.ur("3 cm^-1"), "x": scp.ur("5 cm^-1")}
cross_peaks = dataset.find_peaks(
    dims=("y", "x"), height=0.01, distance=distance, as_result=True
)
bleach_peaks = (-dataset).find_peaks(
    dims=("y", "x"), height=0.02, distance=distance, as_result=True
)

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
