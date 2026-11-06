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
negative diagonal bleach peaks, and inspects their coordinates.
"""

import spectrochempy as scp

# %%
# Download and read the 2D matrix
# --------------------------------
dataset = scp.read("irdata/2dir/silicalite_210C_0ps_2DIR.scp")

prefs = scp.preferences
prefs.figure.figsize = (7, 5)
prefs.colormap = "RdBu_r"
prefs.colorbar = True
_ = dataset.plot()

# %%
# Detect positive cross-peaks and negative diagonal bleach
# ---------------------------------------------------------
distance = {"y": 3 * scp.ur.cm**-1, "x": 5 * scp.ur.cm**-1}
cross_peaks = dataset.find_peaks(
    dims=("y", "x"), height=0.01, distance=distance, as_result=True
)
bleach_peaks = (-dataset).find_peaks(
    dims=("y", "x"), height=0.02, distance=distance, as_result=True
)

sorted(cross_peaks.to_dict(), key=lambda row: -row["height"].magnitude)[:5]

# %%
# Peak coordinates are reported in the dataset's probe (y) and pump (x) axes.
sorted(bleach_peaks.to_dict(), key=lambda row: -row["height"].magnitude)[:5]
