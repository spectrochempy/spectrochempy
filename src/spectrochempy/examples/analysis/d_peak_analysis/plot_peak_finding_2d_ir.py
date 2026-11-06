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

import csv

import spectrochempy as scp

# %%
# Download and read the 2D matrix
# --------------------------------
data_file = "irdata/2dir/silicalite_210C_0ps_2DIR.csv"
csv_path = scp.preferences.datadir / data_file
if not csv_path.exists():
    scp.read_csv(data_file, download_only=True)

with csv_path.open(encoding="utf-8", newline="") as stream:
    rows = list(csv.reader(stream))

pump = [float(value) for value in rows[0][1:]]
probe = [float(row[0]) for row in rows[1:]]
signal = [[float(value) for value in row[1:]] for row in rows[1:]]

dataset = scp.NDDataset(
    signal,
    coordset=[
        scp.Coord(probe, title="probe wavenumber", units="cm^-1"),
        scp.Coord(pump, title="pump wavenumber", units="cm^-1"),
    ],
    title="2D-IR difference absorbance",
    units="absorbance",
)

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
