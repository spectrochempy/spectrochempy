# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Slice an NDDataset with indices and coordinates
===============================================

This example shows how to combine standard Python slicing with
coordinate-aware slicing on an infrared time series.
"""

import spectrochempy as scp

# %%
# Load an IR dataset and express the acquisition axis in minutes.

dataset = scp.read_omnic(
    "irdata/CO@Mo_Al2O3.SPG",
    description="CO adsorption, difference spectra",
)
dataset.y = (dataset.y - dataset[0].y).to("minute")
dataset

# %%
# Plot the full dataset once for context.

prefs = scp.preferences
prefs.figure.figsize = (7, 4)
_ = dataset.plot()

# %%
# Standard integer slices work as expected on both dimensions.

first_four = dataset[:4]
every_other_point = dataset[:, ::2]

print(first_four.shape)
print(every_other_point.shape)

# %%
# Coordinate-aware slicing is often more convenient for spectroscopy work.
# Using floats slices directly on axis coordinates instead of integer indices.

carbonyl_region = dataset[:, 2300.0:1900.0]
_ = carbonyl_region.plot()

# %%
# The same applies to the time axis.

window = dataset[80.0:180.0, 2300.0:1900.0]
_ = window.plot()

# %%
# A single float selects the closest spectrum on that axis.

selected = dataset[60.0]
selected.y

# %%
# This ends the example. Uncomment the next line to display the figures when
# running the script directly with Python.

# scp.show()
