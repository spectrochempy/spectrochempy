# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Using `plot_multiple` to plot several datasets on the same figure
=================================================================
In this example, we will display several Raman datasets on the same figure
using the `plot_multiple` helper. Unlike `multiplot`, which creates a grid of
panels, `plot_multiple` overlays several datasets on one shared axes. Several
options are available to customize that overlay.
"""

# %%
# Import spectrochempy as usual
import os
from pathlib import Path

import spectrochempy as scp

# %%
# Load and inspect the data
# --------------------------
TEST_FILE = Path(os.environ.get("TEST_FILE", "ramandata/labspec/serie190214-1.txt"))
B1 = scp.read(TEST_FILE)

# %%
# Basic plot with categorical colors (``cmap=None``):
_ = B1.plot(cmap=None, lw=1)

# %%
# Preprocess: restrict and detrend
# ---------------------------------
# Restrict the region of interest (use floats for coordinate slicing):
B2 = B1[:, 60.0:]

# %%
# Remove the drift with ``detrend``:
B3 = scp.detrend(B2)
_ = B3.plot(cmap=None)

# %%
# Select a subset of spectra for the overlay:
B4 = B3[:5]

# %%
_ = B4.plot(cmap=None)

# %%
# Overlay spectra with ``plot_multiple``
# ---------------------------------------
# Unlike ``multiplot`` (which creates a grid of panels), ``plot_multiple``
# overlays several 1D datasets on a shared axes.  Here we use line rendering
# with a vertical shift for visual separation:
datasets = list(B4)
_ = scp.plot_multiple(
    datasets,
    method="pen",
    legend="best",
    labels=["A", "B", "C", "D", "E"],
    color=["black", "red", "green", "blue", "violet"],
    lw=[1, 2.5, 1, 1, 1],
    ls="-",
    shift=1000,
)

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
