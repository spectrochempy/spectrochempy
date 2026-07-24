# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Choose an explicit plot type for a 2D IR dataset
================================================

This example compares a few explicit plotting methods available for the same
infrared dataset.
"""

import os
from pathlib import Path

import spectrochempy as scp

# %%
# Load and prepare the dataset
# -----------------------------
TEST_FILE = Path(os.environ.get("TEST_FILE", "irdata/nh4y-activation.spg"))
dataset = scp.read(TEST_FILE)
dataset = dataset[:, 4000.0:650.0]
dataset.y -= dataset.y[0]
dataset.y.ito("hour")
dataset.y.title = "Time on stream"
dataset[:, 1290.0:920.0] = scp.MASKED

prefs = scp.preferences
prefs.figure.figsize = (7, 4)

# %%
# Single-spectrum line plot
# --------------------------
single = dataset[0]
_ = single.plot()

# %%
# 2D dataset plots
# -----------------
# Explicit methods make the intended representation clear:
_ = dataset.plot_lines()
_ = dataset.plot_image(colorbar=True)
_ = dataset.plot_contour(colorbar=True)

# %%
# Customize plot options
# -----------------------
_ = dataset.plot_image(
    cmap="plasma",
    xlim=(2000, 1300),
    ylim=(1, 5),
    colorbar=True,
)

# %%
# This ends the example. Uncomment the next line to display the figures when
# running the script directly with Python.

# scp.show()
