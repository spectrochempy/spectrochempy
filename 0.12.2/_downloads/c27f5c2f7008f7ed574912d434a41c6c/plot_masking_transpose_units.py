# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Mask a saturated region and transform an IR dataset
==================================================

This example shows three common operations on a 2D infrared dataset:
masking a saturated region, transposing the dataset, and converting axis units.
"""

import spectrochempy as scp

# %%
# Load and inspect the dataset
# -----------------------------
dataset = scp.read_omnic("irdata/nh4y-activation.spg")
dataset.y -= dataset.y[0]
dataset.y.title = "time"

prefs = scp.preferences
prefs.figure.figsize = (7, 3.5)
prefs.colormap = "Dark2"
prefs.colorbar = True
_ = dataset.plot()

# %%
# Mask a saturated region
# ------------------------
dataset[:, 1290.0:890.0] = scp.MASKED
_ = dataset.plot_stack()

# %%
# The mask is respected by subsequent operations such as reductions:
dataset.max()

# %%
# Transpose the dataset
# ----------------------
# Transposition exchanges axes while preserving data and mask:
transposed = dataset.T
_ = transposed.plot()

# %%
# Convert coordinate units
# -------------------------
dataset.y.ito("hours")
_ = dataset.plot()

# %%
# This ends the example. Uncomment the next line to display the figures when
# running the script directly with Python.

# scp.show()
