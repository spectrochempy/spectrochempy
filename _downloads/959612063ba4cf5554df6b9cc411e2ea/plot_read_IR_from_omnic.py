# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Loading an IR (omnic SPG) experimental file
===========================================

Here we load an experimental SPG file (OMNIC) and plot it.

"""

# %%
import spectrochempy as scp

# %%
# Load and display the dataset
# -----------------------------
datadir = scp.preferences.datadir
dataset = scp.read_omnic(datadir / "irdata" / "nh4y-activation.spg")
_ = dataset.plot_stack(style="paper")

# %%
# Adjust axis metadata
# ---------------------
dataset.y.to("hour")
dataset.y -= dataset.y[0]
dataset.y.title = "acquisition time"

_ = dataset.plot_stack()

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
