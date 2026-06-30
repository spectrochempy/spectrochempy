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
import spectrochempy as scp
from _demo import load_raman_demo_dataset

# %%
# Load the data (here 2D spectrum made from a list of 1D spectra):

B1 = load_raman_demo_dataset()

# %%
# First we show the basic plot. Here `cmap=None` uses categorical rotating
# colors for a line-based plot, and `lw` is the short alias for `linewidth`.
# We also enlarge the figure for readability.
_ = B1.plot(cmap=None, lw=1)

# %%
# We will limit the x range to the region of interest
# note the float number to specify that we use coordinates and not indices
B2 = B1[:, 60.0:]

# %%
# As there is obviously a drift in these spectra, we will use detrend to remove it.
B3 = scp.detrend(B2)
_ = B3.plot(cmap=None)

# %%
# To demonstrate the use of `plot_multiple` we will take only a few spectra.
# For instance the 5 first spectra:
B4 = B3[:5]

# %%
# plot it to see what we have selected
_ = B4.plot(cmap=None)

# %%
# Now use `plot_multiple` to overlay the selected spectra on one axes.  We use
# `method="pen"` for line rendering, set labels for the combined legend, and
# apply `shift` so the traces remain visually separated.
datasets = list(B4)
_ = scp.plot_multiple(
    datasets,
    method="pen",
    legend="best",
    labels=["A", "B", "C", "D", "E"],
    color=["black", "red", "green", "blue", "violet"],
    lw=[1, 2.5, 1, 1, 1],  # line width (we use here different values)
    ls="-",  # solid line style
    shift=1000,  # vertical shift
)

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
