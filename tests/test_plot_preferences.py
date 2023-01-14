# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa

import spectrochempy as scp


def test_plotpreferences():
    dataset = scp.NDDataset.read("irdata/nh4y-activation.spg")

    prefs = dataset.preferences  # we will use prefs instead of dataset.preference
    prefs.figure.figsize = (6, 3)  # The default figsize is (6.8,4.4)
    prefs.colorbar = True  # This add a color bar on a side
    prefs.colormap = "magma"  # The default colormap is viridis
    prefs.axes.facecolor = ".95"  # Make the graph background colored in a light gray
    prefs.axes.grid = True
    print(f"font before reset: {prefs.font.family}")
    prefs.reset()
    print(f"font after reset: {prefs.font.family}")
    prefs.style = "grayscale"
    prefs.style = "ggplot"
    prefs.reset()
    prefs.style = "grayscale", "paper"

    prefs.colormap = "magma"

    prefs.available_styles

    prefs.makestyle("scpy")
    prefs.makestyle()

    # %% [markdown]
    # **Example:**
    #

    # %%
    prefs.reset()
    prefs.colorbar = True
    prefs.colormap = "jet"
    prefs.font.family = "monospace"
    prefs.font.size = 14
    prefs.axes.labelcolor = "blue"
    prefs.axes.grid = True
    prefs.axes.grid_axis = "x"
