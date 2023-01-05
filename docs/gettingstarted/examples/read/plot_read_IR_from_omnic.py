# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
Loading an IR (omnic SPG) experimental file
============================================

Here we load an experimental SPG file (OMNIC) and plot it.

"""

import spectrochempy as scp

# %%
# Loading and stacked plot of the original

datadir = scp.preferences.datadir

dataset = scp.NDDataset.read_omnic(datadir / "irdata" / "nh4y-activation.spg")

dataset.plot_stack(style="paper")

# %%
# change the unit of y-axis, the y origin as well as the title of the axis

dataset.y.to("hour")
dataset.y -= dataset.y[0]
dataset.y.title = "acquisition time"

dataset.plot_stack()

# scp.show()  # uncomment to show plot if needed (not necessary in jupyter notebook)
