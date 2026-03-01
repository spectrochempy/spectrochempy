# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
Introduction to the plotting librairie
=======================================


"""
# %%

import spectrochempy as scp

# %%
# The location of the spectrochempy_data can be found in preferences

datadir = scp.preferences.datadir

# %%
# Let's read on of the dataset (in `spg` Omnnic format)

dataset = scp.NDDataset.read_omnic(datadir / "irdata" / "nh4y-activation.spg")

# %%
# First we do a generic plot (with the default style):

ax = dataset[0].plot()

# %%
# plot generic style

ax = dataset[0].plot(style="classic")

# %%
# check that style reinit to default
# should be identical to the first one
ax = dataset[0].plot()

# %%
# Multiple plots
dataset = dataset[:, ::100]

datasets = [dataset[0], dataset[10], dataset[20], dataset[50], dataset[53]]
labels = ["sample {}".format(label) for label in ["S1", "S10", "S20", "S50", "S53"]]

_ = scp.plot_multiple(method="scatter", datasets=datasets, labels=labels, legend="best")

# %%
# plot multiple with style
_ = scp.plot_multiple(
    method="scatter", style="sans", datasets=datasets, labels=labels, legend="best"
)

# %%
# check that style reinit to default
_ = scp.plot_multiple(method="scatter", datasets=datasets, labels=labels, legend="best")

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
