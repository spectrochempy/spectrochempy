# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
# ---

# %%

# %%

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

# %% [markdown]
"""
Introduction to the plotting librairie
===========================================
"""
# %%
import spectrochempy as scp
import os

# this also import the os namespace

# sp.set_loglevel('DEBUG')
datadir = scp.preferences.datadir
dataset = scp.NDDataset.read_omnic(
    os.path.join(datadir, 'irdata', 'nh4y-activation.spg'))

# %% [markdown]
# plot generic
# %%
ax = dataset[0].plot(color='blue')
ax = dataset[0].plot_pen(color='red')
ax = dataset[0].plot_scatter(mfc='red')

# %% [markdown]
# plot generic style
# %%
ax = dataset[0].plot(style='classic')

# %% [markdown]
# check that style reinit to default
# should be identical to the first
# %%
ax = dataset[0].plot()

# %% [markdown]
# Multiple plots
# %%
dataset = dataset[:, ::100]

datasets = [dataset[0], dataset[10], dataset[20], dataset[50], dataset[53]]
labels = ['sample {}'.format(label) for label in
          ["S1", "S10", "S20", "S50", "S53"]]

scp.plot_multiple(method='scatter',
                  datasets=datasets, labels=labels, legend='best')

# %% [markdown]
# plot mupltiple with style
# %%
scp.plot_multiple(method='scatter', style='sans',
                  datasets=datasets, labels=labels, legend='best')

# %% [markdown]
# check that style reinit to default
# %%
scp.plot_multiple(method='scatter',
                  datasets=datasets, labels=labels, legend='best')


scp.plot(dataset)
scp.show()  # uncomment to show plot if needed (not necessary in jupyter notebook)
