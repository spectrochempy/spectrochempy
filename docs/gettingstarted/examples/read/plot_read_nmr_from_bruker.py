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
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

# %% [markdown]
"""
Loading of experimental 1D NMR data
===================================

In this example, we load a NMR dataset (in the Bruker format) and plot it.
"""

# %%
import spectrochempy as scp
import os

# %% [markdown]
# `datadir.path` contains the path to a default data directory.

# %%
datadir = scp.preferences.datadir

path = os.path.join(datadir, 'nmrdata', 'bruker', 'tests', 'nmr', 'topspin_1d')

# %% [markdown]
# load the data in a new dataset

# %%
ndd = scp.NDDataset.read_topspin(path, expno=1, remove_digital_filter=True)

# %% [markdown]
# view it...

# %%
scp.plot(ndd, style='paper')

# scp.show()  # uncomment to show plot if needed (not necessary in jupyter notebook)
