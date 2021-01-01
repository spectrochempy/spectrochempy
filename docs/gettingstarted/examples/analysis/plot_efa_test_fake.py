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
EFA analysis example
======================

In this example, we perform the Evolving Factor Analysis
"""
# %%
import numpy as np

import spectrochempy as scp

# %% [markdown]
# Upload and preprocess a dataset

# %%
dataset = scp.read("irdata/nh4y-activation.spg")

# columns masking
# dataset[:, 1230.0:920.0] = scp.masked  # do not forget to use float in slicing
# dataset[:, 5997.0:5993.0] = scp.masked

# row masking (just for an example
# dataset[10:16] = scp.masked

dataset.plot_stack()

# %% [markdown]
#  Evolving Factor Analysis

# %%
efa = scp.EFA(dataset)

f = efa.cut_f()
f.plot()
b = efa.cut_b()

f.T.plot(yscale="log", labels=f.y.labels, legend='best')
b.T.plot(yscale="log")

# %% [markdown]
# Clearly we can retain 4 components, in agreement with what was used to
# generate the data - we set the cutof of the 5th components
#

# %%
npc = 4
cut = np.max(f[:, npc].data)

f = efa.cut_f(cutoff=cut)
b = efa.cut_b(cutoff=cut)
# we concatenate the datasets to plot them in a single figure
both = scp.concatenate(f, b)
both.T.plot(yscale="log")

c = efa.get_conc(npc, cutoff=cut)
c.T.plot()

# scp.show()  # uncomment to show plot if needed (not necessary in jupyter notebook)
