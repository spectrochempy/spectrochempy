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
EFA analysis example
======================

In this example, we perform the Evolving Factor Analysis
"""
# %%
import os

import spectrochempy as scp

# sphinx_gallery_thumbnail_number = 2

# %% [markdown]
# Upload and preprocess a dataset

# %%
datadir = scp.preferences.datadir
dataset = scp.read_omnic(os.path.join(datadir, 'irdata',
                                      'nh4y-activation.spg'))

# %% [markdown]
# columns masking

# %%
dataset[:, 1230.0:920.0] = scp.MASKED  # do not forget to use float in slicing
dataset[:, 5997.0:5993.0] = scp.MASKED

# %% [markdown]
# difference spectra

# %%
dataset -= dataset[-1]
dataset.plot_stack()  # figure 1

# %% [markdown]
# column masking for bad columns

# %%
dataset[10:12] = scp.MASKED

# %% [markdown]
#  Evolving Factor Analysis

# %%
efa = scp.EFA(dataset)

# %% [markdown]
# Show results

# %%
npc = 4
c = efa.get_conc(npc)
c.T.plot()

# scp.show()  # uncomment to show plot if needed (not necessary in jupyter notebook)
