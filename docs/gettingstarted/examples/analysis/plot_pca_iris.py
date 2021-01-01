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

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

# %% [markdown]
"""
PCA analysis example
---------------------
In this example, we perform the PCA dimensionality reduction of the classical
``iris`` dataset.
"""

# %%
import spectrochempy as scp

# %% [markdown]
# Upload a dataset form a distant server

# %%
dataset = scp.download_IRIS()

# %% [markdown]
# Create a PCA object
# %%
pca = scp.PCA(dataset, centered=True)

# %% [markdown]
# Reduce the dataset to a lower dimensionality (number of
# components is automatically determined)

# %%
S, LT = pca.reduce(n_pc='auto')

print(LT)

# %% [markdown]
# Finally, display the results graphically
#
# ScreePlot
# %%
_ = pca.screeplot()

# %% [markdown]
# ScorePlot of 2 PC's
# %%
_ = pca.scoreplot(1, 2, color_mapping='labels')

# %% [markdown]
# or in 3D for 3 PC's
# %%
_ = pca.scoreplot(1, 2, 3, color_mapping='labels')


# scp.show()  # uncomment to show plot if needed (not necessary in jupyter notebook)
