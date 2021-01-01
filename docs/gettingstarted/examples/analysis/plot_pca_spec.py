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
NDDataset PCA analysis example
-------------------------------
In this example, we perform the PCA dimensionality reduction of a spectra
dataset
"""

# %%
import spectrochempy as scp

# %% [markdown]
# Load a dataset

# %%
dataset = scp.read_omnic("irdata/nh4y-activation.spg")
print(dataset)
dataset.plot_stack()

# %% [markdown]
# Create a PCA object
# %%
pca = scp.PCA(dataset, centered=False)

# %% [markdown]
# Reduce the dataset to a lower dimensionality (number of
# components is automatically determined)

# %%
S, LT = pca.reduce(n_pc=.99)

print(LT)

# %% [markdown]
# Finally, display the results graphically
# ScreePlot
# %%
_ = pca.screeplot()

# %% [markdown]
# Score Plot
# %%
_ = pca.scoreplot(1, 2)

# %% [markdown]
# Score Plot for 3 PC's in 3D
# %%
_ = pca.scoreplot(1, 2, 3)

# %% [markdown]
# Displays the 4-first loadings

# %%
LT[:4].plot_stack()

# scp.show()  # uncomment to show plot if needed (not necessary in jupyter notebook)
