# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Project management

# %%
from spectrochempy import NDDataset, Project
from spectrochempy import general_preferences as prefs

from pathlib import Path

# %% [markdown]
# ## Project creation
# We can easily create a new project to store various datasets

# %%
proj = Project()

# %% [markdown]
# As we did not specify a name, a name has been attributed automatically:

# %%
proj.name

# %% [markdown]
# ------
# To get the signature of the object, one can use the usual '?'. Uncomment the following line to check

# %%
# Project?

# %% [markdown]
# ----
# Let's change this name

# %%
proj.name = 'myNMRdata'
proj

# %% [markdown]
# Now we will add a dataset to the project.
#
# First we read the dataset (here some NMR data) and we give it some name (e.g. 'nmr n°1')

# %%
datadir = Path(prefs.datadir)
path = datadir / 'nmrdata' / 'bruker' / 'tests' / 'nmr'
nd1 = NDDataset.read_topspin(path/'topspin_1d', expno=1, remove_digital_filter=True, name = "NMR_1D")
nd2 = NDDataset.read_topspin(path/'topspin_2d', expno=1, remove_digital_filter=True, name='NMR_2D')

# %% [markdown]
# To add it to the project, we use the `add_dataset` function for a single dataset:

# %%
proj.add_datasets(nd1)

# %% [markdown]
# or `add_datasets` for several datasets.

# %%
proj.add_datasets(nd1, nd2)

# %% [markdown]
# Display its structure

# %%
proj

# %% [markdown]
# It is also possible to add other projects as sub-project (using the `add_project`)

# %% [markdown]
# ## Remove an element from a project

# %%
proj.remove_dataset('NMR_1D')
proj

# %% [markdown]
# ## Get project's elements

# %%
proj.add_datasets(nd1, nd2)
proj

# %% [markdown]
# We can just use the name of the element as a project attribute.

# %%
proj.NMR_1D

# %%
_ = proj.NMR_1D.plot()

# %% [markdown]
# However this work only if the name contains no space, dot, comma, colon, etc. The only special character allowed is the underscore `_`.  If the name is not respecting this, then it is possible to use the following syntax (as a project behave as a dictionary). For example:

# %%
proj['NMR_1D'].data

# %%
proj.NMR_2D

# %% [markdown]
# ## Saving and loading projects

# %%
proj

# %% [markdown]
# #### Saving

# %%
proj.save('NMR')

# %% [markdown]
# #### Loading

# %%
proj2 = Project.load('NMR')

# %%
proj2

# %%
_ = proj2.NMR_1D.plot()

# %%
proj2.NMR_2D  # TODO: something wrong there!

# %%
proj.NMR_2D.plot()

# %%
