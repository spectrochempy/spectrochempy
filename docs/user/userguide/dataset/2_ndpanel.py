# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The NDPanel object

# %%
from spectrochempy import *

# %% [markdown]
# `NDPanel` objects are very similar to `NDDataset` in the sense they can contain array and coordinates.
#
# However unlike `NDDataset`s, `NDPanel`s can contain several arrays whith different shapes, units and/or coordinates. They can store heterogeneous data coming for example from different types of experiments. Arrays present in `NDPanel` can be aligned during objects initialization.   

# %% [markdown]
# ## Creating a NDPanel object

# %%
NDPanel()

# %% [markdown]
# Above we have created an empty panel. To create a more interesting panels, we need to add some datasets.
#
# The most straightforward way is probably to do it at the creation of the ``NDPanel`` object using the following syntax
#
#     ndp = NDPanel(a, b, c ...)
#     
# where `a`, `b`, `c` are ``NDDataset``s or can be casted to ``NDDataset``

# %% [markdown]
# For sake of demonstration, let's take an example.

# %%
# create a first random array
a = np.random.rand(6,8) 
# make to coordinate's arrays for both dimensions
cx = Coord(np.linspace(600,4000,8), units='cm^-1', title='wavenumber')
cy = Coord(np.linspace(0,10,6), units='s', title='time')
# create the dataset
nda = NDDataset(a, coords=(cy, cx), name='a', title='dataset a', units='eV')
nda

# %%
# create a second dataset
b = np.random.rand(10,8) 
cz = Coord(np.linspace(600,4000,8), units='cm^-1', title='wabenumber')
cu = Coord(np.linspace(0,10,10), units='s', title='time')
ndb = NDDataset(b, coords=(cu, cz), name='b', title='dataset b', units='eV')
ndb

# %% [markdown]
# Now we will create a NDPanel using these two datasets

# %%
ndp = NDPanel(nda, ndb)
ndp

# %%
ndp.dims

# %%
ndp.coords

# %% [markdown]
# Why dimension `y` is different from those of `nda` and `ndb`?
# because by default dimensions are merged and aligned (using the 'outer' method)
#
# If we want to avoid this behavior, we need to specify in the arguments:
#
# * **merge** : True or False
#
# and/or
#
# * **align** : None, 'outer', 'inner', 'first' or 'last' 
#

# %% [markdown]
# Examples

# %%
# no merging of the dimensions (4 distinct dimensions)
ndp = NDPanel(nda, ndb, merge=False)  
ndp.dims

# %%
# merging of the dimensions, but no alignment of the coordinates (dimensions x for both dataset
# have the same coordinates so they are merged)
ndp = NDPanel(nda, ndb, merge=True, align=None)  
ndp.dims

# %%
# the default behavior
ndp = NDPanel(nda, ndb, merge=True, align='outer')  
ndp.dims

# %%


