# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Extended description of NDDataset features

# %%
from spectrochempy import *

# %% [markdown]
# Spectrochempy NDDatasets are homogeneous multidimensionnal objects. 
#
# Below we summarize the main properties, attributes and functions

# %%
# import a dataset
ds = NDDataset.read_omnic(os.path.join('irdata', 'nh4y-activation.spg'))
ds

# %% [markdown]
# ## The basic NDDataset attributes

# %% [markdown]
# * **ndim**: 
#
# the number of dimensions of the array (Read-Only)
#
# * **shape**: 
#
# the size of the dimensions of the array. The length of the shape tuple is therefore the number of dimensions:  ndim (Read-Only).
#
# * **size**:
#
# the total number of elements of the array. This is equal to the product of the elements of shape (Read-Only).
#
# * **dtype**: 
#
# an object describing the type of the elements in the array (Read-Only).
#
# * **itemsize**:
#
# the size in bytes of each element of the array (Read-Only). 
#
# * **dims**:
#
# The list of the name of the dimensions. The length of the dims list is therefore the number of dimensions:  ndim.
#
# * **data**:
#
# the numpy array containing the data. Normally, we won’t need to use this attribute because we will access the elements in an array using indexing facilities.

# %%
ds.ndim, ds.shape, ds.size, ds.dtype, ds.itemsize, ds.dims

# %%
ds.data

# %% [markdown]
# ## Other information on the NDDataset

# %% [markdown]
# * **id**:
#
# A unique Identifier for the dataset
#
# * **name**:
#
# An optional name for the dataset. If not set, the id is returned.
#
# * **origin**:
#
# An optional information about the origin of the data
#
# * **author**:
#
# The optional author(s) information for the dataset
#
# * **title**:
#
# A title describing the kind of data. It is particularly used for plotting.

# %%
ds.author = 'Newton et al.'
ds.id, ds.name, ds.origin, ds.author, ds.title

# %%
ds.title

# %% [markdown]
# ## NDDataset creation

# %% [markdown]
# ## Operations on NDDataset

# %% [markdown]
# ### Finding extrema of a NDDataset
#
# numpy functions **np.max** and **np.min** works on `NDDataset` objects. 

# %% [markdown]
# <div class='alert-info'>
#     
# **Note** : when using the numpy functions on NDDataset, only the `axis` and `keepdims` parameters are available.  
#     
# </div>

# %% [markdown]
# When the dataset is reduced to a single element, the coordinates are lost. A scalar number (or a scalar quantity) is just returned.

# %%
np.max(ds)

# %%
np.min(ds)

# %% [markdown]
# If one wants to keep the coordinates information of the extremum, `keepdims` keyword must be set to True. In this case a ``NDDataset`` object is returned. 

# %%
np.max(ds, keepdims=True)

# %% [markdown]
# Alternatively, on can use the equivalent `NDDataset` methods **max** and **min**:

# %%
ds.max()

# %% [markdown]
# As for numpy array, the indices of the extrema along each dimension can be found using **argmax** or **argmin**. 

# %%
np.argmax(ds)

# %% [markdown]
# To get the corresponding coordinates, there is obviously no available numpy functions, but `NDDataset` possess two methods for this purpose: **coordmax** and **coordmin**. 

# %%
ds.coordmax()

# %% [markdown]
# The `axis` keywords can be sued to find extrema along a dimension (Note that only the `axis` but `dim` keyword can be used for the the numpy functions. 

# %%
np.max(ds, axis=1)

# %% [markdown]
# `dim`can be used for the equivalent methods.

# %%
ds.max(dim='x')

# %%
np.max(ds, axis=1, keepdims=True)

# %% [markdown]
# when using the numpy function, it is important to keep the numpy syntax. 
# An extended syntax can be used while using the fonction as NDDataset methods.
#
# For instance, one can use `ds.max(...)` instead of `np.max(ds, ...)`. In this case one can use the `dim` parameter, which cannot be used with numpy functions.

# %%
ds.max(dim='x')

# %%
ds.max(dim='y', keepdims=True)

# %%
ds.max(dim='y', keepdims=True)

# %%
ds.ptp()

# %% [markdown]
# ### Statistical methods

# %%
ds.mean('x', keepdims=True)

# %%
ds.std('y', keepdims=True)

# %%
ds.cumsum('y')

# %%
