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
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""
3D NDDataset creation from scratch
==================================
In this example, we create a 3D NDDataset from scratch,
and then we plot one section (a 2D plane)

"""

# %% [markdown]
# As usual, we start by loading the spectrochempy library
# %%
import spectrochempy as scp

# %% [markdown]
# ## Creation of a 3D NDDataset
#
# Here we will create a 3D NDDataset from scratch
#

# %% [markdown]
# ### Coordinates
#
# The `Coord` object allow making an array of coordinates with additional metadata such as units, labels, title, etc
# %%
coord0 = scp.Coord.linspace(200., 300., 3,  labels=['cold', 'normal', 'hot'], units="K", title='temperature')
coord1 = scp.Coord.linspace(0., 60., 100, labels=None, units="minutes", title='time-on-stream')
coord2 = scp.Coord.linspace(4000., 1000., 100, labels=None, units="cm^-1", title='wavenumber')

coord0.size, coord1.size, coord2.size

# %% [markdown]
# Labels can be useful for instance for indexing
# %%
coord0['normal'].values

# %% [markdown]
# ### Data
#
# Here we use numpy to create a 3D array of data
# (the `data` attribute of the Coord objects are actually numpy's ndarray)

# %%
c0, c1, c2 = coord0.data, coord1.data, coord2.data

import numpy as np
nd_data = np.array([ np.array([ np.sin(2. * np.pi * c2 / 4000.) * np.exp(-y / 60) for y in c1]) * t for t in c0])

nd_data.shape

# %% [markdown]
# ### NDDataset
#
# The `NDDataset` object allow making the array of data with units, etc...
# %%
description= "Dataset example created for this tutorial. It's a 3-D dataset (with dimensionless intensity: absorbance"
mydataset = scp.NDDataset(data=nd_data,
                          coordset=[coord0, coord1, coord2],
                          title="Absorbance",
                          units="absorbance",
                          name = "An example from scratch",
                          author = "Blake and Mortimer",
                          description = description,
                          history='2D dataset creation from scratch')
mydataset

# %% [markdown]
# ### Getting an orthogonal slicing

# %% [markdown]
# We want to plot a section of this 3D NDDataset:
#
# NDDataset can be sliced like conventional numpy-array...
# %%
new = mydataset[..., 0]

# %% [markdown]
# or maybe more conveniently in this case, using an axis labels:
# %%
new = mydataset['hot']

# %% [markdown]
# To plot a dataset, use the `plot` command (generic plot).
# %%
_ = new.plot(method='map', colorbar=True, colormap='magma')

# %% [markdown]
# But it is possible to display image
#
# sphinx_gallery_thumbnail_number = 2
# %%
_ = new.plot(method='image', colormap='magma')

# %% [markdown]
# or stacked plot (which is the default for 2D dataset)
# %%
_ = new.plot(method='stack', colormap='magma') # new.plot has the same effect

# %% [markdown]
# Note that SpectroChemPy allows one to alternatively use the following syntax (using `plot`as an API method):
# %%
_ = scp.plot(new)
