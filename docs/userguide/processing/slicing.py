# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Slicing datasets
#
# Here we shows how to handle NDDatasets using python slicing.


# %%
import numpy as np

import spectrochempy as scp

# %% [markdown]
# ## What is the slicing ?
#
# The slicing of a list or an array means taking elements from a given index (or set of indexes) to another index (or
# set of indexes). Slicing is specified using the colon operator `:` with a `from` and `to` index before and after
# the first column, and a `step` after the second column. Hence a slice of the object `X` will be set as:
#
# `X[from:to:step]`
#
# and will extend from the ‘from’ index, ends one item before the ‘to’ index and with an increment of `step`between
# each index. When not given the default values are respectively 0 (i.e. starts at the 1st index), length in the
# dimension (stops at the last index), and 1.
#
# Let's first illustrate the concept on a 1D example:

# %%
X = np.arange(10)  # generates a 1D array of 10 elements from 0 to 9
print(X)
print(X[2:5])  # selects all elements from 2 to 4
print(X[::2])  # selects one out of two elements
print(X[:-3])  # a negative index will be counted from the end of the array
print(X[::-2])  # a negative step will slice backward, starting from 'to', ending at 'from'

# %% [markdown]
# The same applies to multidimensional arrays by indicating slices separated by commas:

# %%
X = np.random.rand(10, 10)  # genarates a 10x10 array filled with random values
print(X.shape)
print(X[2:5, :].shape)  # slices along the 1st dimension, X[2:5,] is equivalent
print(X[2:5, ::2].shape)  # same slice along 1st dimension and takes one 1 column out of two along the second

# %% [markdown]
# ## Slicing of NDDataset
#
# NDDataset can be sliced like conventional numpy-array with some additional features
#
# *e.g.,*:
#
# 1. by index, using a slice such as [3], [0:10], [:, 3:4], [..., 5:10], ...
#
# 2. by values (coordinates), using a slice such as [3000.0:3500.0], [..., 300.0], ...
#
# 3. by labels, using a slice such as ['monday':'friday'], ...
#
# Let's import a group of IR spectra, look at its content and plot it:

# %%
X = scp.read_omnic('irdata/CO@Mo_Al2O3.SPG', description='CO adsorption, diff spectra')
X.y -= X.y[0]
X.y.ito("minute")
X.y.title = 'Time'
X

# %%
prefs = X.preferences  # set up general preferences for plotting
prefs.colormap = 'Dark2'
prefs.figure.figsize = (7, 3)

# %%
_ = X.plot()  # assignment to _ avoids the display of the text ouput from plot. We want only the plot

# %% [markdown]
# ### Slicing with indexes
#
# The classical slicing, using integers, can be used. For instance, along the 1st dimension:

# %%
print(X[:4])  # selects the first four spectra
print(X[-3:])  # selects the last three spectra
print(X[::2])  # selects one spectrum out of 2

# %% [markdown]
# The same can be made along the second dimension, simultanesly or not with the first one. For instance

# %%
print(X[:, ::2])  # all spectra, one wavenumber out of 2   (note the bug: X[,::2] generates an error)
print(X[0:3, 200:1000:2])  # 3 first spectra, one wavenumbers out of 2, frm index 200 to 1000

# %% Would you easily guess which wavenumber range have been actually selected ?.... probably not because [markdown]
# the relationship between the index and the wavenumber is not straightforward as it depends on the the value of the
# first wavenumber, the wavenumber spacing, and whether the wavenumbers are arranged in ascending or descending
# order... Here is the answer:

# %%
X[:, 200:1000:2].x  # as the Coord can be sliced, the same is obtained with: X.x[200:1000:2]

# %% [markdown]
# ### Slicing with coordinates
#
# Now the spectroscopist is generally interested in a particular region of the spectrum, for instance,
# 2300-1900 cm$^{-1}$. Can you easily guess the indexes that one should use to spectrum this region ? probably not
# without a calculator! ...
#
# Fortunately, a simple mechanism has been implemented in SpectroChemPy for this purpose: the use of floats instead
# of integers will slice the NDDataset at the corresponding coordinates. For instance to select the 2300-1900 cm$^{
# -1}$ region:

# %%
subplot = X[:, 2300.0:1900.0:].plot()

# %% [markdown]
# The same mechanism can be used along the first dimension (`y`). For instance, to select and plot the same region and
# the spectra recorded between 80 and 180 minutes:

# %%
# Note that a decimal point is enough to get a float
# and that a warning is raised if one or several values are beyond the limits
subplot = X[80.:180., 2300.:1900.].plot()
# %% [markdown]
# Similarly, the spectrum recorded at the time the closest to 60 mins can be selected using a float:

# %%
X[60.].y  # X[60.] slices the spectrum,  .y returns the corresponding `y` axis.

# %% [markdown]
# ### Slicing with labels

# %% [markdown]
# The X dataset have several lists of labels in the y dimension. Let's display the first series (which is compose of
# dtetime objects)

# %%
for i, item in enumerate(X.y.labels[:, 0]):
    print(i, '->', item, type(item))

# %% [markdown]
# Let's slice using the date at index 10. We can check that the slicing is correct

# %%
from datetime import datetime, timezone

date1 = datetime(2016, 10, 18, 15, 24, 4, tzinfo=timezone.utc)
X[date1] == X[10]

# %% [markdown]
# One can mixed slicing methods for different dimension:

# %%
d = X[date1, 2000.:2300.]
_ = d.plot()

# %% [markdown]
# We can also take a range of labels

# %%
date2 = datetime(2016, 10, 18, 15, 52, 3, tzinfo=timezone.utc)
d = X[date1:date2, 2000.:2300.]
_ = d.plot()

# %% [markdown]
# ### Slicing dataset with multiple coordinates on one dimension

# %% [markdown]
# <div class='alert alert-warning'>
# <b>Todo</b>
#
#    Slicing dataset with multiple coordinates on one dimension
#
#    Add advanced slicing by array of indexes, array of bool,  ...
# </div>
