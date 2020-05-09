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
# # Arrays and NDDataset

# %% [markdown]
# Before diving in the SpectroChemPy tutorial, we must remind some of the basis of Numpy from which the NDDataset object is derived.

# %% [markdown]
# First we need to import the library

# %%
import numpy as np

# %% [markdown]
# ## Numpy array
#
# Let's create a simple rand 1 (1D) array

# %% [markdown]
# *Note that in a notebook, if one wants to quickly transform a code cell (the default at the cell creation) 
# to a markdown cell (as this one) we can use the key's combination: **\<ESC\>+m** *
#  

# %%
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])   # Create a rank 1 array
a

# %% [markdown]
# Now we can get some information about this newly created array

# %% [markdown]
# For example, its type:

# %%
type(a)          # Prints the type of data

# %% [markdown]
# which can be used in some test, as in the following. 

# %%
isinstance(a, np.ndarray)

# %% [markdown]
# ## Numpy array attributes
#
# Another important feature of arrays is that they have several builtin attributes:

# %%
a.shape            #  the shape is a tuple with the size of all dimensions

# %%
a.size             #  the size is the total number of elements

# %%
a.dtype            # the type (dtype= data type) of the elements (note that a numpy array is homogeneous 
                   # all emements have the same type 

# %% [markdown]
# note that we can change the dtype:

# %%
a = a.astype('float64')
a

# %%
a.dtype

# %% [markdown]
# ## NDDataset
#
# SpectroChemPy NDDataset (dataset) can be constructed from numpy arrays.
#
# Let's show this.
#
# But before we need to import the library : 

# %%
 import spectrochempy as scp

# %%
nd = scp.NDDataset(a)

# %%
type(nd)

# %% [markdown]
# This dataset has the same attributes as a numpy array and some others in addition.

# %%
nd.shape

# %%
nd.size

# %%
nd.dtype

# %% [markdown]
# The underlying array can be retrieved from the `data` attribute:

# %%
nd.data

# %% [markdown]
# Some of the other attributes are displayed when the dataset is displayed:

# %%
nd

# %% [markdown]
# The main public attributes can be found using the function `dir`

# %%
dir(nd)

# %% [markdown]
# Some of them will be used later in this tutorials

# %% [markdown]
# ## Array creation
#
# There many ways to create numpy arrays - it is worth to know some of them.

# %%
b = np.zeros((2,2))   # Create an array of all zeros
b

# %%
c = np.ones((1,2))    # Create an array of all ones
c

# %%
d = np.full((2,2,3), 7)  # Create an array filled with a given value
d

# %%
e = np.eye(3)         # Create a 2x2 identity matrix
e

# %%
f = np.random.random((2,2))  # Create an array filled with random values
f

# %%
scp.random

# %% [markdown]
# ## Slicing

# %%
print(a[0], a[1], a[2])   # Prints "1 2 3"
a[0] = 5                  # Change an element of the array
print(a)                  # Prints "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(b.shape)                     # Prints "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])   # Prints "1 2 4"

# %%

# %%
