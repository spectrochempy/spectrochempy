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
# # Arrays and NDDatasets

# %% [markdown]
# In this tutorial we will go into the specifics of the numpy library to manipulate single
# and multi-dimensional arrays and we will show how spectroChemPy shares many features with
# the numpy library, as well as new features such as the management of coordinates and units.

# %% [markdown]
# First we will import the libraries we will use in this tutorial.
# It is indeed a common practice to place the library import instructions at the beginning of the scripts.

# %%
import numpy as np
import matplotlib.pyplot as plt

import spectrochempy as scp

# %% [markdown]
# ## Numpy array
#
# Let's create a simple rank-1 (1D) numpy array

# %% [markdown]
# ***Note*** *that in a notebook, if one wants to quickly transform a code cell (the default at the cell creation)
#
# to a markdown cell (as this one) we can use the key's combination:* **\<ESC\>+m**
#
#

# %%
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Create a rank 1 array
a

# %% [markdown]
# A frequent error happens for beginers: calling the np.array method with a list of arguments,
# instead of a list (or tuple).

# %%
# a = np.array(1,2,3,4)    # This is WRONG !  it must be np.array((1,2,3,4)) or np.array([1,2,3,4]) !!!

# %% [markdown]
# Now we can get some information about this newly created array

# %% [markdown]
# For example, its type:

# %%
type(a)  # Prints the type of data

# %% [markdown]
# which can be used in some test, as in the following kind of test.

# %%
isinstance(a, np.ndarray)

# %% [markdown]
# ## Numpy array attributes
#
# Another important feature of arrays is that they have several "builtin" attributes:

# %%
a.shape  # the shape is a tuple with the size of all dimensions

# %%
a.size  # the size is the total number of elements

# %%
a.dtype  # the dtype

# %% [markdown]
# The data type (dtype) of the numpy array was automatically set at the array creation from the type of the elements.
# It is important to notice that a numpy array is an homogeneous data object, which means that
# all emements it contains have the same type.
#
# If you need to manage arrays with non-homogeneous type of elements, you should have a look at the excellent library
# [pandas](https://pandas.pydata.org).
#
# In this case for instance, a number such as *5* is an integer, so the array has the dtype `int64` or  `ìnt32`
# dependending of you operating system. To have for instance real number whe must write *5.*:

# %%
type(5), type(5.)  # prints "(int, float)"

# %% [markdown]
# Note that we can change the dtype easily using the method `astype`:

# %%
a = a.astype('float64')
a.dtype

# %%
a.dtype

# %% [markdown]
# But you can also force your data to be in a given type at the  creation step

# %%
b = np.array([[1, 2], [2, 1]], dtype='complex128')  # create a rank-2 array of type complex128
b.dtype

# %%
b.real.dtype, b.imag.dtype  # the real and imaginary part are of type float64

# %% [markdown]
# ## NDDataset
#
# SpectroChemPy NDDataset (dataset) are constructed from numpy arrays. they are also homogeneous objets.
#
# Let's show this.

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
# We can use the python assert method to check that the two object are indeed related (same attributes)

# %%
assert a.size == nd.size  # assert return an AssertionError if the condition is False
assert a.shape == nd.shape
assert a.dtype == nd.dtype
assert np.all(a == nd.data)  # here we use np.all to allow comparison between two arrays

# %% [markdown]
# Some of the other dataset attributes are displayed when the dataset is displayed:

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
# There are many ways to create numpy arrays - it is worth to know some of them.

# %%
b = np.zeros((2, 2))  # Create an array of all zeros
b

# %%
c = np.ones((1, 2))  # Create an array of all ones
c

# %%
d = np.full((2, 2, 3), 7)  # Create an array filled with a given value
d

# %%
e = np.eye(3)  # Create a 3x3 identity matrix
e

# %%
f = np.random.random((2, 2))  # Create an array filled with random values
f

# %%
np.arange(1, 21, 3)  # create a rank-1 numpy array with successive elements
# from 1 to 20, by step of 3 (21 not being ingluded)

# %%
np.linspace(-5, 5, 50)  # create un array with 50 values regularly spaced betwwen -5 and 5

# %% [markdown]
# The same is possible with dataset for some of the above functions (zeros, ones, full, eye (in 0.1.18-dev)

# %%
scp.full((2, 3, 4), 100.)

# %%
# WARNING: scp.eyes and scp.random not yet implemented (will be done for v.0.1.18)

# %% [markdown]
# One interesting trick you can use when creating arrays is `reshaping`. Let's take an example

# %%
a = np.arange(8)  # the method arange create an rank-1 array with the an integer series of 8 element in the present case
a

# %%
a = a.reshape((2, 4))  # transform the array with shape (8,)  to a rank-2 array with shape (2,4)
a

# %%
a = a.reshape((4, 2))  # transform the array with shape (2,4)  to a rank-2 array with shape (4,2)
a

# %%
a = a.reshape((2, 2, 2))  # transform the array with shape (4,2)  to a rank-3 array with shape (2,2,2)
a

# %% [markdown]
# and now we can use the array for creating a new 3D-dataset

# %%
nd3 = scp.NDDataset(a)
nd3

# %%
# warning: nd3.reshape(8) DOES NOT WORK ! # for now no similar function as reshape has been implemented in SpectroChemPy

# (it could be in future roadmap)

# %% [markdown]
# ## Creating an array from files or from functions

# %% [markdown]
# ### Import data from files
# Suppose you want to plot a nice figure with the results of experiments - It is possible to do this purely
#
# using numpy and matplotlib libraries, but we suggest to use the tools included in SpectroChemPy for this task.
#
#
# First, some spectroscopic data import functions are defined in SpectroChemPy and can be used conveniently.
#
# See the [Import-export userguide](https://www.spectrochempy.fr/dev/user/userguide/dataset/4_importexport.html).
# **WARNING:** *it is under work*
#
# Here I will just give an example, using some data from OMNIC.

# %%
path = scp.general_preferences.datadir  # this is a path provided that point on a example set of data.
# It can be replaced by any other path you like

# %% hidden="true" nbsphinx="hidden"
# THESE TWO LINES ARE JUST HERE FOR BUILDING THE DOCUMENTATION AND TESTING
# ITHEY SHOULD BE COMMENTED FOR A INTERACTIVE USE OF THIS TUTORIAL)
sop = scp.os.path
path = sop.join(path, 'irdata')

# %%
fs = scp.FileSelector(path=path, filters=['spg', 'spa'])
fs

# %% nbsphinx="hidden"
# THESE THREE LINES ARE JUST HERE FOR BUILDING THE DOCUMENTATION AND TESTING
# THEY SHOULD BE COMMENTED FOR AN INTERACTIVE USE OF THIS TUTORIAL
fs.fullpath = sop.join(path,'CO@Mo_Al2O3.SPG')
fs.value = 'CO@Mo_Al2O3.SPG'

# %%
fs.fullpath  # retrieve the path selected in the previous step

# %%
data = scp.read(fs.fullpath)  # now import it
data

# %%
ax = data.plot()  # now plot it
_ = ax.set_title(fs.value)  # add a title


# %% [markdown]
# ### Create array from functions and plot it
#
# Let's assume we have some model giving a series of data. For example, we have a simple model
# to generate a NMR free induction decay.
#
# We can create specially dedicated function using the python block struture `def function(): ....`.
#
# But a simple and fast methods is to use `lambda` function.

# %%
def fid(t, w, T2): return (np.cos(2. * np.pi * w * t) + 1j * np.sin(2. * np.pi * w * t)) * np.exp(-t / T2)
# in this model function we generate a sinusoidal time (t) evolution with an angular frequency w, and a relaxation T2.


# %%
# example
t = np.linspace(0, 100., 1000.)  # t in second
y = fid(t, 200., 10.)  # we get a complex array   w in MHz, T2 in s
y.dtype

# %%
# plot it
plt.plot(t, y)

# %% [markdown]
# We can use Spectrochempy to create an dataset:

# %%
timecoord = scp.Coord(t, title='time', units='second')
ndfid = scp.NDDataset(y, dims=['x'], coords=[timecoord], title='signal')
ndfid.plot()

# %% [markdown]
# ## Basic operations on arrays and NDDatasets
#
# An essential property of numpy arrays is the ability to perform mathematical operations on and between them.

# %%
a = np.full((5, 3), 10.)  # A 2D array with 5 rows and 3 columns
b = np.array([[1, 2, 3, 4, 5]])  # A 2D array with & row and 5 columns

# %% [markdown]
# Clearly, the addition of a and b which is perfomed elementwise can not be used in this cas because
# the shapes mismatch.

# %% [markdown]
# the addition of the two array `a+b` gives an error :
#
#
#     ValueError: operands could not be broadcast together with shapes (5,2) (1,5)
#
#
#
# because the shape mismacth.
#
#
# However, there is one case where such operation can be perfomred, this is when the array have shape differing only
#
#

# %%
