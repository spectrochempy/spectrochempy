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

# %% [reduction] [markdown]
# # Introduction

# %% [reduction] [markdown]
# Here we will quickly present the SpectroChemPy library for manipulating multidimensional arrays with features such as
# coordinate and unit management.

# %% [markdown]
# We will first import the SpectroChemPy library and for convenience some particular object of the library

# %%
import spectrochempy as scp

# %% [reduction] [markdown]
# In many aspects the manipulation of the main object of SpectroChemPy is very close to `mumpy`, a popular library in
# the python world for scientific computation .
#
# Actually, the Spectrochempy library is built from numpy arrays. And therefore it supports many methods similar to
# numpy.
#
# We are not going to repeat here the basics of numpy which will be supposed to be known at least in a rudimentary
# way. For a detailed presentation of the library numpy, it will certainly be interesting to consult for example the
# following tutorial: [numpy quikstart](https://numpy.org/doc/stable/user/quickstart.html).

# %% [markdown]
# ## NDDataset
#
# SpectroChemPy NDDataset (dataset) are built from numerical lists or arrays. they are also homogeneous objects.
#
# Let's show it.

# %%
a = scp.NDDataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # create a dataset from a list
a  # to display the nddataset object

# %% [markdown]
# * NDDataset object share with numpy array some attributes and possess a few more.

# %%
a.shape, a.size, a.dtype

# %% [reduction] [markdown]
# * The underlying data can be retrieved from the "data" attribute (its a numpy array):

# %%
a.data

# %% [reduction] [markdown]
# * The main public attributes can be found using the `dir` function.

# %% [reduction]
dir(a)

# %% [markdown]
# Some of them will be used later in this tutorial

# %% [markdown]
# ## Creating a NDDataset
#
# There are many ways to create NDDataset object - it's interesting to learn about some of them.

# %%
b = scp.NDDataset.zeros((2, 3, 2))  # Create an NDDataset of all zeros
b

# %% [markdown]
# * some other functions (See the API reference for more details)
#
#   - scp.ones(shape, value) -> create a dataset with all one's values.
#   - scp.full(shape, value) -> create a dataset with a given initial value
#   - scp.empty(shape) -> create an empty dataset (data not initialised)
#   - scp.eye(N) -> Create a N x N identity matrix
#
#   ...
#

# %%
scp.NDDataset.arange(10, 12)  # create a rank 1 NDDataset with successive elements from 1 to 20, in steps of 3 (21 not
# included)

# %%
scp.NDDataset.linspace(-5, 5, 50,
                       endpoint=True)  # create a rank 1 NDDataset of 50 values evenly spaced between -5 and 5

# %% [reduction] [markdown]
# ## Creating a dataset from files or functions

# %% [reduction] [markdown]
# #### Importing data from files
# #Suppose you want to draw a nice plot with the results of experiments - It is possible to do this purely
# #using the numpy and matplotlib libraries, but we suggest using the tools included in SpectroChemPy for this task.
# First, some spectroscopic data import functions are defined in SpectroChemPy and can be used in a convenient way.
#
# See the [Import-Export User Guide](https://www.spectrochempy.fr/dev/user/userguide/dataset/4_importexport.html).
# **Warning:** *This is in progress*.
#
# I will give an example, using OMNIC data.

# %%
datadir = scp.pathclean(scp.preferences.datadir)  # this is a path as long as it points to an example data set.
# it can be replaced by any other path

# %% nbsphinx="hidden"
path = datadir / 'irdata' / 'CO@Mo_Al2O3.SPG'
data = scp.read(path)
data

# %% nbsphinx="hidden"
ax = data.plot()  # now plot it
_ = ax.set_title("Example")  # add a title

# %% [reduction] [markdown]
# ### Create a table from functions and plot it
#
# Suppose we have a model giving a series of data. For example, we have a simple model
# to generate NMR-free induction decay.
#
# We can create a specially dedicated function using the python block struture `def function(): ....`.
#
# But a quick and simple method is to use the "Lambda" function.

# %%
from numpy import cos, sin, exp


def fid(t, w, T2):
    pir = scp.Quantity("pi radians")
    if isinstance(w, str) and isinstance(T2, str) and t.has_units:
        # probably passed as quantity strings
        w = (scp.Quantity(w) * 2 * pir).to("radian/s")
        T2 = scp.Quantity(T2)
    return (cos(t * w) + 1j * sin(t * w)) * exp(-t / T2)


# in this model function, we generate a sinusoidal time evolution (t) with an angular frequency w, and a relaxation T2.


# %%
# example
t = scp.NDDataset.linspace(0, 100., 1000, units='s')  # t in second
y = fid(t, "200. MHz", "10 s")  # we obtain a complex network w in MHz, T2 in s
y

# %%
# plotting it
y.plot()

# %% [reduction]
# We can use spectrochemistry to create a data set:

# %%
timecoord = scp.Coord(t, title='time', units='second')
ndfid = scp.NDDataset(y, dims=['x'], coordset=[timecoord], title='signal')
ndfid.plot()

# %% [reduction]
# Basic operations on arrays and NDDatasets
#
# A key property of digitized tables is the ability to perform mathematical operations on and between them.

# %%
a = scp.NDDataset.full((5, 3),
                       10.)  # A 2D table with 5 rows and 3 columns  # b = np.array([[1, 2, 3, 4, 5]) # A 2D   #
# table with & row and 5 columns

# %% [markdown]
# It is clear that the addition of a and b which is done in an elementary way cannot be used in this case because
# the shapes don't match.

# %% [markdown]
# the addition of the two `a+b` tables gives an error :
#
#
# ValueError: operands could not be diffused with the forms (5.2) (1.5)
#
#
#
# because the form is missing.
#
#
# However, there is one case where such an operation can be performed, and that is when the tables have a different
# shape only.
#
#
