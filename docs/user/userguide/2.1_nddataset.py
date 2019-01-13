# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 2. Create NDDataset objects

# %%
from spectrochempy import *

# %% [markdown]
# Multidimensional array are defined in Spectrochempy using the ``NDDataset`` object.
#
# ``NDDataset`` objects mostly behave as numpy's `numpy.ndarray`.
#
# However, unlike raw numpy's ndarray, the presence of optional properties such
# as `uncertainty`, `mask`, `units`, `axes`, and axes `labels` make them
# (hopefully) more appropriate for handling spectroscopic information, one of
# the major objectives of the SpectroChemPy package.
#
# Additional metadata can also be added to the instances of this class through the
# `meta` properties.

# %% [markdown]
# ## Create a ND-Dataset from scratch
#
# In the following example, a minimal 1D dataset is created from a simple list, to which we can add some metadata:

# %%
da = NDDataset([1,2,3])
da.title = 'intensity'   
da.history = 'created from scratch'
da.description = 'Some experimental measurements'
da.units = 'dimensionless'
print(da)

# %% [markdown]
# to get a rich display of the dataset, simply type on the last line of the cell.

# %%
da

# %% [markdown]
# Except few addtional metadata such `author`, `created` ..., there is not much
# differences with respect to a conventional `numpy.ndarray`. For example, one
# can apply numpy ufunc's directly to a NDDataset or make basic arithmetic
# operation with these objects:

# %%
da2 = np.sqrt(da**3)
da2

# %%
da3 = da + da/2.
da3

# %% [markdown]
# ## Create a NDDataset : full example
#
# There are many ways to create |NDDataset| objects.
#
# Above we have created a NDDataset from a simple list, but it is generally more
# convenient to create `numpy.ndarray`).
#
# Below is an example of a 3D-Dataset created from a ``numpy.ndarray`` to which axes can be added. 
#
# Let's first create the 3 one-dimensional coordinates, for which we can define labels, units, and masks! 

# %%
coord0 = Coord(data = np.linspace(200., 300., 3),
            labels = ['cold', 'normal', 'hot'],
            mask = None,
            units = "K",
            title = 'temperature')

coord1 = Coord(data = np.linspace(0., 60., 100),
            labels = None,
            mask = None,
            units = "minutes",
            title = 'time-on-stream')

coord2 = Coord(data = np.linspace(4000., 1000., 100),
            labels = None,
            mask = None,
            units = "cm^-1",
            title = 'wavenumber')

# %% [markdown]
# Here is the displayed info for coord1 for instance:

# %%
coord1

# %% [markdown]
# Now we create some 3D data (a ``numpy.ndarray``):

# %%
nd_data=np.array([np.array([np.sin(coord2.data*2.*np.pi/4000.)*np.exp(-y/60.) for y in coord1.data])*float(t) 
         for t in coord0.data])**2

# %% [markdown]
# The dataset is now created with these data and axis. All needed information are passed as parameter of the 
# NDDataset instance constructor. 

# %%
mydataset = NDDataset(nd_data,
               coordset = [coord0, coord1, coord2],
               title='Absorbance',
               units='absorbance'
              )

mydataset.description = """Dataset example created for this tutorial. 
It's a 3-D dataset (with dimensionless intensity)"""

mydataset.author = 'Blake & Mortimer'

# %% [markdown]
# We can get some information about this object:

# %%
print(mydataset)
mydataset

# %% [markdown]
# ## Copying existing NDDataset
#
# To copy an existing dataset, this is as simple as:

# %%
da_copy = da.copy()

# %% [markdown]
# or alternatively:

# %%
da_copy = da[:]

# %% [markdown]
# Finally, it is also possible to initialize a dataset using an existing one:

# %%
dc = NDDataset(da3, name='duplicate', units='absorbance')
dc

# %% [markdown]
# ### Other ways to create NDDatasets
#
# Any numpy creation function can be used to set up the initial dataset array:
#        [numpy array creation routines](https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html#routines-array-creation)
#

# %%
zeros((2,2), units='meters', title='Datasets with only zeros')

# %%
ones((2,2), units='kilograms', title='Datasets with only ones')

# %%
full((2,2), fill_value=1.25, units='radians', title = 'with only float=1.25')  #TODO: check the pb. of radians not displaying

# %% [markdown]
# As with numpy, it is also possible to take another dataset as a template:

# %%
ds = ones((2,3), dtype='bool')
ds

# %% [markdown]
# Now we use ``ds`` as a template, for the shape, but we can change the `dtype`.

# %%
full_like(ds, dtype=float, fill_value=2.5)

# %% [markdown]
# ## Importing from external dataset
#
# NDDataset can be created from the importation of external data
#
# A **test**'s data folder contains some data for experimenting some features of datasets.

# %%
# let check if this directory exists and display its actual content:
import os

datadir = general_preferences.datadir   
if os.path.exists(datadir):
    # let's display only the last part of the path
    print(os.path.basename(datadir))

# %% [markdown]
# ###  Reading a IR dataset saved by OMNIC (.spg extension)
#
# Even if we do not specify the **datadir**, the application first look in tht directory by default.

# %%
dataset = NDDataset.read_omnic(os.path.join('irdata', 'NH4Y-activation.SPG'))
dataset

# %% [markdown]
# ## Slicing a NDDataset

# %% [markdown]
# NDDataset can be sliced like conventional numpy-array...
#
# *e.g.,*:
#
# 1. by index, using a slice such as [3], [0:10], [:, 3:4], [..., 5:10], ...
#
# 2. by values, using a slice such as [3000.0:3500.0], [..., 300.0], ...
#
# 3. by labels, using a slice such as ['monday':'friday'], ...

# %%
new = mydataset[..., 0]
new

# %% [markdown]
# or using the axes labels:

# %%
new = mydataset['hot']
new

# %% [markdown]
# Be sure to use the correct type for slicing.
#
# Floats are use for slicing by values

# %%
correct = mydataset[...,2000.]

# %%
outside_limits = mydataset[...,10000.]

# %% [markdown]
# <div class='alert alert-info'>**NOTE:**
#
# If one use an integer value (2000), then the slicing is made **by index not by value**, and in the following particular case, an `IndexError` is issued as index 2000 does not exists (size along axis -1 is only 100, so that index vary between 0 and 99!).
#
# </div>

# %% [markdown]
# When slicing by index, an error is generated is the index is out of limits:

# %%
try:
    fail = mydataset[...,2000]
except IndexError as e:
    log.error(e)

# %% [markdown]
# One can mixed slicing methods for different dimension:

# %%
new = mydataset['normal':'hot', 0, 4000.0:2000.]
new

# %% [markdown]
# ## Loading of experimental data

# %% [markdown]
#
# ### NMR Data

# %% [markdown]
# Now, lets load a NMR dataset (in the Bruker format).

# %%
path = os.path.join(datadir, 'nmrdata','bruker', 'tests', 'nmr','bruker_1d')

# load the data in a new dataset
ndd = NDDataset()
ndd.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
ndd

# %%
# view it...
_ = ndd.plot(color='blue')

# %%
path = os.path.join(datadir, 'nmrdata','bruker', 'tests', 'nmr','bruker_2d')

# load the data directly (no need to create the dataset first)
ndd2 = NDDataset.read_bruker_nmr(path, expno=1, remove_digital_filter=True)

# view it...
ndd2.x.to('s')
ndd2.y.to('ms')

ax = ndd2.plot() 

# %% [markdown]
# ### IR data

# %%
dataset = NDDataset.read_omnic(os.path.join(datadir, 'irdata', 'NH4Y-activation.SPG'))
dataset

# %%
dataset = read_omnic(NDDataset(), filename=os.path.join(datadir, 'irdata', 'NH4Y-activation.SPG'))

# %%
ax = dataset.plot(method='stack')

# %% [markdown]
# ## Masks
#
# Masking values in a dataset is straigthforward. Just set a value `masked` or True for those data you want to mask.

# %%
dataset[:,1290.:890.] = masked
ax = dataset.plot_stack()

# %% [markdown]
# Here is a display the figure with the new mask

# %%
_ = dataset.plot_stack(figsize=(9, 4))

# %% [markdown]
# ## Transposition

# %% [markdown]
# Dataset can be transposed

# %%
datasetT = dataset.T
datasetT

# %%
_ = datasetT.plot()

# %%
_ = dataset.T[4000.:3000.].plot_stack()

# %% [markdown]
# ## Units
#
#
# Spectrochempy can do calculations with units - it uses [pint](https://pint.readthedocs.io) to define and perform operation on data with units.
#
# ### Create quantities
#
# * to create quantity, use for instance, one of the following expression:

# %%
Quantity('10.0 cm^-1')

# %%
Quantity(1.0, 'cm^-1/hour')

# %%
Quantity(10.0, ur.cm/ur.km)

# %% [markdown]
# or may be (?) simpler,

# %%
10.0 * ur.meter/ur.gram/ur.volt

# %% [markdown]
# `ur` stands for **unit registry**, which handle many type of units
# (and conversion between them)

# %% [markdown]
# ### Do arithmetics with units

# %%
a = 900 * ur.km
b = 4.5 * ur.hours
a/b

# %% [markdown]
# Such calculations can also be done using the following syntax, using a string expression

# %%
Quantity("900 km / (8 hours)")

# %% [markdown]
# ### Convert between units

# %%
c = a/b
c.to('cm/s')

# %% [markdown]
# We can make the conversion *inplace* using *ito* instead of *to*

# %%
c.ito('m/s')
c

# %% [markdown]
# ### Do math operations with consistent units

# %%
x = 10 * ur.radians
np.sin(x)

# %% [markdown]
# Consistency of the units are checked!

# %%
x = 10 * ur.meters
np.sqrt(x)

# %% [markdown]
# but this is wrong...

# %%
x = 10 * ur.meters
try:
    np.cos(x)
except DimensionalityError as e:
    log.error(e)

# %% [markdown]
# Units can be set for NDDataset data and/or Coordinates

# %%
ds = NDDataset([1., 2., 3.], units='g/cm^3', title='concentration')
ds

# %%
ds.to('kg/m^3')

# %% [markdown]
# ## Uncertainties
#
# Spectrochempy can do calculations with uncertainties (and units).
#
# A quantity, with an `uncertainty` is called a **Measurement** .
#
# Use one of the following expression to create such `Measurement`:

# %%
Measurement(10.0, .2, 'cm') 

# %%
Quantity(10.0, 'cm').plus_minus(.2)   

# %% [markdown]
# ## Numpy universal functions (ufunc's)
#
# A numpy universal function (or `numpy.ufunc` for short) is a function that
# operates on `numpy.ndarray` in an element-by-element fashion. It's
# vectorized and so rather fast.
#
# As SpectroChemPy NDDataset imitate the behaviour of numpy objects, many numpy
# ufuncs can be applied directly.
#
# For example, if you need all the elements of a NDDataset to be changed to the
# squared rooted values, you can use the `numpy.sqrt` function:

# %%
da = NDDataset([1., 2., 3.])
da_sqrt = np.sqrt(da)
da_sqrt

# %% [markdown]
# ### Ufuncs with NDDataset with units
#
# When NDDataset have units, some restrictions apply on the use of ufuncs:
#
# Some function functions accept only dimensionless quantities. This is the
# case for example of logarithmic functions: :`exp` and `log`.

# %%
np.log10(da)

# %%
da.units = ur.cm

try:
    np.log10(da)
except DimensionalityError as e:
    log.error(e)

# %% [markdown]
# ## Complex or hypercomplex NDDatasets
#
#
# NDDataset objects with complex data are handled differently than in
# `numpy.ndarray`.
#
# Instead, complex data are stored by interlacing the real and imaginary part.
# This allows the definition of data that can be complex in several axis, and *e
# .g.,* allows 2D-hypercomplex array that can be transposed (useful for NMR data).

# %%
da = NDDataset([  [1.+2.j, 2.+0j], [1.3+2.j, 2.+0.5j],[1.+4.2j, 2.+3j], [5.+4.2j, 2.+3j ] ])
da

# %% [markdown]
# if the dataset is also complex in the first dimension (columns) then we
# should have (note the shape description!):

# %%
da.set_complex(0)
da
