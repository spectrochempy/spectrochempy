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
# # The NDDataset object

# %% [markdown]
# The NDDataset is the main object use by **SpectroChemPy**. 
#
# Like numpy ndarrays, NDDataset have the capability to be sliced, sorted and subject to matematical operations. 
#
# But, in addition, NDDataset may have units, can be masked and each dimensions can have coordinates also with units. This make NDDataset aware of unit compatibility, *e.g.*, for binary operation such as addtions or subtraction or during the application of mathematical operations. In addition or in replacement of numerical data for coordinates, NDDatset can also have labeled coordinates where labels can be different kind of objects (strings, datetime, numpy nd.ndarray or othe NDDatasets, etc...). 
#
# This offers a lot of flexibility in using NDDatasets that,  we hope, will be useful for applications. See the **Tutorials** for more information about such possible applications. 
#
# **SpectroChemPy** provides another kind of data structure, aggregating several datasets: **NDPanel**: See 

# %% [markdown]
# **Below (and in the next sections), we try to give an almost complete view of the NDDataset features.**

# %%
from spectrochempy import *

# %% [markdown]
# Multidimensional array are defined in Spectrochempy using the ``NDDataset`` object.
#
# ``NDDataset`` objects mostly behave as numpy's `numpy.ndarray`.
#
# However, unlike raw numpy's ndarray, the presence of optional properties make them (hopefully) more appropriate for handling spectroscopic information, one of the major objectives of the SpectroChemPy package:
#
# * **mask**, 
# * **units**, 
# * and **coords**.
#
# Additional metadata can also be added to the instances of this class through the `meta` properties.

# %% [markdown]
# ## Create a ND-Dataset from scratch

# %% [markdown]
# ### 1D-Dataset (unidimensional dataset)

# %% [markdown]
# In the following example, a minimal 1D dataset is created from a simple list, to which we can add some metadata:

# %%
d1D = NDDataset([10., 20., 30.])
print_(d1D)


# %% [markdown]
# <div class='alert-info'>
#
# **Note**: In the above code, we use of `print_` (with an underscore) not the usual `print` function. 
# The `print` output only a short line of information
#
# </div>

# %%
d1D.plot()

# %%
print(d1D)

# %% [markdown]
# To get a rich display of the dataset, we can simply type on the last line of the cell: This output a html version of the information string.

# %%
d1D

# %% [markdown]
# Except few addtional metadata such `author`, `created` ..., there is not much
# differences with respect to a conventional `numpy.ndarray`. For example, one
# can apply numpy ufunc's directly to a NDDataset or make basic arithmetic
# operation with these objects:

# %%
np.sqrt(d1D ** 3)

# %%
d1D + d1D / 2.

# %% [markdown]
# As seen above, there is some metadata taht are automatically added to the dataset:
#
# * **`id`**     : This is a unique identifier for the object
# * **`author`** : author determined from the computer name
# * **`created`**: date/time of creation
# * **`modified`**: date/time of modification
#
# additionaly, dataset can have a **`name`** (equal to the `id` if it is not provided)
#
# Some other metadata are defined:
#
# * **`history`**: history of operation achieved on the object since the object creation
# * **`description`**: A user friendly description of the objects purpose or contents.
# * **`title`**: A title that will be used in plots or in some other operation on the objects.
#
#
# All this metadata (except, the `id`, `created`, `modified`) can be changed by the user.
#
# For instance:

# %%
d1D.title = 'intensity'
d1D.name = 'mydataset'
d1D.history = 'created from scratch'
d1D.description = 'Some experimental measurements'
d1D

# %% [markdown]
# d1D is a 1D (1-dimensional) dataset with only one dimension. 
#
# Some attributes are useful to check this kind of information:

# %%
d1D.shape # the shape of 1D contain only one dimension size

# %%
d1D.ndim # the number of dimensions

# %%
d1D.dims # the name of the dimension (it has been automatically attributed)

# %% [markdown]
# **Note**: The names of the dimensions are set automatically. But they can be changed, with the limitation that the name must be a single letter.

# %%
d1D.dims = ['q']  # change the list of dim names.

# %%
d1D.dims

# %% [markdown]
# ### nD-Dataset (multidimensional dataset)

# %% [markdown]
# To create a nD NDDataset, we have to provide a nD-array like object to the NDDataset instance constructor

# %%
a = np.random.rand(2,4,6)   # note here that np (for numpy space has been automatically
                            # imported with spectrochempy, thus no need to use the 
                            # classical `import numpy as np`)
a

# %%
d2D = NDDataset(a)
d2D.title = 'Energy'
d2D.name = '3D dataset creation'
d2D.history = 'created from scratch'
d2D.description = 'Some example'
d2D.dims = ['v','u','t']
d2D

# %% [markdown]
# We can also add all information in a single statement

# %%
d2D = NDDataset(a, dims = ['v','u','t'], title = 'Energy', name = '3D_dataset', 
                history = 'created from scratch', description = 'a single line creation example')
d2D

# %% [markdown]
# Three names are attributed at the creation (if they are not provided with the `dims` attribute, then the name are: 'z','y','x' automatically attributed)

# %%
d2D.dims 

# %%
d2D.ndim

# %%
d2D.shape

# %% [markdown]
# ### Units

# %% [markdown]
# One interesting possibility for a NDDataset is to have defined units for the internal data.

# %%
d1D.units = 'eV'

# %%
d1D  # note the eV symbol of the units added to the values field below

# %% [markdown]
# This allows to make units-aware calculations:

# %%
np.sqrt(d1D) # note the results en eV^0.5

# %%
time = 5.*ur.second   # ur is a registry containing all available units
d1D/time              # here we get results in eV/s

# %% [markdown]
# Conversion can be done between different units transparently

# %%
d1D.to('J')

# %%
d1D.to('K')

# %% [markdown]
# ### Coordinates

# %% [markdown]
# The above created `d2D` dataset has 3 dimensions, but no coordinate for these dimensions. Here arises a big difference with simple `numpy`-arrays: 
# * We can add coordinates to each dimensions of a NDDataset. 

# %% [markdown]
# To get the list of all defined coordinates, we can use the `coords` attribute:

# %%
d2D.coords  # no coordinates, so it returns nothing (None)

# %%
d2D.t       # the same for coordinate  u, v, t which are not yet set

# %% [markdown]
# To add coordinates, on way is to set them one by one:

# %%
d2D.t = np.arange(6)*.1 # we need a sequence of 6 values for `t` dimension (see shape above) 
d2D.t.title = 'time'
d2D.t.units = 'seconds'
d2D.coords # now return a list of coordinates

# %%
d2D.t   

# %%
d2D.coords('t')  # Alternative way to get a given coordinates

# %%
d2D['t'] # another alternative way to get a given coordinates

# %% [markdown]
# The two other coordinates u and v are still undefined

# %%
d2D.u

# %%
d2D.v

# %% [markdown]
# When the dataset is printed, only the information for the existing coordinates is given.

# %%
d2D

# %% [markdown]
# Programatically, we can use the attribute `is_empty` or `has_data` to check this

# %%
d2D.v.has_data, d2D.v.is_empty

# %% [markdown]
# An error is raised when a coordinate doesn't exist

# %%
try:
    d2D.x
except KeyError:
    error_('not found')

# %% [markdown]
# In some case it can also be usefull to get a coordinate from its title instead of its name (the limitation is that if several coordinates have the same title, then only the first ones that is found in the coordinate list, will be returned - this can be ambiguous) 

# %%
d2D['time']

# %%
d2D.time

# %% [markdown]
# It is possible to use labels instead of numerical coordinates. They are sequence of objects .The length of the sequence must be equal to the size of a dimension

# %%
from datetime import datetime, timedelta, time
timedelta()

# %%
start = timedelta(0)
times = [start + timedelta(seconds=x*60) for x in range(6)]
d2D.t = None
d2D.t.labels = times
d2D.t.title = 'time'
d2D

# %%
tags = list('abcdef')
d2D.t.labels = tags
d2D

# %% [markdown]
# In this case, getting a coordinate that doesn't possess numerical data but labels, will return the labels

# %%
d2D.time

# %% [markdown]
# Sometimes it is not necessary to have different coordinates for the various axes. 
#
# For example, if we have a square matrix with the same coordinate in the two dimensions, the second dimension can refer to the first.

# %%
a = np.diag((3,3,2.5))
nd = NDDataset(a, coords=CoordSet(x=np.arange(3), y='x'))
nd

# %% [markdown]
# ## Create a NDDataset: full example
#
# There are many ways to create `NDDataset` objects.
#
# Above we have created a `NDDataset` from a simple list, but also from a `numpy.ndarray`).
#
# Below is an example of a 3D-Dataset created from a ``numpy.ndarray`` to which axes for each dimension can be added at creation. 
#
# Let's first create the 3 one-dimensional coordinates, for which we can define `labels`, `units`, and `masks`! 

# %%
coord0 = Coord(data=np.linspace(4000., 1000., 100),
               labels=None,
               mask=None,
               units="cm^-1",
               title='wavenumber')

coord1 = Coord(data=np.linspace(0., 60., 60),
               labels=None,
               mask=None,
               units="minutes",
               title='time-on-stream')

coord2 = Coord(data=np.linspace(200., 300., 3),
               labels=['cold', 'normal', 'hot'],
               mask=None,
               units="K",
               title='temperature')

# %% [markdown]
# Here is the displayed info for coord1 for instance:

# %%
coord1

# %% [markdown]
# Now we create some 3D data (a ``numpy.ndarray``):

# %%
nd_data = np.array(
    [np.array([np.sin(coord2.data * 2. * np.pi / 4000.) * np.exp(-y / 60.) for y in coord1.data]) * float(t)
     for t in coord0.data]) ** 2

# %% [markdown]
# The dataset is now created with these data and axis. All needed information are passed as parameter of the 
# NDDataset instance constructor. 

# %%
d3D = NDDataset(nd_data,
                      name = 'mydataset',
                      coords=[coord0, coord1, coord2],
                      title='Absorbance',
                      units='absorbance'
                      )

d3D.description = """Dataset example created for this tutorial. 
It's a 3-D dataset (with dimensionless intensity)"""

d3D.author = 'Blake & Mortimer'

# %% [markdown]
# We can get some information about this object:

# %%
d3D

# %% [markdown]
# One can set all the coordinates independantly

# %%
d3D = NDDataset(nd_data,
                      name = 'mydataset',
                      title='Absorbance',
                      units='absorbance'
                      )
d3D.description = """Dataset example created for this tutorial. 
It's a 3-D dataset (with dimensionless intensity)"""

d3D.author = 'Blake & Mortimer'
d3D

# %%
d3D.set_coords(x=coord2, y=coord1, z=coord0)          # syntax 1
d3D.set_coords({'x':coord2, 'y':coord1, 'z':coord0})  # syntax 2
d3D

# %% [markdown]
# One can add several coordinates to the same dimension

# %%
coord2b = Coord([1,2,3], units='millitesla', title='magnetic field')

# %%
d3D.set_coords(x=CoordSet(coord2,coord2b), y=coord1, z=coord0)
d3D


# %% [markdown]
# Some additional information about coordinate setting syntax

# %%
# A. fist syntax (probably the safer because thename of the dimension is specified, so this is less prone to errors!)
d3D.set_coords(x=CoordSet(coord2,coord2b), y=coord1, z=coord0)
d3D.set_coords(x=[coord2,coord2b], y=coord1, z=coord0) # equivalent

# B. second syntax in the order of the dimensions: z,y,x (if no swap or transpopse has been performed)
d3D.set_coords(coord0, coord1, [coord2,coord2b])
d3D.set_coords((coord0, coord1, [coord2,coord2b]))  # equivalent
   
# C. third syntax (from a dictionary)
d3D.set_coords({'z':coord0, 'y':coord1, 'x':[coord2,coord2b]})

# D. Fourth syntax (from another coordset)
d3D.set_coords(**CoordSet(z=coord0, y=coord1, x=[coord2,coord2b]))   # note the **

# It is also possible to use the coords property (with slightly less possibility)
d3D.coords = coord0, coord1,[coord2,coord2b]
d3D.coords = {'z':coord0, 'y':coord1, 'x':[coord2,coord2b]}
d3D.coords = CoordSet(z=coord0, y=coord1, x=[coord2,coord2b])

# %% [markdown]
# WARNING: do not use list for setting multiples coordinates! use tuples

# %%
# This raise an error (list have another signification: it's used to set a "same dim" CoordSet see example A or B)
try:
    d3D.coords = [coord0, coord1, coord2]
except ValueError:
    error_('Coordinates must be of the same size for a dimension with multiple coordinates')
    
# This works (not a tuple `()`, not a list `[]`)
d3D.coords = (coord0, coord1, coord2) 

# %% [markdown]
# ## Copying existing NDDataset
#
# To copy an existing dataset, this is as simple as:

# %%
d3D_copy = d3D.copy()

# %% [markdown]
# or alternatively:

# %%
d3D_copy = d3D[:]

# %% [markdown]
# Finally, it is also possible to initialize a dataset using an existing one:

# %%
d3Dduplicate = NDDataset(d3D, name='duplicate of %s'%d3D.name , units='absorbance')
d3Dduplicate

# %% [markdown]
# ### Other ways to create NDDatasets
#
# Some numpy creation function can be used to set up the initial dataset array:
#        [numpy array creation routines](https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html#routines-array-creation)
#

# %%
dz = zeros((2, 2), units='meters', title='Datasets with only zeros')
dz

# %%
do = ones((2, 2), units='kilograms', title='Datasets with only ones')
do

# %%
df = full((2, 2), fill_value=1.25, units='radians',
     title='with only float=1.25')  
df

# %% [markdown]
# As with numpy, it is also possible to take another dataset as a template:

# %%
do = ones((2, 3), dtype=bool)
do[1,1]=0
do

# %% [markdown]
# Now we use the previous dataset ``do`` as a template, for the shape, but we can change the `dtype`.

# %%
df = full_like(d3D, dtype=np.float64, fill_value=2.5)
df


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
dataset = NDDataset.read_omnic(os.path.join('irdata', 'nh4y-activation.spg'))
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
new = d3D[..., 0]
new

# %% [markdown]
# or using the axes labels:

# %%
new = d3D[..., 'hot']
new

# %% [markdown]
# Be sure to use the correct type for slicing.
#
# Floats are used for slicing by values

# %%
correct = d3D[2000.]
correct

# %%
outside_limits = d3D[2000]

# %% [markdown]
# <div class='alert alert-info'>
#     
# **NOTE:**
# If one use an integer value (2000), then the slicing is made **by index not by value**, and in the following particular case, an `Error` is issued as index 2000 does not exists (size along axis `x` (axis:0) is only 100, so that index vary between 0 and 99!). 
#
# </div>

# %% [markdown]
# One can mixed slicing methods for different dimension:

# %%
new = d3D[4000.0:2000., 0, 'normal':'hot']
new

# %% [markdown]
# ## Loading of experimental data

# %% [markdown]
#
# ### NMR Data

# %% [markdown]
# Now, lets load a NMR dataset (in the Bruker format).

# %%
path = os.path.join(datadir, 'nmrdata', 'bruker', 'tests', 'nmr', 'bruker_1d')

# load the data in a new dataset
ndd = NDDataset()
ndd.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
ndd

# %%
# view it...
_ = ndd.plot(color='blue')

# %%
path = os.path.join(datadir, 'nmrdata', 'bruker', 'tests', 'nmr', 'bruker_2d')

# load the data directly (no need to create the dataset first)
ndd2 = NDDataset.read_bruker_nmr(path, expno=1, remove_digital_filter=True)

# view it...
ndd2.x.to('s')
ndd2.y.to('ms')

ax = ndd2.plot(method='map')
ndd2

# %% [markdown]
# ### IR data

# %%
dataset = NDDataset.read_omnic(os.path.join(datadir, 'irdata', 'NH4Y-activation.SPG'))
dataset

# %%
ax = dataset.plot(method='stack')

# %% [markdown]
# ## Masks

# %% [markdown]
# if we try to get for example the maximum of the previous dataset, we face a problem due to the saturation around 1100 cm$^{-1}$.

# %%
dataset.max()

# %% [markdown]
# One way is to apply the max function to only a part of the spectrum. Another way is to mask the undesired data.
#
# Masking values in a dataset is straigthforward. Just set a value `masked` or True for those data you want to mask.

# %%
dataset[1290.:890.] = MASKED

# %% [markdown]
# Now the max function return the  correct position 

# %%
dataset.max()

# %% [markdown]
# Here is a display the figure with the new mask

# %%
_ = dataset.plot_stack()

# %% [markdown]
# ## Transposition

# %% [markdown]
# Dataset can be transposed

# %%
datasetT = dataset.T
datasetT

# %% [markdown]
# As it can be observed the dimension `x`and `y`have been exchanged, *e.g.* the originalshape was **(x:5549, y:55)**, and after transposition it is **(y:55, x:5549)**.
# (the dimension names stay the same, but the index of the corresponding axis are exchanged).

# %% [markdown]
# Let's vizualize the result:

# %%
_ = datasetT.plot()

# %%
dataset[:, 4000.:3000.], datasetT[4000.:3000.]

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
    error_(e)

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
da = NDDataset([[1. + 2.j, 2. + 0j], [1.3 + 2.j, 2. + 0.5j], [1. + 4.2j, 2. + 3j], [5. + 4.2j, 2. + 3j]])
da

# %% [markdown]
# A dataset of type float can be transformed into a complex dataset (using two cionsecutive rows to create a complex row)

# %%
da = NDDataset(np.arange(40).reshape(10,4))
da

# %%
dac = da.set_complex()
dac

# %% [markdown]
# Note the `x`dimension size is divided by a factor of two 

# %% [markdown]
# A dataset which is complex in two dimensions is called hypercomplex (it's datatype in SpectroChemPy is set to quaternion). 

# %%
daq = da.set_quaternion()   # equivalently one can use the set_hypercomplex method
daq

# %% pycharm={"name": "#%%\n"}
daq.dtype


