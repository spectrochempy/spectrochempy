# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.8
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %% [markdown]
# # The NDDataset object

# %% [markdown]
# The NDDataset is the main object use by **SpectroChemPy**.
#
# Like numpy ndarrays, NDDataset have the capability to be sliced, sorted and subject to
# mathematical operations.
#
# But, in addition, NDDataset may have units, can be masked and each dimensions can have
# coordinates also with units.
# This make NDDataset aware of units compatibility, *e.g.*, for binary operation such as
# additions or subtraction or during the application of mathematical operations.
# In addition or in replacement of numerical data for coordinates,
# NDDataset can also have labeled coordinates where labels can be different kind of
# objects (strings, datetime,
# numpy nd.ndarray or other NDDatasets, etc...).
#
# This offers a lot of flexibility in using NDDatasets that,  we hope, will be useful
# for applications.
# See the See the [Examples](../../../gettingstarted/examples/gallery/auto_examples_core/index.rst) for
# additional information about such possible applications.

# %% [markdown]
# **Below (and in the next sections), we try to give an almost complete view of the
# NDDataset features.**

# %% [markdown]
# As we will make some reference to the
# **[numpy](https://numpy.org/doc/stable/index.html)** library, we also import it here.
# %%
import numpy as np

import spectrochempy as scp

# %% [markdown]
# We additionally import the three main SpectroChemPy objects that we will use through
# this tutorial
# %%
from spectrochempy import Coord
from spectrochempy import CoordSet
from spectrochempy import NDDataset

# %% [markdown]
# For a convenient usage of units, we will also directly import
# **[ur]((#Units)**, the unit registry which contains all available
# units.
# %%
from spectrochempy import ur

# %% [markdown]
# Multidimensional array are defined in Spectrochempy using the `NDDataset` object.
#
# `NDDataset` objects mostly behave as numpy's `numpy.ndarray`
# (see for instance __
# [numpy quickstart tutorial](https://numpy.org/doc/stable/user/quickstart.html)__).

# %% [markdown]
# However, unlike raw numpy's ndarray, the presence of optional properties make
# them (hopefully) more appropriate for handling spectroscopic information,
# one of the major objectives of the SpectroChemPy package:
#
# *  `mask`: Data can be partially masked at will
# *  `units`: Data can have units, allowing units-aware operations
# *  `CoordSet`: Data can have a set of coordinates, one or several by dimensions
#
# Additional metadata can also be added to the instances of this class
# through the `meta` properties.

# %% [markdown]
# ## 1D-Dataset (unidimensional dataset)

# %% [markdown]
# In the following example, a minimal 1D dataset is created from a simple list,
# to which we can add some metadata:

# %%
d1D = NDDataset(
    [10.0, 20.0, 30.0],
    name="Dataset N1",
    author="Blake and Mortimer",
    description="A dataset from scratch",
    history="creation",
)
d1D

# %%
print(d1D)

# %%
d1D.plot(figsize=(3, 2))

# %% [markdown]
# Except few additional metadata such `author` , `created` ..., there is not much
# difference with respect to a conventional
# **[numpy.array](
# https://numpy.org/doc/stable/reference/generated/numpy.array.html#numpy.array)**.
# For example, one
# can apply numpy
# **[ufunc](https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs)'s**
# directly to a NDDataset or make basic arithmetic
# operation with these objects:

# %%
np.sqrt(d1D)

# %%
d1D += d1D / 2.0
d1D

# %% [markdown]
# As seen above, there are some attributes that are automatically added to the dataset:
#
# * `id`      : This is a unique identifier for the object.
# * `name`: A short and unique name for the dataset. It will beequal to the automatic
# `id` if it is not provided.
# * `author`  : Author determined from the computer name if not provided.
# * `created` : Date and time of creation.
# * `modified`: Date and time of modification.
#
# These attributes can be modified by the user, but the `id` , `created` and `modified`
# attributes are read only.
#
# Some other attributes are defined to describe the data:
# * `title`: A long name that will be used in plots or in some other operations.
# * `history`: history of operation achieved on the object since the object creation.
# * `description`: A comment or a description of the objects purpose or contents.
# * `origin`: An optional reference to the source of the data.

# %% [markdown]
# Here is an example of the use of the NDDataset attributes:

# %%
d1D.title = "intensity"
d1D.name = "mydataset"
d1D.history = "created from scratch"
d1D.description = "Some experimental measurements"
d1D

# %% [markdown]
# d1D is a 1D (1-dimensional) dataset with only one dimension.
#
# Some attributes are useful to check this kind of information:

# %%
d1D.shape  # the shape of 1D contain only one dimension size

# %%
d1D.ndim  # the number of dimensions

# %%
d1D.dims  # the name of the dimension (it has been automatically attributed)

# %% [markdown]
# **Note**: The names of the dimensions are set automatically. But they can be changed,
# with <u>the limitation</u> that the
# name must be a single letter.

# %%
d1D.dims = ["q"]  # change the list of dim names.

# %%
d1D.dims

# %% [markdown]
# ### nD-Dataset (multidimensional dataset)

# %% [markdown]
# To create a nD NDDataset, we can provide a nD-array like object to the NDDataset
# instance constructor

# %%
a = np.random.rand(2, 4, 6)
a

# %%
d3D = NDDataset(a)
d3D.title = "energy"
d3D.author = "Someone"
d3D.name = "3D dataset creation"
d3D.history = "created from scratch"
d3D.description = "Some example"
d3D.dims = ["u", "v", "t"]
d3D

# %% [markdown]
# We can also add all information in a single statement

# %%
d3D = NDDataset(
    a,
    dims=["u", "v", "t"],
    title="Energy",
    author="Someone",
    name="3D_dataset",
    history="created from scratch",
    description="a single statement creation example",
)
d3D

# %% [markdown]
# Three names are attributed at the creation (if they are not provided with the `dims`
# attribute, then the name are:
# 'z','y','x' automatically attributed)

# %%
d3D.dims

# %%
d3D.ndim

# %%
d3D.shape

# %% [markdown]
# ## About the dates and times
# The dates and times are stored internally as
# [UTC (Coordinated_Universal_Time)](https://en.wikipedia.org/wiki/Coordinated_
# Universal_Time).
# Timezone information is stored in the timezone attribute.
# If not set, the default is to use the local timezone,
# which is probably the most common case.

# %%
nd = NDDataset()
nd.created

# %% [markdown]
# In this case our local timezone has been used by default for the conversion from
# UTC datetime.

# %%
nd.local_timezone

# %%
nd.timezone = "EST"
nd.created

# %% [markdown]
# For a list of timezone code (TZ) you can have a look at
# [List_of_tz_database_time_zones](
# https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).

# %% [markdown]
# ## About the `history` attribute

# %% [markdown]
# The history is saved internally into a list, but its has a different behaviour than
# the usual list.
# The first time a NDDataset is created, the list is empty

# %%
nd = NDDataset()
nd.history

# %% [markdown]
# Assigning a string to the history attribute has two effects. The first one is that
# the string is appended automatically to the previous history list, and second it is
# preceeded by the time it has been added.

# %%
nd.history = "some history"
nd.history = "another history to append"
nd.history = "..."
nd.history

# %% [markdown]
# If you want to erase the history, assign an empty list

# %%
nd.history = []
nd.history

# %% [markdown]
# If you want to replace the full history use bracket around your history line:

# %%
nd.history = "Created form scratch"
nd.history = "a second ligne that will be erased"
nd.history = ["A more interesting message"]
nd.history

# %% [markdown]
# ## Units

# %% [markdown]
# One interesting possibility for a NDDataset is to have defined units for the internal
# data.

# %%
d1D.units = ur.eV  # ur is a registry containing all available units

# %%
d1D  # note the eV symbol of the units added to the values field below

# %% [markdown]
# This allows to make units-aware calculations:

# %%
d1D**2  # note the results in eV^2

# %%
np.sqrt(d1D)  # note the result in e^0.5

# %%
time = 5.0 * ur.second
d1D / time  # here we get results in eV/s

# %% [markdown]
# Conversion can be done between different units transparently

# %%
d1D.to("J")

# %%
d1D.to("K")

# %% [markdown]
# For more examples on how to use units with NDDataset, see the
# [gallery example](gettingstarted/examples/gallery/auto_examples_core/a_nddataset/plot_c_units.html

# %% [markdown]
# ## Coordinates

# %% [markdown]
# The above created `d3D` dataset has 3 dimensions, but no coordinate for these
# dimensions. Here arises a big difference
# with simple `numpy`-arrays:
# * We can add coordinates to each dimension of a NDDataset.

# %% [markdown]
# To get the list of all defined coordinates, we can use the `coords` attribute:

# %%
d3D.coordset  # no coordinates, so it returns nothing (None)

# %%
d3D.t  # the same for coordinate  t, v, u which are not yet set

# %% [markdown]
# To add coordinates, on way is to set them one by one:

# %%
d3D.t = (
    Coord.arange(6) * 0.1
)  # we need a sequence of 6 values for `t` dimension (see shape above)
d3D.t.title = "time"
d3D.t.units = ur.seconds
d3D.coordset  # now return a list of coordinates

# %%
d3D.t

# %%
d3D.coordset("t")  # Alternative way to get a given coordinates

# %%
d3D["t"]  # another alternative way to get a given coordinates

# %% [markdown]
# The two other coordinates u and v are still undefined

# %%
d3D.u, d3D.v

# %% [markdown]
# When the dataset is printed, only the information for the existing coordinates is
# given.

# %%
d3D

# %% [markdown]
# Programmatically, we can use the attribute `is_empty` or `has_data` to check this

# %%
d3D.v.has_data, d3D.v.is_empty

# %% [markdown]
# An error is raised when a coordinate doesn't exist

# %%
try:
    d3D.x
except KeyError as e:
    scp.error_(KeyError, e)

# %% [markdown]
# In some case it can also be useful to get a coordinate from its title instead of its
# name (the limitation is that if
# several coordinates have the same title, then only the first ones that is found in
# the coordinate list, will be
# returned - this can be ambiguous)

# %%
d3D["time"]

# %%
d3D.time

# %% [markdown]
# ## Labels

# %% [markdown]
# It is possible to use labels instead of numerical coordinates. They are sequence of
# objects .The length of the
# sequence must be equal to the size of a dimension.

# %% [markdown]
# The labels can be simple strings, *e.g.,*

# %%
tags = list("ab")
d3D.u.title = "some tags"
d3D.u.labels = tags  # TODO: avoid repetition
d3D

# %% [markdown]
# or more complex objects.
#
# For instance here we use datetime.timedelta objects:

# %%
from datetime import timedelta

start = timedelta(0)
times = [start + timedelta(seconds=x * 60) for x in range(6)]
d3D.t = None
d3D.t.labels = times
d3D.t.title = "time"
d3D

# %% [markdown]
# In this case, getting a coordinate that doesn't possess numerical data but labels,
# will return the labels

# %%
d3D.time

# %% [markdown]
# # More insight on coordinates

# %% [markdown]
# ## Sharing coordinates between dimensions

# %% [markdown]
# Sometimes it is not necessary to have different coordinates for each axe. Some can be
# shared between axes.
#
# For example, if we have a square matrix with the same coordinate in the two
# dimensions, the second dimension can
# refer to the first. Here we create a square 2D dataset, using the `diag` method:

# %%
nd = NDDataset.diag((3, 3, 2.5))
nd

# %% [markdown]
# and then we add the same coordinate for both dimensions

# %%
coordx = Coord.arange(3)
nd.set_coordset(x=coordx, y="x")
nd

# %% [markdown]
# ## Setting coordinates using `set_coordset`

# %% [markdown]
# Let's create 3 `Coord` objects to be used as coordinates for the 3 dimensions of the
# previous d3D dataset.

# %%
d3D.dims = ["t", "v", "u"]
s0, s1, s2 = d3D.shape
coord0 = Coord.linspace(10.0, 100.0, s0, units="m", title="distance")
coord1 = Coord.linspace(20.0, 25.0, s1, units="K", title="temperature")
coord2 = Coord.linspace(0.0, 1000.0, s2, units="hour", title="elapsed time")

# %% [markdown]
# ### Syntax 1

# %%
d3D.set_coordset(u=coord2, v=coord1, t=coord0)
d3D

# %% [markdown]
# ### Syntax 2

# %%
d3D.set_coordset({"u": coord2, "v": coord1, "t": coord0})
d3D

# %% [markdown]
# ## Adding several coordinates to a single dimension
# We can add several coordinates to the same dimension

# %%
coord1b = Coord([1, 2, 3, 4], units="millitesla", title="magnetic field")

# %%
d3D.set_coordset(u=coord2, v=[coord1, coord1b], t=coord0)
d3D

# %% [markdown]
# We can retrieve the various coordinates for a single dimension easily:

# %%
d3D.v_1

# %% [markdown]
# ## Math operations on coordinates
# Arithmetic operations can be performed on single coordinates:

# %%
d3D.u = d3D.u * 2
d3D.u

# %% [markdown]
# The ufunc numpy functions can also be applied, and will affect both the magnitude and
# the units of the coordinates:

# %%
d3D.u = 1.5 + np.sqrt(d3D.u)
d3D.u

# %% [markdown]
# A particularly frequent use case is to subtract the initial value from a coordinate. This can be done
# directly with the `-` operator:

# %%
d3D.u = d3D.u - d3D.u[0]
d3D.u

# %% [markdown]
# The operations above will generally *not* work on multiple coordinates, and
# will raise an error if attempted:

# %%
try:
    d3D.v = d3D.v - 1.5
except NotImplementedError as e:
    scp.error_(NotImplementedError, e)

# %% [markdown]
# Only subtraction between multiple coordinates is allowed, and will return a new `CoordSet` where each coordinate
# has been subtracted:

# %%
d3D.v = d3D.v - d3D.v[0]
d3D.v

# %% [markdown]
# It is always possible to carry out operations on a given coordinate
# of a CoordSet. This must be done by accessing the coordinate by its name, e.g. `'temperature'` or `'_2'` for
# the second coordinate of the `v` dimension:

# %%
d3D.v["_2"] = d3D.v["_2"] + 5.0
d3D.v

# %% [markdown]
# ## Summary of the coordinate setting syntax
# Some additional information about coordinate setting syntax

# %% [markdown]
# **A.** First syntax (probably the safer because the name of the dimension is
# specified, so this is less prone to
# errors!)

# %%
d3D.set_coordset(u=coord2, v=[coord1, coord1b], t=coord0)
# or equivalent
d3D.set_coordset(u=coord2, v=CoordSet(coord1, coord1b), t=coord0)
d3D

# %% [markdown]
# **B.** Second syntax assuming the coordinates are given in the order of the
# dimensions.
#
# Remember that we can check this order using the `dims` attribute of a NDDataset

# %%
d3D.dims

# %%
d3D.set_coordset((coord0, [coord1, coord1b], coord2))
# or equivalent
d3D.set_coordset(coord0, CoordSet(coord1, coord1b), coord2)
d3D

# %% [markdown]
# **C.** Third syntax (from a dictionary)

# %%
d3D.set_coordset({"t": coord0, "u": coord2, "v": [coord1, coord1b]})
d3D

# %% [markdown]
# **D.** It is also possible to use directly the `CoordSet` property

# %%
d3D.coordset = coord0, [coord1, coord1b], coord2
d3D

# %%
d3D.coordset = {"t": coord0, "u": coord2, "v": [coord1, coord1b]}
d3D

# %%
d3D.coordset = CoordSet(t=coord0, u=coord2, v=[coord1, coord1b])
d3D

# %% [markdown]
# <div class='alert alert-warning'>
# <b>WARNING</b>
#
# Do not use list for setting multiples coordinates! use tuples
# </div>

# %% [markdown]
# This raise an error (list have another signification: it's used to set a "same dim"
# CoordSet see example A or B)

# %%
try:
    d3D.coordset = [coord0, coord1, coord2]
except ValueError:
    scp.error_(
        ValueError,
        "Coordinates must be of the same size for a dimension with multiple coordinates",
    )

# %% [markdown]
# This works : it uses a tuple `()` , not a list `[]`

# %%
d3D.coordset = (
    coord0,
    coord1,
    coord2,
)  # equivalent to d3D.coordset = coord0, coord1, coord2
d3D

# %% [markdown]
# **E.** Setting the coordinates individually

# %% [markdown]
# Either a single coordinate

# %%
d3D.u = coord2
d3D
# %% [markdown]
# or multiple coordinates for a single dimension

# %%
d3D.v = [coord1, coord1b]
d3D
# %% [markdown]
# or using a CoordSet object.

# %%
d3D.v = CoordSet(coord1, coord1b)
d3D

# %% [markdown]
# # Methods to create NDDataset
#
# There are many ways to create `NDDataset` objects.
#
# Let's first create 2 coordinate objects, for which we can define `labels` and `units`!
# Note the use of the function
# `linspace`to generate the data.

# %%
c0 = Coord.linspace(
    start=4000.0, stop=1000.0, num=5, labels=None, units="cm^-1", title="wavenumber"
)

# %%
c1 = Coord.linspace(
    10.0, 40.0, 3, labels=["Cold", "RT", "Hot"], units="K", title="temperature"
)

# %% [markdown]
# The full coordset will be the following

# %%
cs = CoordSet(c0, c1)
cs


# %% [markdown]
# Now we will generate the full dataset, using a `fromfunction` method.
# All needed information are passed as
# parameter of the NDDataset instance constructor.

# %% [markdown]
# ## Create a dataset from a function


# %%
def func(x, y, extra):
    return x * y / extra


# %%
ds = NDDataset.fromfunction(
    func,
    extra=100 * ur.cm**-1,  # extra arguments passed to the function
    coordset=cs,
    name="mydataset",
    title="absorbance",
    units=None,
)  # when None, units will be determined from the function results

ds.description = """Dataset example created for this tutorial.
It's a 2-D dataset"""

ds.author = "Blake & Mortimer"
ds

# %% [markdown]
# ## Using numpy-like constructors of NDDatasets

# %%
dz = NDDataset.zeros(
    (5, 3), coordset=cs, units="meters", title="Datasets with only zeros"
)

# %%
do = NDDataset.ones(
    (5, 3), coordset=cs, units="kilograms", title="Datasets with only ones"
)

# %%
df = NDDataset.full(
    (5, 3), fill_value=1.25, coordset=cs, units="radians", title="with only float=1.25"
)
df

# %% [markdown]
# As with numpy, it is also possible to take another dataset as a template:

# %%
df = NDDataset.full_like(d3D, dtype="int", fill_value=2)
df

# %%
nd = NDDataset.diag((3, 3, 2.5))
nd

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
d3Dduplicate = NDDataset(d3D, name=f"duplicate of {d3D.name}", units="absorbance")
d3Dduplicate

# %% [markdown]
# ## Importing from external dataset
#
# NDDataset can be created from the importation of external data
#
# A **test**'s data folder contains some data for experimenting some features of
# datasets.

# %%
# let check if this directory exists and display its actual content:
datadir = scp.preferences.datadir
if datadir.exists():
    print(datadir.name)

# %% [markdown]
# Let's load grouped IR spectra acquired using OMNIC:

# %%
nd = NDDataset.read_omnic(datadir / "irdata/nh4y-activation.spg")
nd.preferences.reset()
nd.plot()

# %% [markdown]
# Even if we do not specify the **datadir**, the application first look in the
# directory by default.

# %% [markdown]
# Now, lets load a NMR dataset (in the Bruker format).

# %%
path = datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_2d"

# load the data directly (no need to create the dataset first)
nd2 = NDDataset.read_topspin(path, expno=1, remove_digital_filter=True)

# view it...
nd2.x.to("s")
nd2.y.to("ms")

ax = nd2.plot(method="map")
