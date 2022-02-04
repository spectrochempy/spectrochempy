---
jupytext:
  formats: ipynb,py:percent,md:myst
  notebook_metadata_filter: all
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
language_info:
  codemirror_mode:
    name: ipython
    version: 3
  file_extension: .py
  mimetype: text/x-python
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
  version: 3.9.9
widgets:
  application/vnd.jupyter.widget-state+json:
    state: {}
    version_major: 2
    version_minor: 0
---

# The NDDataset object

+++

The [NDDataset](../reference/generated/spectrochempy.NDDataset.rst) is the main object use by **SpectroChemPy**.

Like numpy ndarrays, NDDataset have the capability to be sliced, sorted and subject to mathematical operations.

But, in addition, NDDataset may have units, can be masked and each dimensions can have coordinates also with units.
This make NDDataset aware of units compatibility, *e.g.*, for binary operation such as additions or subtraction or
during the application of mathematical operations. In addition or in replacement of numerical data for coordinates,
NDDataset can also have labeled coordinates where labels can be different kind of objects (strings, datetime,
numpy nd.ndarray or other NDDatasets, etc...).

This offers a lot of flexibility in using NDDatasets that,  we hope, will be useful for applications.
See the **[Examples](../../gettingstarted/gallery/auto_examples/index.rst)** for additional information about such possible applications.

+++

**Below (and in the next sections), we try to give an almost complete view of the NDDataset features.**

```{code-cell} ipython3
import spectrochempy as scp
```

As we will make some reference to the **[numpy](https://numpy.org/doc/stable/index.html)** library, we also import it here.

```{code-cell} ipython3
import numpy as np
```

We additionally import the three main SpectroChemPy objects that we will use through this tutorial

```{code-cell} ipython3
from spectrochempy import NDDataset, CoordSet, Coord
```

For a convenient usage of units, we will also directly import **[ur]((../units/units.ipynb)**, the unit registry which contains all available
units.

```{code-cell} ipython3
from spectrochempy import ur
```

Multidimensional array are defined in Spectrochempy using the `NDDataset` object.

`NDDataset` objects mostly behave as numpy's `numpy.ndarray`
(see for instance __
[numpy quickstart tutorial](https://numpy.org/doc/stable/user/quickstart.html)__).

+++

However, unlike raw numpy's ndarray, the presence of optional properties make
them (hopefully) more appropriate for handling spectroscopic information,
one of the major objectives of the SpectroChemPy package:

*  `mask`: Data can be partially masked at will
*  `units`: Data can have units, allowing units-aware operations
*  `coordset`: Data can have a set of coordinates, one or several by dimensions

Additional metadata can also be added to the instances of this class
through the `meta` properties.

+++ {"tags": []}

## 1D-Dataset (unidimensional dataset)

+++

In the following example, a minimal 1D dataset is created from a simple list,
to which we can add some metadata:

```{code-cell} ipython3
d1D = NDDataset(
    [10.0, 20.0, 30.0],
    name="Dataset N1",
    author="Blake and Mortimer",
    comment="A dataset from scratch",
)
d1D
```

<div class='alert alert-info'>
    <b>Note</b>

 In the above code, run in a notebook, the output of d1D is in html for a nice display.

 To get the same effect, from a console script, one can use `print_` (with an underscore) and not the usual python
 function `print`. As you can see below, the `print` function only gives a short summary of the information,
 while the `print_` method gives more detailed output

</div>

```{code-cell} ipython3
print(d1D)
```

```{code-cell} ipython3
scp.print_(d1D)
```

```{code-cell} ipython3
_ = d1D.plot(figsize=(3, 2))
```

Except few additional metadata such `author`, `created` ..., there is not much
difference with respect to a conventional **[numpy.array](https://numpy.org/doc/stable/reference/generated/numpy.array.html#numpy.array)**. For example, one
can apply numpy **[ufunc](https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs)'s** directly to a NDDataset or make basic arithmetic
operation with these objects:

```{code-cell} ipython3
np.sqrt(d1D)
```

```{code-cell} ipython3
d1D + d1D / 2.0
```

As seen above, there are some attributes that are automatically added to the dataset:

* `id`      : This is a unique identifier for the object.
* `name`: A short and unique name for the dataset. It will beequal to the automatic `id` if it is not provided.
* `author`  : Author determined from the computer name if not provided.
* `created` : Date and time of creation.
* `modified`: Date and time of modification.

These attributes (except, the `id`, `created`, `modified` which are read-only) can be changed by the user.

Some other attributes are defined to describe the data:
* `long_name`: A long name that will be used in plots or in some other operations.
* `history`: history of operation achieved on the object since the object creation.
* `comment`: A comment or a description of the objects purpose or contents.
* `source`: An optional reference to the source of the data.

+++

Here is an example of the use of the NDDataset attributes:

```{code-cell} ipython3
d1D.name = "mydataset"
d1D.long_name = "intensity"
d1D.history = "created from scratch"
d1D.comment = "Some experimental measurements"
d1D
```

d1D is a 1D (1-dimensional) dataset with only one dimension.

Some attributes are useful to check this kind of information:

```{code-cell} ipython3
d1D.shape  # the shape of 1D contain only one dimension size
```

```{code-cell} ipython3
d1D.ndim  # the number of dimensions
```

```{code-cell} ipython3
d1D.dims  # the name of the dimension (it has been automatically attributed)
```

**Note**: The names of the dimensions are set automatically. But they can be changed, with <u>the limitation</u> that the
name must be a single letter.

```{code-cell} ipython3
d1D.dims = ["q"]  # change the list of dim names.
```

```{code-cell} ipython3
d1D.dims
```

<div class='alert alert-info'>
    <b>Note</b>
    The attributes comment**, **long_name** and **source** where added in version 0.4.0 in replacement of **description**, **title** and **origin**, respectively. For backward compatibility, the previous attributes are still accessible but issue a deprecated warning when used.   
</div>

+++

### nD-Dataset (multidimensional dataset)

+++

To create a nD NDDataset, we can provide a nD-array like object to the NDDataset instance constructor

```{code-cell} ipython3
a = np.random.rand(2, 4, 6)
a
```

```{code-cell} ipython3
d3D = NDDataset(a)
d3D.long_name = "energy"
d3D.author = "Someone"
d3D.name = "3D dataset creation"
d3D.history = "created from scratch"
d3D.comment = "Some example"
d3D.dims = ["u", "v", "t"]
d3D
```

We can also add all information in a single statement

```{code-cell} ipython3
d3D = NDDataset(
    a,
    dims=["u", "v", "t"],
    long_name="Energy",
    author="Someone",
    name="3D_dataset",
    history="created from scratch",
    comment="a single statement creation example",
)
d3D
```

Three names are attributed at the creation (if they are not provided with the `dims` attribute, then the name are:
'z','y','x' automatically attributed)

```{code-cell} ipython3
d3D.dims
```

```{code-cell} ipython3
d3D.ndim
```

```{code-cell} ipython3
d3D.shape
```

## Units

+++

One interesting possibility for a NDDataset is to have defined units for the internal data.

```{code-cell} ipython3
d1D.units = ur.eV  # ur is a registry containing all available units
```

```{code-cell} ipython3
d1D  # note the eV symbol of the units added to the values field below
```

This allows to make units-aware calculations:

```{code-cell} ipython3
d1D ** 2  # note the results in eV^2
```

```{code-cell} ipython3
np.sqrt(d1D)  # note the result in e^0.5
```

```{code-cell} ipython3
time = 5.0 * ur.second
d1D / time  # here we get results in eV/s
```

Conversion can be done between different units transparently

```{code-cell} ipython3
d1D.to("J")
```

```{code-cell} ipython3
d1D.to("K")
```

## Coordinates

+++

The above created `d3D` dataset has 3 dimensions, but no coordinate for these dimensions. Here arises a big difference
with simple `numpy`-arrays:
* We can add coordinates to each dimension of a NDDataset.

+++

To get the list of all defined coordinates, we can use the `coords` attribute:

```{code-cell} ipython3
d3D.coordset  # no coordinates, so it returns nothing (None)
```

```{code-cell} ipython3
d3D.t  # the same for coordinate  t, v, u which are not yet set
```

To add coordinates, on way is to set them one by one:

```{code-cell} ipython3
d3D.t = (
    Coord.arange(6) * 0.1
)  # we need a sequence of 6 values for `t` dimension (see shape above)
d3D.t.long_name = "time"
d3D.t.units = ur.seconds
d3D.coordset  # now return a list of coordinates
```

```{code-cell} ipython3
d3D.t
```

```{code-cell} ipython3
d3D.coordset("t")  # Alternative way to get a given coordinates
```

```{code-cell} ipython3
d3D["t"]  # another alternative way to get a given coordinates
```

The two other coordinates u and v are still undefined

```{code-cell} ipython3
d3D.u, d3D.v
```

When the dataset is printed, only the information for the existing coordinates is given.

```{code-cell} ipython3
d3D
```

Programmatically, we can use the attribute `is_empty` or `has_data` to check this

```{code-cell} ipython3
d3D.v.has_data, d3D.v.is_empty
```

An error is raised when a coordinate doesn't exist

```{code-cell} ipython3
try:
    d3D.x
except AttributeError as e:
    scp.error_(e)
```

In some case it can also be useful to get a coordinate from its long_name instead of its name (the limitation is that if
several coordinates have the same long_name, then only the first ones that is found in the coordinate list, will be
returned - this can be ambiguous)

```{code-cell} ipython3
d3D["time"]
```

```{code-cell} ipython3
d3D.time
```

## Labels

+++

It is possible to use labels instead of numerical coordinates. They are sequence of objects .The length of the
sequence must be equal to the size of a dimension.

+++

The labels can be simple strings, *e.g.,*

```{code-cell} ipython3
tags = list("ab")
d3D.u.long_name = "some tags"
d3D.u.labels = tags  # TODO: avoid repetition
d3D
```

or more complex objects.

For instance here we use timedelta objects:

```{code-cell} ipython3
from numpy import timedelta64

start = timedelta64(0)
times = [start + timedelta64(x * 60, "s") for x in range(6)]
d3D.t = None
d3D.t.labels = times
d3D.t.long_name = "time"
d3D
```

In this case, getting a coordinate that doesn't possess numerical data but labels, will return the labels

```{code-cell} ipython3
d3D.time
```

# More insight on coordinates

+++

## Sharing coordinates between dimensions

+++

Sometimes it is not necessary to have different coordinates for each axe. Some can be shared between axes.

For example, if we have a square matrix with the same coordinate in the two dimensions, the second dimension can
refer to the first. Here we create a square 2D dataset, using the `diag` method:

```{code-cell} ipython3
nd = NDDataset.diag((3, 3, 2.5))
nd
```

and then we add the same coordinate for both dimensions

```{code-cell} ipython3
coordx = Coord.arange(3)
nd.set_coordset(x=coordx, y="x")
nd
```

## Setting coordinates using `set_coordset`

+++

Let's create 3 `Coord` objects to be used as coordinates for the 3 dimensions of the previous d3D dataset.

```{code-cell} ipython3
d3D.dims = ["t", "v", "u"]
s0, s1, s2 = d3D.shape
coord0 = Coord.linspace(10.0, 100.0, s0, units="m", long_name="distance")
coord1 = Coord.linspace(20.0, 25.0, s1, units="K", long_name="temperature")
coord2 = Coord.linspace(0.0, 1000.0, s2, units="hour", long_name="elapsed time")
```

### Syntax 1

```{code-cell} ipython3
d3D.set_coordset(u=coord2, v=coord1, t=coord0)
d3D
```

### Syntax 2

```{code-cell} ipython3
d3D.set_coordset({"u": coord2, "v": coord1, "t": coord0})
d3D
```

## Adding several coordinates to a single dimension
We can add several coordinates to the same dimension

```{code-cell} ipython3
coord1b = Coord([1, 2, 3, 4], units="millitesla", long_name="magnetic field")
```

```{code-cell} ipython3
d3D.set_coordset(u=coord2, v=[coord1, coord1b], t=coord0)
d3D
```

We can retrieve the various coordinates for a single dimension easily:

```{code-cell} ipython3
d3D.v_1
```

## Summary of the coordinate setting syntax
Some additional information about coordinate setting syntax

+++

**A.** First syntax (probably the safer because the name of the dimension is specified, so this is less prone to
errors!)

```{code-cell} ipython3
d3D.set_coordset(u=coord2, v=[coord1, coord1b], t=coord0)
# or equivalent
d3D.set_coordset(u=coord2, v=CoordSet(coord1, coord1b), t=coord0)
d3D
```

**B.** Second syntax assuming the coordinates are given in the order of the dimensions.

Remember that we can check this order using the `dims` attribute of a NDDataset

```{code-cell} ipython3
d3D.dims
```

```{code-cell} ipython3
d3D.set_coordset((coord0, [coord1, coord1b], coord2))
# or equivalent
d3D.set_coordset(coord0, CoordSet(coord1, coord1b), coord2)
d3D
```

**C.** Third syntax (from a dictionary)

```{code-cell} ipython3
d3D.set_coordset({"t": coord0, "u": coord2, "v": [coord1, coord1b]})
d3D
```

**D.** It is also possible to use directly the `coordset` property

```{code-cell} ipython3
d3D.coordset = coord0, [coord1, coord1b], coord2
d3D
```

```{code-cell} ipython3
d3D.coordset = {"t": coord0, "u": coord2, "v": [coord1, coord1b]}
d3D
```

```{code-cell} ipython3
d3D.coordset = CoordSet(t=coord0, u=coord2, v=[coord1, coord1b])
d3D
```

<div class='alert alert-warning'>
<b>WARNING</b>

Do not use list for setting multiples coordinates! use tuples
</div>

+++

This raise an error (list have another signification: it's used to set a "same dim" CoordSet see example A or B)

```{code-cell} ipython3
try:
    d3D.coordset = [coord0, coord1, coord2]
except scp.utils.exceptions.InvalidCoordinatesSizeError as e:
    scp.error_(e)
```

This works : it uses a tuple `()`, not a list `[]`

```{code-cell} ipython3
d3D.coordset = (
    coord0,
    coord1,
    coord2,
)  # equivalent to d3D.coordset = coord0, coord1, coord2
d3D
```

**E.** Setting the coordinates individually

+++

Either a single coordinate

```{code-cell} ipython3
d3D.u = coord2
d3D
```

or multiple coordinates for a single dimension

```{code-cell} ipython3
d3D.v = [coord1, coord1b]
d3D
```

or using a CoordSet object.

```{code-cell} ipython3
d3D.v = CoordSet(coord1, coord1b)
d3D
```

# Methods to create NDDataset

There are many ways to create `NDDataset` objects.

Let's first create 2 coordinate objects, for which we can define `labels` and `units`! Note the use of the function
`linspace`to generate the data.

```{code-cell} ipython3
c0 = Coord.linspace(
    start=4000.0, stop=1000.0, num=5, labels=None, units="cm^-1", long_name="wavenumber"
)
```

```{code-cell} ipython3
c1 = Coord.linspace(
    10.0, 40.0, 3, labels=["Cold", "RT", "Hot"], units="K", long_name="temperature"
)
```

The full coordset will be the following

```{code-cell} ipython3
cs = CoordSet(c0, c1)
cs
```

Now we will generate the full dataset, using a ``fromfunction`` method. All needed information are passed as
parameter of the NDDataset instance constructor.

+++

## Create a dataset from a function

```{code-cell} ipython3
def func(x, y, extra):
    return x * y / extra
```

```{code-cell} ipython3
ds = NDDataset.fromfunction(
    func,
    extra=100 * ur.cm ** -1,  # extra arguments passed to the function
    coordset=cs,
    name="mydataset",
    long_name="absorbance",
    units=None,
)  # when None, units will be determined from the function results

ds.comment = """Dataset example created for this tutorial.
It's a 2-D dataset"""

ds.author = "Blake & Mortimer"
ds
```

## Using numpy-like constructors of NDDatasets

```{code-cell} ipython3
dz = NDDataset.zeros(
    (5, 3), coordset=cs, units="meters", commment="Datasets with only zeros"
)
```

```{code-cell} ipython3
do = NDDataset.ones(
    (5, 3), coordset=cs, units="kilograms", comment="Datasets with only ones"
)
```

```{code-cell} ipython3
df = NDDataset.full(
    (5, 3),
    fill_value=1.25,
    coordset=cs,
    units="radians",
    comment="with only float=1.25",
)
df
```

As with numpy, it is also possible to take another dataset as a template:

```{code-cell} ipython3
df = NDDataset.full_like(d3D, dtype="int", fill_value=2)
df
```

```{code-cell} ipython3
nd = NDDataset.diag((3, 3, 2.5))
nd
```

## Copying existing NDDataset

To copy an existing dataset, this is as simple as:

```{code-cell} ipython3
d3D_copy = d3D.copy()
```

or alternatively:

```{code-cell} ipython3
d3D_copy = d3D[:]
```

Finally, it is also possible to initialize a dataset using an existing one:

```{code-cell} ipython3
d3Dduplicate = NDDataset(d3D, name="duplicate of %s" % d3D.name, units="absorbance")
d3Dduplicate
```

## Importing from external dataset

NDDataset can be created from the importation of external data

A **test**'s data folder contains some data for experimenting some features of datasets.

```{code-cell} ipython3
# let check if this directory exists and display its actual content:
datadir = scp.preferences.datadir
if datadir.exists():
    print(datadir.name)
```

Let's load grouped IR spectra acquired using OMNIC:

```{code-cell} ipython3
nd = NDDataset.read_omnic(datadir / "irdata/nh4y-activation.spg")
nd.preferences.reset()
_ = nd.plot()
```

Even if we do not specify the **datadir**, the application first look in the directory by default.

+++

Now, lets load a NMR dataset (in the Bruker format).

```{code-cell} ipython3
path = datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_2d"

# load the data directly (no need to create the dataset first)
nd2 = NDDataset.read_topspin(path, expno=1, remove_digital_filter=True)

# view it...
nd2.x.to("s")
nd2.y.to("ms")

ax = nd2.plot(method="map")
```
