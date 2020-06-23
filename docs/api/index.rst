.. _api_loading:

.. currentmodule:: spectrochempy

API reference
===============

Loading the API
----------------

The |scpy| API exposes many objects and functions.

To use the API, one must load it using one of the following syntax and used as
follow:

.. ipython:: python

    import spectrochempy as scp
    nd = scp.NDDataset(...)

.. ipython:: python

    from spectrochempy import *
    nd = NDDataset(...)


In the second syntax, as usual in python, access to the objects/functions
may be simplified (*e.g.*, we can use `NDDataset` without any prefix  instead
of  `scp.NDDataset` is the first syntax)
but there is always a risk of overwriting some variables already in the
namespace. Therefore, the first syntax
is in general recommended, although that, for the examples in this
documentation, we have often use the
second one for simplicity.


The NDDataset object
---------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    NDDataset

Coordinates objects
--------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    Coord
    CoordSet
    CoordRange

Creating NDDataset
-------------------

A NDDataset can be created using the NDDataset class constructor,
for instance here we create a dataset from a random
two dimensional array:

.. ipython:: python

    import numpy as np
    X = np.random.random((4,4))
    nd = NDDataset(X)


(see the User Guide for examples on how to use this constructor.)


Creation using Numpy-like functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These functions mimics numpy equivalent, but output a NDDataset object

.. autosummary::
    :nosignatures:
    :toctree: generated/

    empty
    empty_like
    zeros
    eye
    identity
    zeros_like
    ones
    ones_like
    full
    full_like

Import of data from external sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: generated/

    read_bruker_nmr
    read_zip
    read_csv
    read_dir
    read_carroucell
    read_jdx
    read_matlab
    read_omnic
    read_spg
    read_spa
    read_opus

Export a NDDataset
-------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    to_xarray
    to_dataframe
    write_jdx

Coordinates manipulation
-------------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    add_coords
    coord
    delete_coords

Select data in a NDDataset
--------------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    take
    clip

Mathematical operations
-----------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    abs
    conjugate
    conj
    negative
    absolute
    fabs
    rint
    sign
    exp
    exp2
    log
    log2
    log10
    expm1
    log1p
    sqrt
    square
    cbrt
    reciprocal
    sin
    cos
    tan
    arcsin
    arccos
    arctan
    sinh
    cosh
    tanh
    arcsinh
    arccosh
    arctanh
    deg2rad
    rad2deg
    floor
    ceil
    trunc
    amax
    amin
    argmin
    argmax
    cumsum
    diag
    mean
    pipe
    ptp
    round
    std
    sum
    var
    dot

Plotting functions
-------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    plot
    plot_1D
    plot_pen
    plot_scatter
    plot_bar
    plot_2D
    plot_map
    plot_stack
    plot_image
    plot_surface
    plot_3D

Processing
----------

.. autosummary::
    :nosignatures:
    :toctree: generated/

Transformation and processing
------------------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    sort
    copy
    squeeze
    swapaxes
    transpose
    set_complex
    set_quaternion


Utilities
----------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    upload_IRIS





