.. _api_reference:

.. currentmodule:: spectrochempy


##############
API reference
##############

****************
Loading the API
****************

The |scpy| API exposes many objects and functions.

To use the API, you must import it using one of the following syntax:

.. ipython:: python

    import spectrochempy as scp
    nd = scp.NDDataset()

.. ipython:: python

    from spectrochempy import *
    nd = NDDataset()

With the second syntax, as often in python, the access to objects/functions
can be greatly simplified. For example, we can use "NDDataset" without a prefix
instead of `scp.NDDataset` which is the first syntax) but there is always a risk
of overwriting some variables or functions already present in the namespace.
Therefore, the first syntax is generally highly recommended.


The NDDataset object
***

The NDDataset is the main object use by |scpy|.

Like numpy ndarrays, NDDataset have the capability to be sliced,
sorted and subject to mathematical operations.

But, in addition, NDDataset may have units, can be masked and each
dimensions can have coordinates also with units. This make NDDataset
aware of unit compatibility, *e.g.*, for binary operation such as
additions or subtraction or during the application of mathematical
operations. In addition or in replacement of numerical data for
coordinates, NDDataset can also have labeled coordinates where labels
can be different kind of objects (strings, datetime, numpy nd.ndarray
or othe NDDatasets, etc...).

This offers a lot of flexibility in using NDDatasets that, we hope,
will be useful for applications. See the **Tutorials** for more
information about such possible applications.

**SpectroChemPy** provides another kind of data structure,
aggregating several datasets: **NDPanel**: See

.. autosummary::
    :nosignatures:
    :toctree: generated/

    NDDataset


Coordinates related objects
---------------------------

NDDataset in SpectroChemPy in contrast to numpy nd-arrays can have coordinates for each dimension.
The individual coordinates are represented by a specific object: Coord.
All coordinates of an NDDataset are grouped in a particular object: CoordSet.
Finally, a range of coordinates can be represented by the object: CoordRange.

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
    zeros_like
    ones
    ones_like
    full
    full_like
    eye
    identity

Import of data from external sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: generated/

    read_omnic
    read_spg
    read_spa
    read_srs
    read_opus
    read_topspin
    read_zip
    read_csv
    read_jcamp
    read_matlab
    read_dir
    read_carroucell


Export a NDDataset
-------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    write_jcamp


Select data in a NDDataset
--------------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    take
    clip


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
-----------

Transformations
~~~~~~~~~~~~~~~~

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



Unary mathematical operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Binary mathematical operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: generated/

    dot

Other processing operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: generated/

    BaselineCorrection
    align
    autosub
    ab
    em
    gm
    sp
    sine
    qsin
    sinm
    concatenate
    stack
    fft
    ifft
    detrend
    savgol_filter
    interpolate
    smooth



Project management
--------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    Project

Scripting
----------

This is rather experimental

.. autosummary::
    :nosignatures:
    :toctree: generated/

    Script


Utilities
----------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    download_IRIS
