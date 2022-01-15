.. _api_reference:

.. currentmodule:: spectrochempy


####################
Public API reference
####################

The |scpy| API publicly exposes many objects and functions.
They are listed below exhaustively.
What is not listed here is reserved for developers and should not normally be necessary for normal use of |scpy|.

.. contents:: Table of Contents

****************
Loading the API
****************

To use the API, you must import it using one of the following syntax:

.. ipython:: python

    import spectrochempy as scp  # Recommended!
    nd = scp.NDDataset()

.. ipython:: python

    from spectrochempy import *
    nd = NDDataset()

With the second syntax, as often in python, the access to objects/functions can be greatly simplified. For example,
we can use `NDDataset` without a prefix instead of `scp.NDDataset` which is the first syntax) but there is always a risk
of overwriting some variables or functions already present in the namespace. Therefore, the first syntax is generally
highly recommended.


*********************
The NDDataset Object
*********************

The NDDataset is the main object use by |scpy|.

Like numpy ndarrays, NDDataset have the capability to be sliced, sorted and subject to mathematical operations.

But, in addition, NDDataset may have units, can be masked and each dimension can also have coordinated with units.
This make NDDataset aware of unit compatibility, *e.g.,*, for binary operation such as additions or subtraction or
during the application of mathematical operations. In addition or in replacement of numerical data for coordinates,
NDDataset can also have labeled coordinates where labels can be different kinds of objects (strings, datetime, numpy
nd.ndarray or other NDDatasets, etc...).

This offers a lot of flexibility in using NDDatasets that, we hope, will be useful for applications. See the
**Tutorials** for more information about such possible applications.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    NDDataset


***************************
Coordinates-related objects
***************************

|NDDataset| in |scpy| in contrast to numpy nd-arrays can have coordinates for each dimension.
The individual coordinates are represented by a specific object: |Coord|.
All coordinates of a |NDDataset| are grouped in a particular object: |CoordSet|.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    Coord
    LinearCoord
    CoordSet


*******************
Creating NDDataset
*******************

A |NDDataset| can be created using the |NDDataset| class constructor, for instance here we create a dataset from a
random two-dimensional array:

.. ipython:: python

    import numpy as np
    X = np.random.random((4,4))
    nd = NDDataset(X)

The above code in |scpy| can be simplified using the ``random`` creation method:

.. ipython:: python

    X = NDDataset.random((4,4))


(see the :ref:`userguide` for a large set of examples on how to use this constructor.)

Many SpectroChemPy methods mimics numpy equivalent, but output a NDDataset object.


Basic creation methods
=======================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    empty
    zeros
    ones
    full
    empty_like
    zeros_like
    ones_like
    full_like
    eye
    identity
    random
    diag


Creation from existing data
============================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    copy
    fromfunction
    fromiter


Creation from numerical ranges
==============================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    arange
    linspace
    logspace
    geomspace


Creation from from external sources
====================================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    read
    read_omnic
    read_spg
    read_spa
    read_srs
    read_opus
    read_labspec
    read_topspin
    read_zip
    read_csv
    read_jcamp
    read_matlab
    read_quadera
    read_dir
    read_carroucell


******************
Export a NDDataset
******************

.. autosummary::
    :nosignatures:
    :toctree: generated/

    NDDataset.save
    NDDataset.save_as
    load
    write
    write_jcamp
    to_array
    to_xarray



**************************
Select data in a NDDataset
**************************

.. autosummary::
    :nosignatures:
    :toctree: generated/

    take


********************
Plotting functions
********************

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
    plot_waterfall


************
Processing
************

Transpose-like oprations
========================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    transpose
    NDDataset.T
    swapdims


Changing number of dimensions
=============================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    squeeze


Changing type
==============

.. autosummary::
    :nosignatures:
    :toctree: generated/

    set_complex
    set_hypercomplex
    set_quaternion


Joining or splitting datasets
=============================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    concatenate
    stack


Indexing
========

.. autosummary::
    :nosignatures:
    :toctree: generated/

    diag
    diagonal
    take


Sorting
=======

.. autosummary::
    :nosignatures:
    :toctree: generated/

    sort


Minimum and maximum
===================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    argmin
    argmax
    coordmin
    coordmax
    min
    max
    ptp


Clipping and rounding
=====================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    clip
    rint
    floor
    ceil
    fix
    trunc
    around
    round


Algebra
========

.. autosummary::
    :nosignatures:
    :toctree: generated/

    dot
    SVD
    LSTSQ
    NNLS


Logic functions
================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    all
    any
    sign


Sums, integal, difference
==========================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    sum
    cumsum
    trapezoid
    simpson


Complex
=======

.. autosummary::
    :nosignatures:
    :toctree: generated/

    NDDataset.real
    NDDataset.imag
    NDDataset.RR
    NDDataset.RI
    NDDataset.IR
    NDDataset.II
    conj
    conjugate
    abs
    absolute


Masks
=====

.. autosummary::
    :nosignatures:
    :toctree: generated/

    remove_masks


Units manipulation
===================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    to
    to_base_units
    to_reduced_units
    ito
    ito_base_units
    ito_reduced_units
    is_units_compatible
    deg2rad
    rad2deg


Unary mathematical operations
==============================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    fabs
    negative
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


Statistical operations
=======================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    mean
    average
    std
    sum
    var


Baseline correction
====================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    BaselineCorrection
    autosub
    ab
    abc
    basc
    detrend


Fourier transform
==================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    fft
    ifft


Smoothing, apodization
=======================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    savgol_filter
    smooth
    bartlett
    blackmanharris
    hamming
    triang
    em
    gm
    sp
    sine
    qsin
    sinm


Alignment, interpolation
========================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    align
    interpolate


Miscellaneous
=============

.. autosummary::
    :nosignatures:
    :toctree: generated/

    pipe


********************
Fitting
********************

.. autosummary::
    :nosignatures:
    :toctree: generated/

    Fit
    CurveFit
    LSTSQ
    NNLS


*******************
Analysis
*******************

.. autosummary::
    :nosignatures:
    :toctree: generated/

    find_peaks
    EFA
    PCA
    NNMF
    SIMPLISMA
    MCRALS
    IRIS


********************
Project management
********************

.. autosummary::
    :nosignatures:
    :toctree: generated/

    Project


**********
Scripting
**********

This is rather experimental


.. autosummary::
    :nosignatures:
    :toctree: generated/

    Script


**********
Utilities
**********

.. autosummary::
    :nosignatures:
    :toctree: generated/

    download_iris
