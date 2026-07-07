.. _api_reference:

.. currentmodule:: spectrochempy


####################
Public API reference
####################

The `SpectroChemPy` :term:`API` publicly exposes many objects and functions.
They are listed below exhaustively.
What is not listed here is reserved for developers and should not normally be necessary for normal use of `SpectroChemPy` .

.. contents:: Table of Contents

***************
Loading the API
***************

To use the API, you must import it using one of the following syntax:

.. ipython:: python

    import spectrochempy as scp  # Recommended!
    nd = scp.NDDataset()

.. ipython:: python

    from spectrochempy import NDDataset
    from spectrochempy import *  # strongly discouraged
    nd = NDDataset()

With the second syntax, as often in python, the access to objects/functions can be greatly simplified. For example,
we can use `NDDataset` without a prefix instead of `scp.NDDataset` which is the first syntax) but there is always a risk
of overwriting some variables or functions already present in the namespace. Therefore, the first syntax is generally
highly recommended.

However instead of the second syntax, one can always use the following way to import objects or functions:

.. ipython:: python

    from spectrochempy import NDDataset
    nd = NDDataset()


********************
The NDDataset Object
********************

The `NDDataset` is the main object use by `SpectroChemPy` .

Like `numpy.ndarray`s, `NDDataset` have the capability to be sliced, sorted and subject to mathematical operations.

But, in addition, `NDDataset` may have units, can be masked and each dimension can also have coordinates with units.
This make `NDDataset` aware of unit compatibility, *e.g.,*, for binary operation such as additions or subtraction or
during the application of mathematical operations. In addition or in replacement of numerical data for coordinates,
NDDataset can also have labeled coordinates where labels can be different kinds of objects (strings, datetime, numpy
nd.ndarray or other NDDatasets, etc...).

This offers a lot of flexibility in using NDDatasets that, we hope, will be useful for applications. See the
:ref:`user_guide` for more information about such possible applications.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    NDDataset


Coordinates-related objects
===========================

`NDDataset` in `SpectroChemPy` in contrast to numpy nd-arrays can have coordinates for each dimension.
The individual coordinates are represented by a specific object: `Coord`.
All coordinates of a `NDDataset` are grouped in a particular object: `CoordSet`.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    Coord
    CoordSet


******************
Creating NDDataset
******************

A `NDDataset` can be created using the `NDDataset` class constructor, for instance here we create a dataset from a
`~numpy.random.random` two-dimensional array:

.. ipython:: python

    import numpy as np
    X = np.random.random((4,4))
    nd = NDDataset(X)

The above code in `SpectroChemPy` can be simplified using the `random` creation method:

.. ipython:: python

    X = NDDataset.random((4,4))


(see the :ref:`user_guide` for a large set of examples on how to use this constructor.)

Many SpectroChemPy methods mimic `numpy` equivalent, but output a `NDDataset` object.


Basic creation methods
======================

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
    normal
    diag


Creation from existing data
===========================

.. autosummary::
    :nosignatures:
    :toctree: generated/

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

Select data in a NDDataset
==========================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    take

*************
Import/export
*************

Import a NDDataset from external source
======================================

Reader functions may return either a single ``NDDataset`` or a list-like
``ScpObjectList`` when several datasets are discovered and not merged. In that
case, helper methods such as ``datasets.names``,
``datasets.select_largest(ndim=2)``, and ``datasets.select_by_name("spectra")``
make it easier to pick the dataset you want while staying in the public API.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    load
    read
    read_csv
    read_ddr
    read_dir
    read_hdr
    read_jcamp
    read_labspec
    read_wire
    read_wdf
    read_mat
    read_matlab
    read_omnic
    read_opus
    read_quadera
    read_sdr
    read_soc
    read_spa
    read_spc
    read_spg
    read_srs
    NDDataset.from_xarray
    NDDataset.from_netcdf
    load_iris
    download_nist_ir

Export a NDDataset
==================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    NDDataset.save
    NDDataset.save_as
    write
    write_csv
    write_jcamp
    write_mat
    write_matlab
    to_array
    to_netcdf
    to_xarray


********
Plotting
********

.. autosummary::
    :nosignatures:
    :toctree: generated/

    plot
    plot_1D
    plot_pen
    plot_scatter
    plot_scatter_pen
    plot_with_transposed
    plot_bar
    plot_2D
    plot_contour
    plot_contourf
    plot_image
    plot_lines
    plot_map
    plot_stack
    plot_3D
    plot_surface
    plot_waterfall
    plot_multiple
    multiplot
    multiplot_contour
    multiplot_contourf
    multiplot_image
    multiplot_lines
    multiplot_map
    multiplot_scatter
    multiplot_stack
    multiplot_with_transposed
    show


**********
Processing
**********

Transpose-like operations
=========================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    transpose
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


Joining or splitting datasets
=============================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    concatenate
    stack


Chemometric preprocessing
=========================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    normalize
    center
    autoscale
    snv
    msc
    pareto_scale
    range_scale
    robust_scale
    log_transform
    CenterTransformer
    AutoscaleTransformer
    SNVTransformer
    NormalizeTransformer
    MSCTransformer
    ParetoScaleTransformer
    RangeScaleTransformer
    RobustScaleTransformer
    LogTransformer


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
    amin
    amax
    min
    max
    ptp


Clipping and rounding
=====================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    clip
    around
    round


Algebra
=======

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


Sums, integral, difference
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

    Unit
    Quantity
    to
    to_base_units
    to_reduced_units
    ito
    ito_base_units
    ito_reduced_units
    is_units_compatible


Mathematical operations
=======================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    mc
    ps


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

    Baseline
    autosub
    get_baseline
    basc
    detrend
    asls
    snip


Fourier transform
==================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    fft
    ifft
    ht
    fsh
    fsh2


Phasing
=======

.. autosummary::
    :nosignatures:
    :toctree: generated/

    pk
    pk_exp


Time-domain processing
======================

Offset correction
-----------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    dc


Zero-filling
------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    zf
    zf_auto
    zf_double
    zf_size


Rolling
-------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    cs
    ls
    roll
    rs


Apodization
-----------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    bartlett
    blackmanharris
    hamming
    general_hamming
    hann
    triang
    em
    gm
    sp
    sine
    qsin
    sinm


Smoothing, filtering, denoising
===============================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    Filter
    savgol
    smooth
    whittaker
    denoise
    despike


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


********
Analysis
********


Linear regression
=================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    LSTSQ
    NNLS


Non-linear optimization and curve fit
=====================================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    Optimize

Partial Least Square regression
===============================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    PLSRegression

Evolving factor analysis
========================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    EFA

Multivariate Curve Resolution - Alternating Least Squares
=========================================================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    MCRALS

MCRALS Constraints
==================

These public classes describe scientific prior knowledge about the
concentration (``"C"``) or spectral (``"St"``) profiles estimated by
:class:`MCRALS`. They form the foundation of the future constraint-based
API for ``MCRALS`` and are currently declarative containers and
validators only — they are not yet connected to the internal ALS
constraint engine. See the project RFC for the planned migration path.

Import them from the :mod:`spectrochempy.analysis.constraints` submodule::

    from spectrochempy.analysis import constraints

    constraints.NonNegative("C")
    constraints.Closure("C")
    constraints.ReferenceProfile("St", component=0, data=reference)

.. autosummary::
    :nosignatures:
    :toctree: generated/

    analysis.constraints.Constraint
    analysis.constraints.Closure
    analysis.constraints.FixedValues
    analysis.constraints.Monotonic
    analysis.constraints.NonNegative
    analysis.constraints.ModelProfile
    analysis.constraints.ReferenceProfile
    analysis.constraints.Selectivity
    analysis.constraints.Unimodal
    analysis.constraints.ZeroRegion

Independent Component Analysis
==============================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    FastICA

Non-Negative Matrix Factorization
=================================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    NMF

Singular value decomposition and Principal component analysis
=============================================================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    PCA
    SVD

SIMPLe to use Interactive Self-modeling Mixture Analysis
========================================================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    SIMPLISMA

Phase Sensitive Detection
=========================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    PSD

Utilities
=========

Lineshape models
----------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    polynomial
    gaussian
    gaussianmodel
    lorentzian
    lorentzianmodel
    voigt
    voigtmodel
    asymmetricvoigt
    asymmetricvoigtmodel
    sigmoid
    sigmoidmodel
    polynomialbaseline

Find peaks
----------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    find_peaks

Kinetics
--------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    ActionMassKinetics
    PFR

********************
Project management
********************

.. autosummary::
    :nosignatures:
    :toctree: generated/

    Project

**********
Utilities
**********

Logging
=======

.. autosummary::
    :nosignatures:
    :toctree: generated/

    set_loglevel
    get_loglevel
    debug_
    info_
    warning_
    error_

Misc
====

.. autosummary::
    :nosignatures:
    :toctree: generated/

    show_versions

Multi-object files
==================

.. autosummary::
    :nosignatures:
    :toctree: generated/

    ScpObjectList

File
====

.. autosummary::
    :nosignatures:
    :toctree: generated/

    pathclean
