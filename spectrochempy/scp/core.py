# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================
"""
This module adds numpy-like methods to the main |scp| API.

"""
__all__ = ['empty', 'empty_like', 'zeros', 'zeros_like', 'ones',
           'ones_like', 'full', 'full_like']

import numpy as np
from spectrochempy.dataset.nddataset import NDDataset

def empty(shape, dtype=None, **kwargs):
    """
    Return a new |NDDataset| of given shape and type,  without initializing
    entries.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array
    dtype : data-type, optional
        Desired output data-type.

    Returns
    -------
    out : |NDDataset|
        Array of uninitialized (arbitrary) data of the given shape, dtype, and
        order.  Object arrays will be initialized to None.

    See Also
    --------
    empty_like, zeros, ones

    Notes
    -----
    `empty`, unlike `zeros`, does not set the array values to zero,
    and may therefore be marginally faster.  On the other hand, it requires
    the user to manually set all the values in the array, and should be
    used with caution.

    Examples
    --------
    >>> from spectrochempy import scp # doctest: +ELLIPSIS
    SpectroChemPy's API...

    >>> scp.empty([2, 2]) # doctest: +ELLIPSIS
    NDDataset: [[...]] unitless

    >>> scp.empty([2, 2], dtype=int, units='s') # doctest: +ELLIPSIS
    NDDataset: [[...]] s

    """
    return NDDataset(np.empty(shape, dtype=dtype), **kwargs)

def empty_like(a, dtype=None):
    """
    Return a new array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of the
        returned array.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data with the same
        shape and type as `a`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.

    Notes
    -----
    This function does *not* initialize the returned array; to do that use
    for instance `zeros_like`, `ones_like` or `full_like` instead.  It may be
    marginally faster than the functions that do set the array values.

    """
    new = a.copy()
    return new

def zeros(shape, dtype=None, **kwargs):
    """
    Return a new |NDDataset| of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    **kwargs : keyword args to pass to the |NDDataset| constructor

    Returns
    -------
    out : |NDDataset|
        Array of zeros with the given shape, dtype.

    See Also
    --------
    ones, zeros_like

    Examples
    --------
    >>> from spectrochempy import scp
    >>> scp.zeros(5)
    NDDataset: [   0.000,    0.000,    0.000,    0.000,    0.000] unitless

    >>> scp.zeros((5,), dtype=np.int)
    NDDataset: [       0,        0,        0,        0,        0] unitless

    >>> s = (2,2)
    >>> scp.zeros(s, units='m')
    NDDataset: [[   0.000,    0.000],
                [   0.000,    0.000]] m

    """
    return NDDataset(np.zeros(shape, dtype=dtype), **kwargs)


def ones(shape, dtype=None, **kwargs):
    """
    Return a new |NDDataset| of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    **kwargs : keyword args to pass to the |NDDataset| constructor

    Returns
    -------
    out : |NDDataset|
        Array of ones with the given shape, dtype.

    See Also
    --------
    zeros, ones_like

    Examples
    --------
    >>> from spectrochempy import scp
    >>> scp.ones(5, units='km')
    NDDataset: [   1.000,    1.000,    1.000,    1.000,    1.000] km

    >>> scp.ones((5,), dtype=np.int, mask=[True, False, False, False, True])
    NDDataset: [  --,        1,        1,        1,   --] unitless

    >>> scp.ones((2, 2))
    NDDataset: [[   1.000,    1.000],
                [   1.000,    1.000]] unitless

    """
    return NDDataset(np.ones(shape, dtype=dtype), **kwargs)


def zeros_like(a, dtype=None,):
    """
    Return a |NDDataset| of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : |NDDataset|
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    out : |NDDataset|
        Array of zeros with the same shape and type as `a`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.

    Examples
    --------
    >>> from spectrochempy import scp
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x = NDDataset(x, units='s')
    >>> x
    NDDataset: [[       0,        1,        2],
                [       3,        4,        5]] s
    >>> scp.zeros_like(x)
    NDDataset: [[       0,        0,        0],
                [       0,        0,        0]] s


    """
    new = a.copy()
    new.data = np.zeros_like(a)
    return new


def ones_like(a, dtype=None):
    """
    Return |NDDataset| of ones with the same shape and type as a given array.

    It preserves original mask, units, coords and uncertainty

    Parameters
    ----------
    a : |NDDataset|
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    out : |NDDataset|
        Array of ones with the same shape and type as `a`.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.

    Examples
    --------
    >>> from spectrochempy import scp
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x = NDDataset(x, units='s')
    >>> x
    NDDataset: [[       0,        1,        2],
                [       3,        4,        5]] s
    >>> scp.ones_like(x)
    NDDataset: [[       1,        1,        1],
                [       1,        1,        1]] s

    """
    new = a.copy()
    new.data = np.ones_like(a)
    return new

def full(shape, fill_value, dtype=None, **kwargs):
    """
    Return a new |NDDataset| of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `np.int8`.  Default
        is `float`, but will change to `np.array(fill_value).dtype` in a
        future release.
    **kwargs : keyword args to pass to the |NDDataset| constructor

    Returns
    -------
    out : |NDDataset|
        Array of `fill_value` with the given shape, dtype, and order.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    full_like : Fill an array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.

    Examples
    --------
    >>> from spectrochempy import scp
    >>> scp.full((2, 2), np.inf)
    NDDataset: [[     inf,      inf],
                [     inf,      inf]] unitless
    >>> scp.full((2, 2), 10, dtype=np.int)
    NDDataset: [[      10,       10],
                [      10,       10]] unitless

    """
    return NDDataset(np.full(shape, fill_value=fill_value, dtype=dtype),
                     **kwargs)

def full_like(a, fill_value, dtype=None):
    """
    Return a |NDDataset| with the same shape and type as a given array.

    Parameters
    ----------
    a : |NDDataset| or array-like
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    array-like
        Array of `fill_value` with the same shape and type as `a`.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.
    full : Fill a new array.

    Examples
    --------
    >>> from spectrochempy import scp

    >>> x = np.arange(6, dtype=int)
    >>> scp.full_like(x, 1)
    array([       1,        1,        1,        1,        1,        1])

    >>> x = NDDataset(x, units='m')
    >>> scp.full_like(x, 0.1)
    NDDataset: [       0,        0,        0,        0,        0,        0] m
    >>> scp.full_like(x, 0.1, dtype=np.double)
    NDDataset: [   0.100,    0.100,    0.100,    0.100,    0.100,    0.100] m
    >>> scp.full_like(x, np.nan, dtype=np.double)
    NDDataset: [     nan,      nan,      nan,      nan,      nan,      nan] m

    """
    new = a.copy()
    new.data = np.full_like(a, fill_value=fill_value, dtype=dtype)
    return new






# =============================================================================
if __name__ == '__main__':

    pass
