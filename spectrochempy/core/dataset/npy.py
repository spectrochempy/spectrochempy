# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""
In this module, we define basic functions adapted from numpy but able to handle
our NDDataset objects

"""
__all__ = ['diag', 'dot']

# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np

from spectrochempy.core.dataset.nddataset import NDDataset
# from spectrochempy.core.dataset.ndcoordset import CoordSet
# from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.utils import NOMASK, make_new_object


# ............................................................................
def dot(a, b, strict=True, out=None):
    """
    Return the dot product of two NDDatasets.

    This function is the equivalent of `numpy.dot` that takes NDDataset as input

    .. note::
      Works only with 2-D arrays at the moment.

    Parameters
    ----------
    a, b : masked_array_like
        Inputs arrays.
    strict : bool, optional
        Whether masked data are propagated (True) or set to 0 (False) for
        the computation. Default is False.  Propagating the mask means that
        if a masked value appears in a row or column, the whole row or
        column is considered masked.
    out : masked_array, optional
        Output argument. This must have the exact kind that would be returned
        if it was not used. In particular, it must have the right type, must be
        C-contiguous, and its dtype must be the dtype that would be returned
        for `dot(a,b)`. This is a performance feature. Therefore, if these
        conditions are not met, an exception is raised, instead of attempting
        to be flexible.

    See Also
    --------
    numpy.dot : Equivalent function for ndarrays.
    numpy.ma.dot : Equivalent function for masked ndarrays

    """
    # if not a.implements('NDDataset'):
    #     raise TypeError('A dataset of type NDDataset is  '
    #                     'expected as a source of data, but an object'
    #                     ' of type {} has been provided'.format(
    #         type(a).__name__))
    #
    # if not b.implements('NDDataset'):
    #     raise TypeError('A dataset of type NDDataset is  '
    #                     'expected as a source of data, but an object'
    #                     ' of type {} has been provided'.format(
    #         type(b).__name__))

    # TODO: may be we can be less strict, and allow dot products with
    #      different kind of objects, as far they are numpy-like arrays

    if not isinstance(a, NDDataset) and not isinstance(a, NDDataset):
        # must be between numpy object or something non valid. Let numpy deal with this
        return np.dot(a, b)

    if not isinstance(a, NDDataset):
        # try to cast to NDDataset
        a = NDDataset(a)

    if not isinstance(b, NDDataset):
        # try to cast to NDDataset
        b = NDDataset(b)

    data = np.ma.dot(a.masked_data, b.masked_data)
    mask = data.mask
    data = data.data

    if a.coordset is not None:
        coordy = getattr(a, a.dims[0])
    else:
        coordy = None
    if b.coordset is not None:
        coordx = getattr(b, b.dims[1])
    else:
        coordx = None

    history = 'Dot product between %s and %s' % (a.name, b.name)

    # make the output
    # ------------------------------------------------------------------------------------------------------------------
    new = make_new_object(a)
    new._data = data
    new._mask = mask
    new.set_coordset(y=coordy, x=coordx)
    new.history = history
    if a.unitless:
        new.units = b.units
    elif b.unitless:
        new.units = a.units
    else:
        new.units = a.units * b.units

    return new


# ............................................................................
def diag(dataset, k=0):
    """
    Extract a diagonal or construct a diagonal array.

    See the more detailed documentation for ``numpy.diagonal`` if you use this
    function to extract a diagonal and wish to write to the resulting array;
    whether it returns a copy or a view depends on what version of numpy you
    are using.

    Adapted from numpy (licence #TO ADD)

    Parameters
    ----------
    v : array_like
        If `v` is a 2-D array, return a copy of its `k`-th diagonal.
        If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
        diagonal.

    Returns
    -------
    out : ndarray
        The extracted diagonal or constructed diagonal array.

    """
    # TODO: fix this - other diagonals
    # k : int, optional
    # Diagonal in question. The default is 0. Use `k>0` for diagonals
    # above the main diagonal, and `k<0` for diagonals below the main
    # diagonal.

    # check if we have the correct input
    # ------------------------------------------------------------------------------------------------------------------

    if not isinstance(dataset, NDDataset):
        # must be a numpy object or something non valid. Let numpy deal with this
        return np.diag(dataset)

    s = dataset.data.shape

    if len(s) == 1:
        # construct a diagonal array
        # --------------------------
        data = np.diag(dataset.data)
        mask = NOMASK
        if dataset.is_masked:
            size = dataset.size
            m = np.repeat(dataset.mask, size).reshape(size, size)
            mask = m | m.T
        coords = None
        if dataset.coordset is not None:
            coords = dataset.coordset
        history = 'Diagonal array build from the 1D dataset'
        units = dataset.units
        dims = dataset.dims * 2

    elif len(s) == 2:
        # extract a diagonal
        # ------------------
        data = np.diagonal(dataset.data, k).copy()
        mask = NOMASK
        if dataset.is_masked:
            mask = np.diagonal(dataset.mask, k).copy()
        coords = None
        if dataset.coordset is not None:
            coords = [dataset.coordset[0]]  # TODO: this is likely not
            #       correct for k != 0
        history = 'Diagonal of rank %d extracted from original dataset' % k
        units = dataset.units
        dims = dataset.dims[-1]

    else:
        raise ValueError("Input must be 1- or 2-d.")

    # make the output
    # ------------------------------------------------------------------------------------------------------------------
    new = dataset.copy()
    new._data = data
    new._mask = mask
    new.history = history
    new.units = units
    new.dims = dims

    if coords:
        new.set_coordset(coords)

    return new


# ======================================================================================================================
if __name__ == '__main__':
    pass
