# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""
In this module, we define basic functions adapted from numpy but able to handle
our NDDataset objects

"""

# ----------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------

import numpy as np
from numpy.ma import nomask

# ----------------------------------------------------------------------------
# localimports
# ----------------------------------------------------------------------------

__all__ = ['diag', 'dot']


def dot(a, b, strict=True, out=None):
    """
    Return the dot product of two NDDatasets.

    This function is the equivalent of `numpy.dot` that takes NDDataset for
    input.

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

        .. versionadded:: 1.10.2

    See Also
    --------
    numpy.dot : Equivalent function for ndarrays.
    numpy.ma.dot : Equivalent function for masked ndarrays

    """
    if not a.implements('NDDataset'):
        raise TypeError('A dataset of type NDDataset is  '
                        'expected as a source of data, but an object'
                        ' of type {} has been provided'.format(
            type(a).__name__))

    if not b.implements('NDDataset'):
        raise TypeError('A dataset of type NDDataset is  '
                        'expected as a source of data, but an object'
                        ' of type {} has been provided'.format(
            type(b).__name__))

    #TODO: may be we can be less strict, and allow dot products with
    # different kind of objects, as far they are numpy-like arrays

    data = np.ma.dot(a.masked_data, b.masked_data)
    mask = data.mask
    data = data.data
    uncertainty = None
    if a.is_uncertain or b.is_uncertain:
            raise NotImplementedError('uncertainty not yet implemented')

    coordset = None
    if a.coordset is not None:
        coordset = [a.coordset[0]]
    if b.coordset is not None:
        if coordset is None:
            coordset = [None]
        coordset.append(b.coordset[1])
    elif coordset is not None:
        coordset.append(None)

    history = 'dot product between %s and %s'%(a.name, b.name)

    # make the output
    # ---------------
    new = a.copy()
    new._data = data
    new._mask = mask
    new._uncertainty = uncertainty
    new._coordset =  type(new.coordset)(coordset)
    new.history = history

    return new

def diag(source, k=0):
    """
    Extract a diagonal or construct a diagonal array.

    See the more detailed documentation for ``numpy.diagonal`` if you use this
    function to extract a diagonal and wish to write to the resulting array;
    whether it returns a copy or a view depends on what version of numpy you
    are using.

    Parameters
    ----------
    v : array_like
        If `v` is a 2-D array, return a copy of its `k`-th diagonal.
        If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
        diagonal.
    k : int, optional
        Diagonal in question. The default is 0. Use `k>0` for diagonals
        above the main diagonal, and `k<0` for diagonals below the main
        diagonal.

    Returns
    -------
    out : ndarray
        The extracted diagonal or constructed diagonal array.

    copied from numpy (licence
    """

    # check if we have the correct input
    # ----------------------------------

    if not source.implements('NDDataset'):
        raise TypeError('A dataset of type NDDataset is  '
                        'expected as a source of data, but an object'
                        ' of type {} has been provided'.format(
            type(source).__name__))

    s = source.data.shape

    if len(s) == 1:
        # construct a diagonal array
        # --------------------------
        data = np.diag(source.data)
        mask = nomask
        if source.is_masked:
            size = source.size
            m = np.repeat(source.mask, size).reshape(size, size)
            mask = m | m.T
        uncertainty = None
        if source.is_uncertain:
            uncertainty = np.diag(source.uncertainty)
        coordset = None
        if source.coordset is not None:
            coordset = [source.coordset[0]]*2
        history = 'diagonal array build from the 1D source'

    elif len(s) == 2:
        # extract a diagonal
        # ------------------
        data = np.diagonal(source.data, k).copy()
        mask = None
        if source.is_masked:
            mask = np.diagonal(source.mask, k).copy()
        uncertainty = None
        if source.is_uncertain:
            uncertainty = np.diagonal(source.uncertainty, k).copy()
        coordset = None
        if source.coordset is not None:
            coordset = [source.coordset[0]]  # TODO: this is likely not
                                             #       correct for k != 0
        history = 'diagonal of rank %d extracted from original source'%k

    else:
        raise ValueError("Input must be 1- or 2-d.")

    # make the output
    # ---------------
    new = source.copy()
    new._data = data
    new._mask = mask
    new._uncertainty = uncertainty
    new.coordset = coordset
    new.history = history

    return new



# =============================================================================
if __name__ == '__main__':
    pass
