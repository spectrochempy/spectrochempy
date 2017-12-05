# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to provide a general
# API for displaying, processing and analysing spectrochemical data.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
# =============================================================================



"""Plugin module to perform automatic subtraction of ref on a dataset.

"""

import numpy as np
from scipy.optimize import minimize_scalar

from spectrochempy.dataset.ndcoords import CoordRange

__all__ = ['autosub']


def autosub(source, ref, *ranges, axis=-1, method='vardiff', inplace=False):
    """Automatic subtraction of ref to the dataset to minimise peaks due to ref

    Parameters
    -----------
    source : :class:`~spectrochempy.dataset.nddataset.NDDataset`.
        Dataset to which we want to subtract the reference data

    ref : :class:`~spectrochempy.dataset.nddataset.NDDataset`.
         1D reference data, with a size maching the axis to subtract
         (axis parameter) #TODO : optionally use title of axis

    xrange : pair(s) of values. Any number of pairs is allowed.
        Coord range(s) in which the variance is minimized

    inplace : `bool`, optional, default = False.
        True if the subtraction is done in place.
        In this case we do not need to catch the function output

    axis : int [optional, default = -1].
        Tells on which axis to perform the subtraction

    method  : str [optional, default = 'vardiff'].


    Returns
    --------

    out : :class:`~spectrochempy.dataset.nddataset.NDDataset`.
        The subtracted dataset

    Examples
    ---------

    >>> import os
    >>> from spectrochempy.api import NDDataset, scpdata #doctest: +ELLIPSIS
    <BLANKLINE>
    ...
    <BLANKLINE>
    >>> path_A = os.path.join(scpdata, 'irdata', 'NH4Y-activation.SPG')
    >>> A = NDDataset.load(path_A, protocol='omnic')
    >>> ref = A[0]  # let's subtrack the first row
    >>> B = A.autosub(ref, [3900., 3700.], [1600., 1500.], inplace=False)
    >>> B #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    NDDataset: [[...]] dimensionless

    """

    # output dataset

    if not inplace:
        new = source.copy()
    else:
        new = source

    # we assume that the last dimension if always the dimension
    # to which we want to subtract.
    # Swap the axes to be sure to be in this situation

    if axis == new.ndim - 1:
        axis = -1

    try:
        ref.to(source.units)
    except:
        raise ValueError('Units of the dataset and reference are not compatible')

    swaped = False
    if axis != -1:
        new = new.swapaxes(axis, -1)
        swaped = True

    # TODO: detect the case where the ref is not exactly
    # with same coords: interpolate?

    # selection of the multiple ranges

    shape = list(new.shape)
    ranges = tuple(np.array(ranges, dtype=float))
    # must be float to be considered as frequency for instance

    coords = new.coordset[-1]
    xrange = CoordRange(*ranges, reversed=coords.is_reversed).ranges

    s = []
    r = []
    for xpair in xrange:
        # determine the slices

        sl = slice(*xpair)
        s.append(source[..., sl])
        r.append(ref[..., sl])

    # faster with np.array
    X_r = np.concatenate((*s,), axis=-1)
    ref_r = np.concatenate((*r,), axis=-1)

    indices, _ = list(zip(*np.ndenumerate(X_r[:, 0])))  # .squeeze())))

    # table of subtraction coefficients
    x = []

    # two methods
    if method == 'chi2':
        f = lambda alpha, p: np.sum((p - alpha * ref_r) ** 2)
    else:
        f = lambda alpha, p: np.var(np.diff(p - alpha * ref_r))

    for tup in indices:
        # slices = [i for i in tup]
        # slices.append(slice(None))
        # args = (X_r[slices],)
        args = (X_r[tup, :])
        res = minimize_scalar(f, args=args, method='brent')
        x.append(res.x)

    x = np.asarray(x)

    if not new.name.startswith('*'):
        new.name = '*' + new.name

    new._data -= np.dot(x.reshape(-1, 1), ref.data.reshape(1,-1))

    if swaped:
        new = new.swapaxes(axis, -1)

    new.history = str(
            new.modified) + ': ' + 'Automatic subtraction of :' + ref.name + '\n'

    return new
