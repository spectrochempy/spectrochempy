# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================

"""Plugin module to perform automatic subtraction of ref on a dataset.

"""

import numpy as np
from scipy.optimize import minimize_scalar

from spectrochempy.core.dataset.ndaxes import AxisRange

__all__ = ['autosub']

def autosub(source, ref, *ranges, axis=-1, method='vardiff', inplace=False):
    """Automatic subtraction of ref to the dataset to minimise peaks due to ref

    Parameters
    -----------
    source : :class:`~spectrochempy.core.dataset.nddataset.NDDataset`.
        Dataset to which we want to subtract the reference data

    ref : :class:`~spectrochempy.core.dataset.nddataset.NDDataset`.
         1D reference data, with a size maching the axis to subtract
         (axis parameter) #TODO : optionally use title of axis

    xrange : pair(s) of values. Any number of pairs is allowed.
        Axis range(s) in which the variance is minimized

    inplace : `bool`, optional, default = False.
        True if the subtraction is done in place.
        In this case we do not need to catch the function output

    axis : `int` [optional, default = -1].
        Tells on which axis to perform the subtraction

    method  : `str` [optional, default = 'vardiff'].


    Returns
    --------

    out : :class:`~spectrochempy.core.dataset.nddataset.NDDataset`.
        The subtracted dataset

    Examples
    ---------

    >>> import os
    >>> from spectrochempy.api import NDDataset, data #doctest: +ELLIPSIS
    <BLANKLINE>
    ...
    <BLANKLINE>
    >>> path_A = os.path.join(data, 'irdata', 'NH4Y-activation.SPG')
    >>> A = NDDataset.load(path_A, protocol='omnic')
    >>> ref = A[0]  # let's subtrack the first row
    >>> B = A.autosub(ref, [3900., 3700.], [1600., 1500.], inplace=False)
    >>> B
    ... #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    NDDataset([[-4.98e-08, -4.99e-08, ... -0.479,    -0.48]])

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
        raise TypeError('Units of the dataset and reference are not compatible')

    swaped = False
    if axis != -1:
        new = new.swapaxes(axis, -1)
        swaped = True

    # TODO: detect the case where the ref is not exactly
    # with same coords: interpolate?

    # selection of the multiple ranges

    shape = list(new.shape)
    # shape[-1] = 0
    # #shape = tuple(shape)
    # X_r = np.zeros(shape)
    #
    # shaper = [1 for _ in shape]
    # shaper[-1] = 0
    # #shaper = tuple(shaper)
    # ref_r = np.zeros(shaper)

    ranges = tuple(np.array(ranges,
                            dtype=float))   # must be float to be considered
                                            # as frequency for instance
    coords = new.coords(-1)
    xrange = AxisRange(*ranges, reversed=coords.is_reversed).ranges

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

    new._data -= np.dot(x.reshape(*shape[:-1], 1), ref.data)

    if swaped:
        new = new.swapaxes(axis, -1)

    new.history = str(
        new.modified) + ': ' + 'Automatic subtraction of :' + ref.name + '\n'

    return new
