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
# author : A.Travert
# =============================================================================


import datetime as datetime
import scipy.interpolate
import numpy as np


from ...dataset.ndaxes import AxisRange

__all__=['basecor']

def basecor(source,*ranges,axis=-1,
            method='sequential',
            interpolation='polynomial',
            order=0,
            npc=1):
    """Base function for dataset baseline correction.

    2 methods are proposed:

    * ``sequential`` (default) = classical polynom fit or spline \
        interpolation with one fit per spectrum
    * ``multivariate`` = SVD modeling of baseline, polynomial fit of PC's \
    and calculation of the modelled baseline spectra.
    """

    # output dataset
    new = source.copy()

    # we assume that the last dimension if always the dimension to which
    # we want to subtract the baseline.
    # Swap the axes to be sure to be in this situation

    if axis == new.ndim - 1:
        axis = -1

    swaped = False
    if axis != -1:
        new.swapaxes(axis, -1, inplace=True)
        swaped = True

    # most of the time we need sorted axis, so let's do it now
    coords = new.coords(-1)

    sorted=False
    if new.coords(-1).is_reversed:
        new.sort(axis=-1, inplace=True)
        sorted = True

    coords = new.coords(-1)
    baseline = np.zeros_like(new)
    ranges = AxisRange(*ranges)

    # Extract: Sbase: the matrix of data correponding to ranges
    #          xbase: the xaxis values corresponding to ranges

    s = []
    for pair in ranges:
        # determine the slices

        sl = slice(*pair)
        s.append(new[..., sl])

    sbase = s[0].concatenate(*s[1:],
                            axis=-1)    # TODO: probably we could use masked
                                        # data instead of concatenating
    xbase = sbase.coords(-1)

    if method == 'sequential':

        if interpolation == 'polynomial':
            # bad fit when NaN values => are replaced by 0
            if np.any(np.isnan(sbase)):
                sbase[np.isnan(sbase)] = 0

            polycoef = np.polynomial.polynomial.polyfit(xbase.data, \
                                                        sbase.data.T, deg=order,
                                                        rcond=None, full=False)
            baseline = np.polynomial.polynomial.polyval(coords.data, polycoef)

        elif interpolation == 'pchip':
            for i in range(new.shape[0]):
                y = scipy.interpolate.PchipInterpolator(
                                                   xbase.data, sbase.data[i, :])
                baseline[i, :] = y(coords)

    elif method == 'multivariate':

        # SVD of Sbase
        U, s, Vt = np.linalg.svd(sbase.data, full_matrices=False, \
                                 compute_uv=True)

        # select npc loadings & compute scores
        Pt = (Vt[0:npc, :])
        T = np.dot(U[:, 0:npc], np.diag(s)[0:npc, 0:npc])

        baseline_loadings = np.zeros((npc, new.shape[1]))

        if interpolation == 'pchip':
            for i in range(npc):
                y = scipy.interpolate.PchipInterpolator(xbase.data, Pt[i, :])
                baseline_loadings[i, :] = y(coords)

        elif interpolation == 'polynomial':
            polycoef = np.polynomial.polynomial.polyfit(xbase.data,
                                                        Pt.T, deg=order,
                                                        rcond=None, full=False)

            baseline_loadings = np.polynomial.polynomial.polyval(coords.data,
                                                                 polycoef)

        baseline = np.dot(T, baseline_loadings)

    new.name = '*' + source.name
    new.data = new.data - baseline

    # eventuallly sort back to the original order
    if sorted:
        new.sort(inplace=True, descend=True)

    new.history = str(new.modified) + \
                  ': ' + 'Baseline correction.' + ' Method: '
    if method == 'multivariate':
        new.history = 'multivariate (' + str(npc) + ' PCs).'
    else:
        new.history = 'sequential.'

    new.history = 'Interpolation: '

    if interpolation == 'polynomial':
        new.history = 'polynomial, order=' + str(order) + '.\n'
    else:
        new.history = 'pchip. \n'

    if swaped:
        new = new.swapaxes(axis, -1)

    return new



