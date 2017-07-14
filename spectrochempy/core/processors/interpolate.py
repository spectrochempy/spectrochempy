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


import scipy.interpolate
import numpy as np

#from ..dataset import NDDataset


__all__ = []


def interpolate(source, axis=0, size=None):
    # TODO: a simple interpolator of the data (either to reduce
    # or increase number of points in every dimension)
    raise NotImplementedError('Not yet implemented')


def align(source, ref, axis=0, refaxis=None, kind='linear', fill_value=np.nan,
          inplace=False):
    """
    Align the current dataset on a reference dataset along a given axis by
    interpolation

    Parameters
    -----------
    source : :class:`~spectrochempy.core.dataset.nddataset.NDDataset`.
        Dataset to align

    ref : :class:`~spectrochempy.core.dataset.nddataset.NDDataset`.
        Reference dataset on which to align

    axis : `int`[optional, default = 0].
        Tells on which axis to perform the alignement.

    kind: `str`[optional among [`linear`,`pchip`], default=`linear`].
        Kind of interpolation to performs for the alignment.

    Returns
    --------
    aligned_dataset : :class:`~spectrochempy.core.dataset.nddataset.NDDataset`
       A dataset aligned with ref along the axis


    #TODO : Perform an alignment along numeric axis labels

    """
    # TODO: Perform an alignment along numeric labels


    if refaxis is None:
        refaxis = axis

    if axis == -1:
        axis = source.ndim - 1

    # reorders source and reference in ascending order
    sourceordered = source.sort(axis)
    refordered = ref.sort(refaxis)

    try:
        sourceordered.coords(axis).to(refordered.coords(refaxis).units)
    except:
        raise TypeError(
            'units of the dataset and reference axes on which interpolate are not compatible')

    oldaxisdata = sourceordered.coords(axis).data
    refaxisdata = refordered.coords(
        refaxis).data  # TODO: at the end restore the original order

    if kind == 'linear':

        interpolator = lambda data, ax=-1: scipy.interpolate.interp1d(
            oldaxisdata, data, axis=ax, kind=kind,
            bounds_error=False, fill_value=fill_value,
            assume_sorted=True)

    elif kind == 'pchip':

        interpolator = lambda data, ax=-1: scipy.interpolate.PchipInterpolator(
            oldaxisdata, data,
            axis=ax, extrapolate=False)
    else:
        raise TypeError('Not recognised option kind for `align`')

    interpolate_data = interpolator(sourceordered.data, axis)
    newdata = interpolate_data(refaxisdata)

    interpolate_mask = interpolator(sourceordered.mask, axis)
    newmask = interpolate_mask(refaxisdata)

    interpolate_uncertainty = interpolator(sourceordered.uncertainty, axis)
    newuncertainty = interpolate_uncertainty(refaxisdata)

    interpolate_axis = interpolator(sourceordered.coords(axis).data)
    newaxisdata = interpolate_axis(refaxisdata)

    interpolate_axis_mask = interpolator(sourceordered.coords(axis).mask)
    newaxismask = interpolate_axis_mask(refaxisdata)

    if kind == 'pchip' and not np.isnan(fill_value):
        index = np.any(np.isnan(newdata))
        newdata[index] = fill_value

        index = np.any(np.isnan(newaxisdata))
        newaxisdata[index] = fill_value

    # create the new axis
    newaxes = source.axes.copy()
    newaxes[axis]._data = newaxisdata
    newaxes[axis]._mask = newaxismask
    newaxes[axis]._labels = np.array([''] * newaxisdata.size)

    # transform the dataset
    if inplace:
        out = source
    else:
        out = source.copy()

    out._data = newdata
    out._axes = newaxes
    out._mask = newmask
    out._uncertainty = newuncertainty

    out.name = '*' + source.name
    out.title = '*' + source.title

    out.history = '{}: aligned along dim {} with respect to dataset {} using axis {} \n'.format(
            str(source.modified), axis, ref.name, ref.axes[refaxis].title)

    return out

