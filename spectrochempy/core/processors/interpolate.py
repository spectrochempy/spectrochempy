# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""
This module defines functions related to interpolations

"""

__all__ = ['interpolate', 'align']

__dataset_methods__ = __all__

# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------

import scipy.interpolate
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# localimports
# ----------------------------------------------------------------------------------------------------------------------

from spectrochempy.utils import SpectroChemPyWarning, NOMASK


# ............................................................................
def interpolate(dataset, axis=0, size=None):
    # TODO: a simple interpolator of the data (either to reduce
    # or increase number of points in every dimension)
    raise NotImplementedError('Not yet implemented')


# ............................................................................
def align(dataset, ref, **kwargs):
    """Align the current dataset on a reference dataset along a given axis by
    interpolation

    Parameters
    -----------
    dataset : |NDDataset|.
        Dataset to align

    ref : |NDDataset|.
        Reference dataset on which to align

    axis : int, [optional, default = 0].
        Tells on which axis to perform the alignement.

    method: str[optional among [`linear`,`pchip`], default=`linear`].
        method of interpolation to performs for the alignment.

    Returns
    --------
    aligned_dataset : |NDDataset|
       A dataset aligned with ref along the axis


    #TODO : Perform an alignment along numeric axis labels

    """
    # TODO: Perform an alignment along numeric labels

    axis = dataset.get_axis(**kwargs)
    refaxis = kwargs.pop('refdim', None)

    if refaxis is None:
        refaxis = axis

    if axis == -1:
        axis = dataset.ndim - 1

    # reorders dataset and reference in ascending order
    is_sorted = False
    if dataset.coords(axis).reversed:
        datasetordered = dataset.sort(axis, descend=False)
        refordered = ref.sort(refaxis, descend=False)
        is_sorted = True
    else:
        datasetordered = dataset.copy()
        refordered = ref.copy()

    try:
        datasetordered.coords(axis).to(refordered.coords(refaxis).units)
    except:
        raise ValueError(
            'units of the dataset and reference axes on which interpolate are not compatible')

    oldaxisdata = datasetordered.coords(axis).data
    refaxisdata = refordered.coords(refaxis).data  # TODO: at the end restore the original order

    method = kwargs.pop('method', 'linear')
    fill_value = kwargs.pop('fill_value', np.NaN)

    if method == 'linear':
        interpolator = lambda data, ax=0: scipy.interpolate.interp1d(
            oldaxisdata, data, axis=ax, kind=method, bounds_error=False, fill_value=fill_value, assume_sorted=True)

    elif method == 'pchip':
        interpolator = lambda data, ax=0: scipy.interpolate.PchipInterpolator(
            oldaxisdata, data, axis=ax, extrapolate=False)
    else:
        raise AttributeError(f'{method} is not a recognised option method for `align`')

    interpolate_data = interpolator(datasetordered.data, axis)
    newdata = interpolate_data(refaxisdata)

    if datasetordered.is_masked:
        interpolate_mask = interpolator(datasetordered.mask, axis)
        newmask = interpolate_mask(refaxisdata)
    else:
        newmask = NOMASK

    # interpolate_axis = interpolator(datasetordered.coords(axis).data)
    # newaxisdata = interpolate_axis(refaxisdata)
    newaxisdata = refaxisdata.copy()

    if method == 'pchip' and not np.isnan(fill_value):
        index = np.any(np.isnan(newdata))
        newdata[index] = fill_value

        index = np.any(np.isnan(newaxisdata))
        newaxisdata[index] = fill_value

    # create the new axis
    newaxes = dataset.coords.copy()
    newaxes[axis]._data = newaxisdata
    newaxes[axis]._labels = np.array([''] * newaxisdata.size)

    # transform the dataset
    inplace = kwargs.pop('inplace', False)

    if inplace:
        out = dataset
    else:
        out = dataset.copy()

    out._data = newdata
    out._coords = newaxes
    out._mask = newmask

    out.name = '*' + dataset.name
    out.title = '*' + dataset.title

    out.history = '{}: aligned along dim {} with respect to dataset {} using coords {} \n'.format(
        str(dataset.modified), axis, ref.name, ref.coords[refaxis].title)

    if is_sorted and out.coords(axis).reversed:
        out.sort(axis, descend=True, inplace=True)
        ref.sort(refaxis, descend=True, inplace=True)

    return out
