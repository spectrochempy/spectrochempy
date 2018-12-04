# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""
This module defines functions related to interpolations

"""

__all__ = ['interpolate', 'align']

__dataset_methods__ = __all__

# ----------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------

import scipy.interpolate
import numpy as np

# ----------------------------------------------------------------------------
# localimports
# ----------------------------------------------------------------------------

from spectrochempy.utils import SpectroChemPyWarning


# ............................................................................
def interpolate(dataset, axis=0, size=None):
    # TODO: a simple interpolator of the data (either to reduce
    # or increase number of points in every dimension)
    raise NotImplementedError('Not yet implemented')


# ............................................................................
def align(dataset, ref, axis=0, refaxis=None, method='linear', fill_value=np.nan,
          inplace=False):
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


    if refaxis is None:
        refaxis = axis

    if axis == -1:
        axis = dataset.ndim - 1

    # reorders dataset and reference in ascending order
    datasetordered = dataset.sort(axis)
    refordered = ref.sort(refaxis)

    try:
        datasetordered.coordset(axis).to(refordered.coordset(refaxis).units)
    except:
        raise ValueError(
                'units of the dataset and reference axes on which interpolate are not compatible')

    oldaxisdata = datasetordered.coordset(axis).data
    refaxisdata = refordered.coordset(
            refaxis).data  # TODO: at the end restore the original order

    if method == 'linear':

        interpolator = lambda data, ax=-1: scipy.interpolate.interp1d(
                oldaxisdata, data, axis=ax, kind=method,
                bounds_error=False, fill_value=fill_value,
                assume_sorted=True)

    elif method == 'pchip':

        interpolator = lambda data, ax=-1: scipy.interpolate.PchipInterpolator(
                oldaxisdata, data,
                axis=ax, extrapolate=False)
    else:
        raise AttributeError('Not recognised option method for `align`')

    interpolate_data = interpolator(datasetordered.data, axis)
    newdata = interpolate_data(refaxisdata)

    interpolate_mask = interpolator(datasetordered.mask, axis)
    newmask = interpolate_mask(refaxisdata)

    interpolate_uncertainty = interpolator(datasetordered.uncertainty, axis)
    newuncertainty = interpolate_uncertainty(refaxisdata)

    interpolate_axis = interpolator(datasetordered.coordset(axis).data)
    newaxisdata = interpolate_axis(refaxisdata)

    interpolate_axis_mask = interpolator(datasetordered.coordset(axis).mask)
    newaxismask = interpolate_axis_mask(refaxisdata)

    if method == 'pchip' and not np.isnan(fill_value):
        index = np.any(np.isnan(newdata))
        newdata[index] = fill_value

        index = np.any(np.isnan(newaxisdata))
        newaxisdata[index] = fill_value

    # create the new axis
    newaxes = dataset.coordset.copy()
    newaxes[axis]._data = newaxisdata
    newaxes[axis]._mask = newaxismask
    newaxes[axis]._labels = np.array([''] * newaxisdata.size)

    # transform the dataset
    if inplace:
        out = dataset
    else:
        out = dataset.copy()

    out._data = newdata
    out._coordset = newaxes
    out._mask = newmask
    out._uncertainty = newuncertainty

    out.name = '*' + dataset.name
    out.title = '*' + dataset.title

    out.history = '{}: aligned along dim {} with respect to dataset {} using coords {} \n'.format(
            str(dataset.modified), axis, ref.name, ref.coordset[refaxis].title)

    return out
