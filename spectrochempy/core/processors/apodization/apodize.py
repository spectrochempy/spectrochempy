# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ["apodize"]

__dataset_methods__ = []

# ======================================================================================================================
# Third party imports
# ======================================================================================================================
import numpy as np

# ======================================================================================================================
# Local imports
# ======================================================================================================================
from spectrochempy.units.units import ur, Quantity
from spectrochempy.utils import epsilon
from spectrochempy.core import general_preferences, error_

import spectrochempy.extern.nmrglue as ng

# ======================================================================================================================
# generic apodization function
# ======================================================================================================================
def apodize(dataset, method, *apod, inv=False, rev=False, inplace=True, dim=-1, **kwargs):
    """Calculate an apodization window function

    In principle, this function should not be called directly: its purpose is to be used by `em`, `gm`, `sinm`, etc...
    kind of apodization functions.

    Parameters
    ----------
    method : Callable.
        Apodization function
    *apod : tuple.
        Apodization window parameter(s) depending on the actual apodization `method` used.
    inv : bool, optional, default=False.
        True for inverse apodization.
    rev : bool, optional, default=False.
        True to reverse the apodization before applying it to the data.
    inplace : bool, optional, default=True.
        True if we make the transform inplace.  If False, the function return a new dataset
    dim : str or int, optional, default='x'.
        Specify on which dimension to apply this method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    
    Returns
    -------
    object : nd-dataset or nd-array
        apodized dataset if apply is True, the apodization array if not True.

    """

    # output dataset inplace (by default) or not
    if not inplace:
        new = dataset.copy()  # copy to be sure not to modify this dataset
    else:
        new = dataset
        
    # On which axis do we want to apodize? (get axis from arguments)
    axis, dim = dataset.get_axis(dim, negative_axis=True)
    
    # The last dimension if always the dimension on which we apply the apodization window.
    # If needed, swap the dimensions to be sure to be in this situation

    swaped = False
    if axis != -1:
        new.swapaxes(axis, -1, inplace=True)  # must be done in  place
        swaped = True

    lastcoord = new.coords[-1]
    if (lastcoord.unitless or lastcoord.dimensionless or
            lastcoord.units.dimensionality != '[time]'):
        error_('apodization functions apply only to dimensions '
                  'with [time] dimensionality')
        return dataset

    # # first parameters (apodization in Hz) ?
    # apod = kwargs.get('apod', kwargs.get('apod1', 0))
    # if not isinstance(apod, Quantity):
    #     # we default to Hz units
    #     apod = apod * ur.Hz
    #
    # # second parameters (second apodization parameter in Hz) ?
    # apod2 = kwargs.get('apod2', 0)
    # if not isinstance(apod2, Quantity):
    #     # we default to Hz units
    #     apod2 = apod2 * ur.Hz
    #
    # # if no parameter passed
    # if np.abs(apod.magnitude) <= epsilon and np.abs(apod2.magnitude) <= epsilon:
    #     # nothing to do
    #     return new, np.ones_like(new.data)
    #
    # # create the args list
    # args = []
    #
    # # convert (1./apod) to the axis time units
    # if apod.magnitude > epsilon:
    #     if not apod.check('[time]'):
    #         apod = 1./apod
    #     tc1 = apod.to(lastcoord.units)
    #     args.append(tc1)
    # else:
    #     args.append(0 * ur.us)
    #
    # # convert (1./apod2) to the axis time units
    # if np.abs(apod2.magnitude) > epsilon:
    #     if not apod2.check('[time]'):
    #         apod2 = 1./apod2
    #     tc2 = apod2.to(lastcoord.units)
    #     args.append(tc2)
    # else:
    #     args.append(0 * ur.us)
    #
    # # should we shift the time origin? (should be in axis units)
    # shifted = kwargs.get('shifted', kwargs.get('apod3', 0))
    # if not isinstance(shifted, Quantity):
    #     # we default to lastcoord.units
    #     shifted = shifted * lastcoord.units
    # else:
    #     if not shifted.check('[time]'):
    #         shifted = 1./apod2
    #     shifted = shifted.to(lastcoord.units)
    # args.append(shifted)
    #
    # pw = kwargs.get('pow', 1.)
    # args.append(pw)
    #
    # compute the apodization function
    x = lastcoord
    apod_arr = method(x, *apod)

    if rev:
        apod_arr = apod_arr[::-1] # reverse apodization

    if inv:
        apod_arr = 1. / apod_arr  # invert apodization

    # if we are in NMR we have an additional complication due to the mode
    # of acquisition (sequential mode when ['QSEQ','TPPI','STATES-TPPI'])
    # TODO: CHECK IF THIS WORK WITH 2D DATA - IMPORTANT - CHECK IN PARTICULAR IF SWAPING ALSO SWAP METADATA (NOT SURE FOR NOW)
    iscomplex = new.is_complex
    isquaternion = new.is_quaternion
    encoding = new.meta.encoding[-1]
    # TODO: handle this eventual complexity

    data = new.data
    if not isquaternion:
        new._data = data * apod_arr
    else:
        data['R'] = data['R'] * apod_arr
        data['I'] = data['I'] * apod_arr
        new._data = data

    # restore original data order if it was swaped
    if swaped:
        new.swapaxes(axis, -1, inplace=True)  # must be done inplace

    name = method.__module__.split('.')[-1]
    new.history = f'{name} apodization performed on dimension {dim} with parameters:' + str(apod)

    curve = new.copy()
    curve.data = apod_arr
    return new, curve

    # shifted = args.shifted  # float(kargs.get('top', 0.0))
    # k_shifted = args.k_shifted
    #
    # if k_shifted and (axis == -1 or axis == 1) and dataset.is_2d:
    #     # in k shifted method for 2D spectra, the top of
    #     # the broadening function follow the top of the echoes
    #     # parameters p and q should be defined in parameters
    #     command = 'k_shifted_em'
    #
    #     p = k_shifted[0]  # coherence
    #     q = k_shifted[1]  # satellite order
    #
    #     if p == 1 and q == 0:
    #         ratio = 1.0
    #     else:
    #         ratio = k_ratio(dataset, p, q)
    #     shifted = np.abs(ratio)
    #     if args.verbose:
    #         print('k_ratio: %.3f' % ratio)
    #
    # par.LB = lb
    #
    # kargs['lb'] = lb / dataset.get_multiplicator(axis)
    # kargs['shifted'] = shifted
    # kargs['states'] = True if 'STATES' in par.encoding else False
    #

    #
    # if axis == 0:
    #     # transpose temporarily the data for indirect dimension ft
    #     datv = datv.T
    #
    # dataset.data = datv
    # dataset.history.append('Exponential apodization lb:%.2f' % par.LB)
