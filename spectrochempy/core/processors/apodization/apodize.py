# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

__all__ = ["apodize"]

__dataset_methods__ = ['apodize']


# =============================================================================
# Third party imports
# =============================================================================
import numpy as np

# =============================================================================
# Local imports
# =============================================================================
from spectrochempy.units.units import ur, Quantity
from spectrochempy.utils import epsilon, interleave, interleaved2complex
from spectrochempy.application import plotter_preferences, log, preferences

# =============================================================================
# generic apodization function
# =============================================================================
def apodize(dataset, **kwargs):
    """Calculate an apodization function

    The apodization is calculated in the last dimension of the dataset.
    The data in the last dimension MUST be time-domain,
    or an error is raised.

    Parameters
    ----------
    method : Callable,
        Apodization function

    apod : float or a `Quantity`, default 0, or list

        Apodization oarameter(s). If it is not a Quantity with units,
        it is assumed to be a broadening expressed in Hz.

    apod2 : float or a `Quantity`, optional, default 0, or list

        Second apodization oarameter(s). If it is not a Quantity with units,
        it is assumed to be a broadening expressed in Hz.

    apod3 : float or a `Quantity`, optional, default 0, or list

        third apodization oarameter(s). If it is not a Quantity with units,
        it is assumed to be a broadening expressed in Hz.
        alias: shifted

    inv : bool, optional

        True for inverse apodization.  False (default) for standard.

    rev : bool, optional.

        True to reverse the apodization before applying it to the data.

    apply : `bool`, optional, default = True

        Should we apply the calculated apodization to the dataset (default)
        or just return the apodization ndarray.

    inplace : `bool`, optional, default = True

        Should we make the transform in place or return a new dataset

    axis : optional, default is -1

    Returns
    -------
    object : nd-dataset or nd-array
        apodized dataset if apply is True, the apodization array if not True.

    """

    # output dataset inplace (by default) or not
    inplace = kwargs.pop('inplace', True)

    # Do we apply the apodization or just return
    # the apodization array
    apply = kwargs.pop('apply', True)
    inplace = inplace and apply  # force inplace false if we do not apply

    if not inplace:
        new = dataset.copy()  # copy to be sure not to modify this dataset
    else:
        new = dataset

    # On which axis do we want to apodize?
    axis = kwargs.pop('axis', -1)

    # if axis < 0:
    #    axis = dataset.ndim + axis
    if axis == new.ndim - 1:
        axis = -1

    # we assume that the last dimension if always the dimension
    # to which we want to apply apodization.

    # swap the axes to be sure to be in this situation

    swaped = False
    if axis != -1:
        new.swapaxes(axis, -1, inplace=True)  # must be done in  place
        swaped = True

    lastcoord = new.coordset[-1]
    if (lastcoord.unitless or lastcoord.dimensionless or
                lastcoord.units.dimensionality != '[time]'):
        log.error('apodization functions apply only to dimensions '
                  'with [time] dimensionality')
        return dataset

    # first parameters (apodization in Hz) ?
    apod = kwargs.get('apod', kwargs.get('apod1', 0))
    if not isinstance(apod, Quantity):
        # we default to Hz units
        apod = apod * ur.Hz

    # second parameters (second apodization parameter in Hz) ?
    apod2 = kwargs.get('apod2', 0)
    if not isinstance(apod2, Quantity):
        # we default to Hz units
        apod2 = apod2 * ur.Hz

    # if no parameter passed
    if np.abs(apod.magnitude) <= epsilon and np.abs(apod2.magnitude) <= epsilon:
        # nothing to do
        if not apply:
            # we have to return something as an array of one,
            # as nothing is calculated
            new.data = np.ones_like(new.data)
        return new

    # create the args list
    args = []

    # convert (1./apod) to the axis time units
    if apod.magnitude > epsilon:
        tc1 = (1. / apod).to(lastcoord.units)
        args.append(tc1)
    else:
        args.append(0 * ur.us)

    # convert (1./apod2) to the axis time units
    if np.abs(apod2.magnitude) > epsilon:
        tc2 = (1. / apod2).to(lastcoord.units)
        args.append(tc2)
    else:
        args.append(0 * ur.us)

    # should we shift the time origin? (should be in axis units)
    shifted = kwargs.get('shifted', kwargs.get('apod3', 0))
    if not isinstance(shifted, Quantity):
        # we default to lastcoord.units
        shifted = shifted * lastcoord.units
    else:
        shifted = shifted.to(lastcoord.units)
    args.append(shifted)

    # compute the apodization function
    x = lastcoord
    method = kwargs.pop('method', None)
    if method is None or method == 'em':
        # em by default
        apod_arr = np.exp(-np.pi * np.abs(x - args[2]) / args[0])
    else:
        apod_arr = method(x, *args)

    if kwargs.pop('rev', False):
        apod_arr = apod_arr[::-1]

    if kwargs.pop('inv', False):
        apod_arr = 1. / apod_arr  # invert apodization

    # apply?
    # if not apply:
    #    return apod_arr

    # if we are in NMR we have an additional complication due to the mode
    # of acquisition (sequential mode when ['QSEQ','TPPI','STATES-TPPI'])
    # TODO: CHECK IF THIS WORK WITH 2D DATA - IMPORTANT - CHECK IN PARTICULAR IF SWAPING ALSO SWAP METADATA (NOT SURE FOR NOW)
    iscomplex = new.is_complex[-1]
    encoding = new.meta.encoding[-1]
    # TODO: handle this eventual complexity

    if iscomplex:
        data = interleaved2complex(new.data)
        if not apply:
            data = np.ones_like(data) + 0j
        data, _ = interleave(data * apod_arr)
        new._data = data
    else:
        data = new.data
        if not apply:
            data = np.ones_like(data).astype(new.data.dtype)
        new._data = data * apod_arr

    # restore original data order if it was swaped
    if swaped:
        new.swapaxes(axis, -1, inplace=True)  # must be done inplace

    if apply:
        name = kwargs.pop('method_name', 'em')
        new.history = str(
                new.modified) + ': ' + '%s apodization performed: ' % name + str(
            apod) + '\n'

    return new

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
