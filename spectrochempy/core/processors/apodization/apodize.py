# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2016 Christian Fernandez
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
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


# =============================================================================
# Standard python imports
# =============================================================================
import logging

# =============================================================================
# Third party imports
# =============================================================================
import numpy as np

# =============================================================================
# Local imports
# =============================================================================
from spectrochempy.core.units import ur, Quantity, Measurement, set_nmr_context
from spectrochempy.utils import epsilon

__all__ = ["apodize"]

# =============================================================================
# Constants
# =============================================================================
from spectrochempy.application import log


# =============================================================================
# generic apodization function
# =============================================================================
def apodize(self, **kwargs):
    """Calculate an apodization function

    The apodization is calculated in the last dimension of the dataset.
    The data in the last dimension MUST be time-domain,
    or an error is raised.

    Parameters
    ----------
    method : Callable,
        Apodization function

    apod : `float` or a `Quantity`, default 0, or list

        Apodization oarameter(s). If it is not a Quantity with units,
        it is assumed to be a broadening expressed in Hz.

    apod2 : `float` or a `Quantity`, optional, default 0, or list

        Second apodization oarameter(s). If it is not a Quantity with units,
        it is assumed to be a broadening expressed in Hz.

    apod3 : `float` or a `Quantity`, optional, default 0, or list

        third apodization oarameter(s). If it is not a Quantity with units,
        it is assumed to be a broadening expressed in Hz.
        alias: shifted

    inv : bool, optional

        True for inverse apodization.  False (default) for standard.

    rev : bool, optional.

        True to reverse the apodization before applying it to the data.

    apply : `bool`, optional, default = `True`

        Should we apply the calculated apodization to the dataset (default)
        or just return the apodization ndarray.

    inplace : `bool`, optional, default = `False`

        Should we make the transform in place or return a new dataset

    axis : optional, default is -1

    Returns
    -------
    object : nd-dataset or nd-array
        apodized dataset if apply is True, the apodization array if not True.

    """

    # which axis ?
    axis = kwargs.pop('axis', -1)

    if axis < 0:
        axis = self.ndim + axis

    # get selected axis
    lastaxe = self.axes[axis]

    if (lastaxe.unitless or lastaxe.dimensionless or
                                      lastaxe.units.dimensionality != '[time]'):
        log.error('apodization apply only to dimensions '
                     'with [time] dimensionality')
        return self

    # first parameters (apodization in Hz) ?
    apod = kwargs.get('apod', kwargs.get('apod1', 0))
    if not isinstance(apod, Quantity):
        # we default to Hz units
        apod = apod * units.Hz

    # second parameters (second apodization parameter in Hz) ?
    apod2 = kwargs.get('apod2', 0)
    if not isinstance(apod2, Quantity):
        # we default to Hz units
        apod2 = apod2 * units.Hz

    # if no parameter passed
    if np.abs(apod.magnitude) <= epsilon and np.abs(apod2.magnitude) <= epsilon:
        # nothing to do
        return self

    # create the args list
    args = []

    # convert (1./apod) to the axis time units
    if apod.magnitude > epsilon:
        tc1 = (1./apod).to(lastaxe.units)
        args.append(tc1)
    else:
        args.append(0 * units.us)

    # convert (1./apod2) to the axis time units
    if np.abs(apod2.magnitude) > epsilon:
        tc2 = (1. / apod2).to(lastaxe.units)
        args.append(tc2)
    else:
        args.append(0 * units.us)

    # should we shift the time origin? (should be in axis units)
    shifted = kwargs.get('shifted', kwargs.get('apod3', 0))
    if not isinstance(shifted, Quantity):
        # we default to lastaxe.units
        shifted = shifted * lastaxe.units
    else:
        shifted = shifted.to(lastaxe.units)
    args.append(shifted)

    # if we are in NMR we have an additional complication due to the mode
    # of acquisition (sequential mode when ['QSEQ','TPPI','STATES-TPPI'])
    iscomplex = self.is_complex[axis]
    encoding = self.meta.encoding[axis]
    #TODO: handle this eventual complexity

    # compute the apodization function
    x = lastaxe
    method = kwargs.pop('method',None)
    if method is None or method=='em':
        # em by default
        apod_arr = np.exp(-np.pi * np.abs(x - args[2])/args[0])
    else:
        apod_arr = method(x, *args)

    if kwargs.pop('rev', False):
        apod_arr = apod_arr[::-1]

    if kwargs.pop('inv', False):
        apod_arr  = 1. / apod_arr  # invert apodization

    # apply?
    apply = kwargs.pop('apply', True)
    if not apply:
        return apod_arr

    # we work on the last dimension always
    if axis != self.ndim - 1:  # swap
        data = self.swapaxes(-1, axis, inplace=False)
    else:
        data = self.copy()

    data = data * apod_arr

    if axis != self.ndim - 1:  # swap back
        data = data.swapaxes(-1, axis)

    # inplace?
    inplace = kwargs.pop('inplace', False)
    if inplace:
        self = data
        return self
    else:
        return data

    # shifted = args.shifted  # float(kargs.get('top', 0.0))
    # k_shifted = args.k_shifted
    #
    # if k_shifted and (axis == -1 or axis == 1) and source.is_2d:
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
    #         ratio = k_ratio(source, p, q)
    #     shifted = np.abs(ratio)
    #     if args.verbose:
    #         print('k_ratio: %.3f' % ratio)
    #
    # par.LB = lb
    #
    # kargs['lb'] = lb / source.get_multiplicator(axis)
    # kargs['shifted'] = shifted
    # kargs['states'] = True if 'STATES' in par.encoding else False
    #

    #
    # if axis == 0:
    #     # transpose temporarily the data for indirect dimension ft
    #     datv = datv.T
    #
    # source.data = datv
    # source.history.append('Exponential apodization lb:%.2f' % par.LB)
