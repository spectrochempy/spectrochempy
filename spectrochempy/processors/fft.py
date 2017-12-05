# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2016 Christian Fernandez
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
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



__all__ = []

# =============================================================================
# Standard python imports
# =============================================================================
from spectrochempy.application import app
plotoptions = app.plotoptions
log = app.log
options = app

# =============================================================================
# Third party imports
# =============================================================================
import numpy as np

# =============================================================================
# Local imports
# =============================================================================
from spectrochempy.utils import closer_power_of_two

epsilon = np.finfo(float).eps


# =============================================================================
# interface for the processing class
# =============================================================================
# NOTE: if these parameters are not set, the name of the module is taken
# __all__ = ["ft"]

# =============================================================================
# generic transform function
# =============================================================================

def ft(self, **kwargs):
    """Apply a complex fast fourier transform

    Parameters
    ----------
    size : int

        size of the transformed dataset dimension

    inplace : `bool`, optional, default = `False`

        Should we make the transform in place or return a new dataset

    axis : optional, default is -1

        The transformation is applied in the last dimension or on the specified
        axis of a dataset.


    Returns
    -------
    object : nd-dataset or nd-array
        transformed dataset

    """

    # which axis ?
    axis = kwargs.pop('axis', -1)

    if axis < 0:
        axis = self.ndim + axis

    # dataset selected axis
    lastaxe = self.axes[axis]

    if (lastaxe.unitless or lastaxe.dimensionless or
                lastaxe.units.dimensionality != '[time]'):
        log.error('ft apply only to dimensions with [time] dimensionality')
        return self

    # first parameters ?
    si = kwargs.get('size', kwargs.get('si', None))
    if si is None:
        # we default to Hz units
        si = closer_power_of_two(self.meta.td[axis] * 2)

    # should we work on complex data
    iscomplex = self.is_complex[axis]

    # if we are in NMR we have an additional complication due to the mode
    # of acquisition (sequential mode when ['QSEQ','TPPI','STATES-TPPI'])
    encoding = self.meta.encoding[axis]

    # we work on the last dimension always
    if axis != self.ndim - 1:  # swap
        data = self.swapaxes(-1, axis, inplace=False)
    else:
        data = self.copy()

    # perform the fft
    if iscomplex and encoding in ['QSIM', 'DQD']:
        arr = np.fft.fft(data.real().data + data.imag().data * 1j, si)
    else:
        raise NotImplementedError(encoding)

    # TODO: we need here to create a new dataset with new shape and axis
    if axis != self.ndim - 1:  # swap back
        data = data.swapaxes(-1, axis)

    # inplace?
    inplace = kwargs.pop('inplace', False)
    if inplace:
        self = data
        return self
    else:
        return data
