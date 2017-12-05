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



# =============================================================================
# Third party imports
# =============================================================================
import numpy as np

# =============================================================================
# Local imports
# =============================================================================
from .apodize import apodize
from spectrochempy.units.units import Quantity

# =============================================================================
# interface for the processing class
# =============================================================================
# __all__ = ["sp"]

__all__ = []

# =============================================================================
# sp function
# =============================================================================

def sp(self, *args, **kwargs):
    """Calculate a Shifted sine-bell apodization

    Functional form of apodization window:

    .. math::
        sp(x) = \\sin(\\frac{pi * off + pi * (end - off) * x} {size - 1})^{pow}

    Parameters
    ----------
    off : float

        offset - Specifies the starting point of the sine-bell in time units
        The default value is 0.0.

    end : float

        end - Specifies the ending point of the sine-bell in time units.

    pow : float

        pow - Specifies the exponent of the sine-bell; Non-integer values
        are allowed. Common values are 1.0 (for ordinary sine-bell) and 2.0
        (for squared-bell functions). The default value is 1.0.


    inv : bool, optional

        True for inverse apodization.  False (default) for standard.

    rev : bool, optional.

        True to reverse the apodization before applying it to the data.

    """
    # TODO: To be finished!!!!

    args = list(args)  # important (args is a tuple)

    # off
    off = kwargs.get('off', 0)
    if off == 0:
        # let's try the args if the kwargs was not passed
        if len(args) > 0:
            off = args.pop(0)

    # end
    end = kwargs.get('end', 1.)
    if end == 1.:
        # let's try the second args if the kwargs was not passed
        if len(args) > 0:
            end = args.pop(0)

    # pow
    pow = kwargs.pop('pow', 1.)
    if pow == 1.:
        # let's try the args if the kwargs was not passed
        if len(args) > 0:
            pow = args.pop(0)

    # apod func (must be func(x, tc1, tc2, tc3) form
    def func(x, off, end, pow):
        w = x[-1].data - x[0].data
        i = np.arange(0, x.size, 1)
        if isinstance(pow, Quantity):
            pow = pow.magnitude
            off = off.magnitude
            end = end.magnitude
        return np.sin(
                np.pi * off / w + np.pi * ((end - off) / w) * i / (
                x.size - 1.)) ** pow

    kwargs['method'] = func
    kwargs['apod'] = off
    kwargs['apod2'] = end
    kwargs['apod3'] = pow

    return apodize(self, **kwargs)
