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
# Third party imports
# =============================================================================
import numpy as np

# =============================================================================
# Local imports
# =============================================================================
from .apodize import apodize
from spectrochempy.utils import epsilon

# =============================================================================
# interface for the processing class
# =============================================================================
#__all__ = ["gm"]

# =============================================================================
# em function
# =============================================================================

def gm(self, *args, **kwargs):
    """Calculate a Lorentz-tp-Gauss apodization

    Functional form of apodization window:

    .. math::
        gm(x) = \\exp(e - g^2)

    Where:

    .. math::
        e = pi * x * lb \\\\
        g = 0.6 * pi * gb * (x - shifted)

    Parameters
    ----------
    lb : float

        Inverse exponential width.
        If it is not a quantity with units,
        it is assumed to be a broadening expressed in Hz.

    gb : float

        Gaussian broadening width.
        If it is not a quantity with units,
        it is assumed to be a broadening expressed in Hz.

    shifted : `float` or `quantity`

        Shift the data time origin by this amount. If it is not a quantity
        it is assumed to be expressed in the data units of the last
        dimension.

    inv : bool, optional

        True for inverse apodization.  False (default) for standard.

    rev : bool, optional.

        True to reverse the apodization before applying it to the data.

    """
    args = list(args)  # important (args is a tuple)

    # lb broadening
    lb = kwargs.get('lb', 0)
    if lb==0:
        # let's try the args if the kwargs was not passed
        if len(args)>0:
            lb = args.pop(0)

    # gb broadening
    gb = kwargs.get('gb', 0)
    if gb==0:
        # let's try the second args if the kwargs was not passed
        if len(args)>0:
            gb = args.pop(0)

    # shifted ?
    shifted = kwargs.pop('shifted', 0)
    if shifted == 0:
        # let's try the args if the kwargs was not passed
        if len(args) > 0:
            shifted = args.pop(0)

    # apod func (must be func(x, tc1, tc2, shifted) form
    def func(x, tc1, tc2, shifted):
        if tc1.magnitude > epsilon:
            e = np.pi * x / tc1
        else:
            e = e = np.zeros_like(x)
        if tc2.magnitude > epsilon:
            g = 0.6 * np.pi * (x - shifted) / tc2
        else:
            g = np.zeros_like(x)

        return np.exp(e - g**2).data


    kwargs['method'] = func
    kwargs['apod'] = lb
    kwargs['apod2'] = gb
    kwargs['shifted'] = shifted

    return apodize(self, **kwargs)
