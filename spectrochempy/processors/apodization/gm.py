# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2016 Christian Fernandez
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# CeCILL FREE SOFTWARE LICENSE AGREEMENT (Version B)
# See full LICENSE agreement in the root directory
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
__all__ = ["gm"]


# =============================================================================
# gm function
# =============================================================================

def gm(source, *args, **kwargs):
    """Calculate a Lorentz-Gauss apodization

    Functional form of apodization window:

    .. math::
        gm(x) = \\exp(e - g^2)

    Where:

    .. math::
        e = pi * x * lb \\\\
        g = 0.6 * pi * gb * (x - shifted)

    Parameters
    ----------
    lb : float or `quantity`

        Inverse exponential width.
        If it is not a quantity with units,
        it is assumed to be a broadening expressed in Hz.

    gb : float or `quantity`

        Gaussian broadening width.
        If it is not a quantity with units,
        it is assumed to be a broadening expressed in Hz.

    shifted : float or `quantity`

        Shift the data time origin by this amount. If it is not a quantity
        it is assumed to be expressed in the data units of the last
        dimension.

    inv : bool, optional

        True for inverse apodization.  False (default) for standard.

    rev : bool, optional.

        True to reverse the apodization before applying it to the data.

    apply : `bool`, optional, default = `True`

        Should we apply the calculated apodization to the dataset (default)
        or just return the apodization ndarray.

    inplace : `bool`, optional, default = `True`

        Should we make the transform in place or return a new dataset

    axis : optional, default is -1

    Returns
    -------
    out : :class:`~spectrochempy.dataset.nddataset.NDDataset`.
        The apodized dataset if apply is True, the apodization array if not True.

    """
    args = list(args)  # important (args is a tuple)

    # lb broadening
    lb = kwargs.get('lb', 0)
    if lb == 0:
        # let's try the args if the kwargs was not passed
        if len(args) > 0:
            lb = args.pop(0)

    # gb broadening
    gb = kwargs.get('gb', 0)
    if gb == 0:
        # let's try the second args if the kwargs was not passed
        if len(args) > 0:
            gb = args.pop(0)

    # shifted ?
    shifted = kwargs.pop('shifted', 0)

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

        return np.exp(e - g ** 2).data

    kwargs['method'] = func
    kwargs['apod'] = lb
    kwargs['apod2'] = gb
    kwargs['shifted'] = shifted

    out = apodize(source, **kwargs)

    return out
