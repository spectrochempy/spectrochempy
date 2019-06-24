# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ["gm"]

__dataset_methods__ = __all__

# ======================================================================================================================
# Third party imports
# ======================================================================================================================
import numpy as np

# ======================================================================================================================
# Local imports
# ======================================================================================================================
from spectrochempy.core.processors.apodization.apodize import apodize
from spectrochempy.utils import epsilon


# ======================================================================================================================
# gm function
# ======================================================================================================================

def gm(dataset, *args, **kwargs):
    """Calculate a Lorentz-to-Gauss apodization

    Functional form of apodization window :

    .. math::
        gm(x) = \exp(e) * \exp(- g^2)

    Where :

    .. math::
        e = \pi * lb * t
        
    and
    
    .. math::
        g = 0.6 * \pi * gb * (t - t_0)

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
    retfunc : `bool`, optional, default=False
        Should we return the calculated apodization function with the dataset.
    inplace : `bool`, optional, default=True
        Should we make the transform in place or return a new dataset
    axis : optional, default is -1

    Returns
    -------
    out : |NDDataset|.
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
    def func(x, tc1, tc2, shifted, pow):
        # pow is not used
        if tc1.magnitude > epsilon:
            e = np.pi * x / tc1
        else:
            e = np.zeros_like(x)
        if tc2.magnitude > epsilon:
            g = 0.6 * np.pi * (x - shifted) / tc2
        else:
            g = np.zeros_like(x)

        return np.exp(e - g ** 2).data

    kwargs['method'] = func
    kwargs['apod'] = lb
    kwargs['apod2'] = gb
    kwargs['shifted'] = shifted

    out = apodize(dataset, **kwargs)

    return out

# ======================================================================================================================
if __name__ == '__main__': # pragma: no cover
    from spectrochempy import *
    
    dataset1D = NDDataset()
    path = os.path.join(general_preferences.datadir, 'nmrdata', 'bruker', 'tests', 'nmr', 'bruker_1d')
    dataset1D.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    
    dataset1D /= dataset1D.real.data.max()  # normalize
    
    p = dataset1D.plot()
    
    new, curve = dataset1D.gm(lb=-100. * ur.Hz, gb=200. * ur.Hz, retfunc=True)
    
    curve.plot(color='r', clear=False)
    new.plot(xlim=(0, 25000), zlim=(-2, 2), data_only=True, color='r', clear=False)

    show()
    
    new, curve = dataset1D.gm(lb=-100. * ur.Hz, gb=200. * ur.Hz, shifted=2500., retfunc=True)

    curve.plot(color='r', clear=False)
    new.plot(xlim=(0, 25000), zlim=(-2, 2), data_only=True, color='r', clear=False)
    
    show()
