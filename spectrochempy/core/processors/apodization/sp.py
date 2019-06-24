# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2016 Christian Fernandez
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ["sp"]

__dataset_methods__ = __all__

# ======================================================================================================================
# Third party imports
# ======================================================================================================================
import numpy as np

# ======================================================================================================================
# Local imports
# ======================================================================================================================
from spectrochempy.core.processors.apodization.apodize import apodize
from spectrochempy.units.units import Quantity


# ======================================================================================================================
# sp function
# ======================================================================================================================

def sp(dataset, *args, **kwargs):
    r"""Calculate a Shifted sine-bell apodization

    Functional form of apodization window :

    .. math::
        sp(x) = \sin(\frac{pi * off + pi * (end - off) * x} {size - 1})^{pow}

    Parameters
    ----------
    dataset : |NDDataset|.
        Dataset we want to apodize using exponential multiplication
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
    apply : `bool`, optional, default= True
        Should we apply the calculated apodization to the dataset (default)
        or just return the apodization ndarray.
    inplace : `bool`, optional, default= True
        Should we make the transform in place or return a new dataset
    axis : optional, default is -1

    Returns
    -------
    out : |NDDataset|.
        The apodized dataset if apply is True, the apodization array if not True.

    """
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

    # apod func (must be func(x, tc1, tc2, shifted, pow) form
    def func(x, off, end, shifted, pow):
        wx = (x.data[-1] - x.data[0])
        off = off.to(x.units).magnitude
        end = end.to(x.units).magnitude
        return np.sin(np.pi * off/wx + np.pi * (end - off) * x.data/wx**2 ) ** pow

    kwargs['method'] = func
    kwargs['apod'] = off
    kwargs['apod2'] = end
    kwargs['pow'] = pow

    out = apodize(dataset, **kwargs)

    return out

def sp2(dataset, *args, **kwargs):
    """Calculate a Shifted sine-bell apodization

    Functional form of apodization window :

    .. math::
        sp(x) = \\sin(\\frac{pi * off + pi * (end - off) * x} {size - 1})^{pow}

    Parameters
    ----------
    dataset : |NDDataset|.
        Dataset we want to apodize using exponential multiplication
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
    apply : `bool`, optional, default= True
        Should we apply the calculated apodization to the dataset (default)
        or just return the apodization ndarray.
    inplace : `bool`, optional, default= True
        Should we make the transform in place or return a new dataset
    axis : optional, default is -1

    Returns
    -------
    out : |NDDataset|.
        The apodized dataset if apply is True, the apodization array if not True.

    """
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
    
    # apod func (must be func(x, tc1, tc2, shifted, pow) form
    def func(x, off, end, shifted, pow):
        wx = (x.data[-1] - x.data[0])
        off = off.to(x.units).magnitude / wx
        end = end.to(x.units).magnitude / wx
        print (off, end)
        xn = x.data / wx
        return np.sin(np.pi * off + np.pi * (end - off) * xn ) ** pow
    
    kwargs['method'] = func
    kwargs['apod'] = off
    kwargs['apod2'] = end
    kwargs['pow'] = pow
    
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
    
    new, curve = dataset1D.sp2(20000.* ur.us, 60000. * ur.us, retfunc=True)
    
    curve.plot(color='r', clear=False)
    new.plot(xlim=(0, 50000), zlim=(-2, 2), data_only=True, color='r', clear=False)
    
    show()
    
