# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2016 Christian Fernandez
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ["sp", "sine", "sinm", "qsin"]

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
# sinm function
# ======================================================================================================================


def sp(dataset, *args, **kwargs):
    r"""Calculate apodization with a Sine window multiplication

    Functional form of apodization window (cfBruker TOPSPIN manual):

    .. math::
        sp(t) = \sin(\frac{(\pi - \phi) t }{\text{aq}} + \phi)^{pow}
        
    where :math:`0 \lt t \lt \text{aq}` and  :math:`\phi = \pi ⁄ \text{sbb}` when :math:`\text{ssb} \ge 2` or
    :math:`\phi = 0` when :math:`\text{ssb} \lt 2`
        
    :math:`\text{aq}` is an acquisition status parameter and :math:`\text{ssb}` is a processing parameter (see the
    `ssb` parameter definition below) and :math:`{pow}` is an exponent equal to 1 for a sine bell window
    or 2 for a squared sine bell window.
    
    Parameters
    ----------
    dataset : |NDDataset|.
        Dataset we want to apodize using Sine Bell or Squared Sine Bell window multiplication
    sbb : int or float, optional, default=1.
        This processing parameter mimics the behaviour of the SSB parameter on bruker TOPSPIN software:
        Typical values are 1 for a pure sine function and 2 for a pure cosine function.
        Values greater than 2 give a mixed sine/cosine function. Note that all values smaller than 2, for example 0,
        have the same effect as :math:`\text{ssb}==1`, namely a pure sine function.
    pow : enum [1,2], optional, default=1
        exponent value - If pow==2 a Squared Sine Bell window multiplication is performed.
        
    Other Parameters
    ----------------
    inv : bool, optional
        True for inverse apodization.  False (default) for standard.
    rev : bool, optional.
        True to reverse the apodization before applying it to the data.
    apply : `bool`, optional, default=True
        Should we apply the calculated apodization to the dataset (default)
        or just return the apodization ndarray.
    inplace : `bool`, optional, default=True
        Should we make the transform in place or return a new dataset
    axis : optional, default is -1

    Returns
    -------
    out : |NDDataset|.
        The apodized dataset if apply is True, the apodization array if not True.

    """
    args = list(args)  # important (args is a tuple)
    
    # ssb
    ssb = kwargs.pop('ssb', None)
    if ssb is None:
        # let's try the args if the kwargs was not passed
        if len(args) > 0:
            ssb = args.pop(0)
        else:
            ssb = 1.
    if ssb<1.:
        ssb = 1.

    # pow
    pow = kwargs.pop('pow', 1)
    pow = 2 if int(pow)%2 == 0 else 1
    
    # func
    
    def func(x, ssb, pow):
        aq = (x.data[-1] - x.data[0])
        t = x.data / aq
        if ssb<2:
            phi = 0.
        else:
            phi = np.pi / ssb
        return np.sin((np.pi-phi)*t+phi ) ** pow
    
    # call the generic apodization function
    out, apodcurve = apodize(dataset, func, ssb, pow, **kwargs)

    if kwargs.pop('retfunc', False) :
        return out, apodcurve
    return out


def sine(dataset, *args, **kwargs):
    """
    Equivalent to :meth:``sp``.
    
    """
    return sp(dataset, *args, **kwargs)

def sinm(dataset, ssb=1, **kwargs):
    """
    Equivalent to :meth:``sp``., with pow = 1 (sine bell apodization window).

    """
    return sp(dataset, ssb=ssb, pow=1, **kwargs)


def qsin(dataset, ssb=1, **kwargs):
    """
    Equivalent to :meth:``sp``., with pow = 2 (squared sine bell apodization window).

    """
    return sp(dataset, ssb=ssb, pow=2, **kwargs)



# ======================================================================================================================
if __name__ == '__main__': # pragma: no cover
    
    from spectrochempy import *
    
    dataset1D = NDDataset()
    path = os.path.join(general_preferences.datadir, 'nmrdata', 'bruker', 'tests', 'nmr', 'bruker_1d')
    dataset1D.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    
    dataset1D /= dataset1D.real.data.max()  # normalize
    
    p = dataset1D.plot()
    
    new, curve = dataset1D.sinm(ssb=2, retfunc=True)
    curve.plot(color='r', clear=False)
    new.plot(xlim=(0, 50000), zlim=(-2, 2), data_only=True, color='r', clear=False)

    new, curve = dataset1D.sinm(ssb=1, retfunc=True)
    curve.plot(color='b', clear=False)
    new.plot(xlim=(0, 50000), zlim=(-2, 2), data_only=True, color='b', clear=False)

    new, curve = dataset1D.qsin(ssb=2, retfunc=True)
    curve.plot(color='m', clear=False)
    new.plot(xlim=(0, 50000), zlim=(-2, 2), data_only=True, color='m', clear=False)
    
    new, curve = dataset1D.qsin(ssb=1, retfunc=True)
    curve.plot(color='g', clear=False)
    new.plot(xlim=(0, 50000), zlim=(-2, 2), data_only=True, color='g', clear=False)

    show()
    
