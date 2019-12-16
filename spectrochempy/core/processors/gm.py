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
from .apodization import apodize
from ...utils import epsilon, docstrings
from ...units import ur, Quantity

# ======================================================================================================================
# gm function
# ======================================================================================================================
docstrings.delete_params('apodize.parameters', 'dataset', 'method', 'apod')

@docstrings.get_sectionsf('gm')
@docstrings.dedent
def gm(dataset, gb=1*ur.Hz, lb=0*ur.Hz, shifted=0, inv=False, rev=False, inplace=False, dim=-1,  **kwargs):
    r"""
    Calculate lorentz-to-gauss apodization.

    Functional form of apodization window :

    .. math::
        gm(t) = \exp(e  - g^2 )

    where :

    .. math::
        e = \pi * lb *  (t - t0)

        
    and
    
    .. math::
        g = 0.6 * \pi * gb * (t - t0)

    Parameters
    ----------
    dataset : |NDDataset|  or |NDPanel|.
        Dataset we want to apodize using an Lorentz Multiplication
    lb : float or `quantity`
        inverse exponential width.
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
    %(apodize.parameters.no_dataset|method|apod)s

    Returns
    -------
    %(apodize.returns)s

    """

    # what's the line broadening ?
    if not isinstance(lb, Quantity):
        # we default to Hz units
        lb = lb * ur.Hz
    if not isinstance(gb, Quantity):
        # we default to Hz units
        gb = gb * ur.Hz

    # is it a shifted broadening?
    if not isinstance(shifted, Quantity):
        # we default to microsecond units
        shifted = shifted * ur.us
    if shifted.magnitude < 0.:
        shifted = 0. * ur.us

    def func(x, gb, lb, shifted):
        
        g = np.ones_like(x)
        if (x.unitless or x.dimensionless or
                x.units.dimensionality != '[time]'):
            error_('gm apply only to dimensions with [time] dimensionality\n'
                   'gm processing was thus cancelled')
            return g
        units = x.units
        shifted = shifted.to(units)
        xs = np.pi * np.abs(x - shifted)
        if abs(lb.magnitude) > epsilon:
            tc1 = 1. / lb
            tc1 = tc1.to(units)
            e = x / tc1
        else:
            e = np.zeros_like(x)
        if gb.magnitude > epsilon:
            tc2 = 1. / gb
            tc2 = tc2.to(units)
            g = 0.6 * xs / tc2
        else:
            g = np.zeros_like(x)

        return np.exp(e - g ** 2).data

    # call the generic apodization function
    out, apodcurve = apodize(dataset, func, (gb, lb, shifted), inv=inv, rev=rev, inplace=inplace, dim=dim, **kwargs)

    if kwargs.pop('retfunc', False):
        apodcurve = type(out)(apodcurve, coords=[out.coords(out.dims[-1])])  # make a dataset from the ndarray apodcurve
        return out, apodcurve

    return out

# ======================================================================================================================
if __name__ == '__main__': # pragma: no cover
    from spectrochempy import *
    
    dataset1D = NDDataset()
    path = os.path.join(general_preferences.datadir, 'nmrdata', 'bruker', 'tests', 'nmr', 'bruker_1d')
    dataset1D.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    
    dataset1D /= dataset1D.real.data.max()  # normalize
    
    p = dataset1D.plot()

    dd = dataset1D.copy()
    new, curve = dd.gm(lb=100. * ur.Hz, gb=200. * ur.Hz, retfunc=True)
    curve.plot(color='r', clear=False)
    new.plot(xlim=(0, 25000), zlim=(-2, 2), data_only=True, color='r', clear=False)

    dd = dataset1D.copy()
    new, curve = dd.gm(lb=200. * ur.Hz, gb=200. * ur.Hz, retfunc=True)
    curve.plot(color='k', clear=False)
    new.plot(xlim=(0, 25000), zlim=(-2, 2), data_only=True, color='k', clear=False)

    dd = dataset1D.copy()
    new, curve = dd.gm(lb=100. * ur.Hz, gb=200. * ur.Hz, shifted=2500., retfunc=True)
    curve.plot(color='g', clear=False)
    new.plot(xlim=(0, 25000), zlim=(-2, 2), data_only=True, color='g', clear=False)
    
    dd = dataset1D.copy()
    new, curve = dd.gm(lb=0. * ur.Hz, gb=200. * ur.Hz, shifted=2500., retfunc=True)
    curve.plot(color='b', clear=False)
    new.plot(xlim=(0, 25000), zlim=(-2, 2), data_only=True, color='b', clear=False)

    show()
