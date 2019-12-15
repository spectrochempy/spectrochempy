# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ["em"]

__dataset_methods__ = __all__

# ======================================================================================================================
# Third party imports
# ======================================================================================================================
import numpy as np

# ======================================================================================================================
# Local imports
# ======================================================================================================================
from spectrochempy.core.processors.apodization.apodize import apodize
from spectrochempy.core import error_
from spectrochempy.utils import epsilon
from spectrochempy.utils import docstrings
from spectrochempy.units.units import ur, Quantity

# ======================================================================================================================
# em function
# ======================================================================================================================
docstrings.delete_params('apodize.parameters', 'dataset', 'method', 'apod')

@docstrings.get_sectionsf('em')
@docstrings.dedent
def em(dataset, lb=0*ur.Hz, shifted=0, inv=False, rev=False, inplace=False, dim=-1, **kwargs):
    r"""
    Calculate exponential apodization.

    For multidimensional NDDataset or NDPanels,
    the apodization is by default performed on the last dimension.
    
    The data in the last dimension MUST be time-domain,
    or an error is raised.

    Functional form of apodization window :

    .. math::
        em(t) = \exp(- e (t-t_0) )
        
    where
    
    .. math::
        e = \pi * lb
        

    Parameters
    ----------
    dataset : |NDDataset|  or |NDPanel|.
        Dataset we want to apodize using an Exponential Multiplication (EM)
    lb : float or |Quantity|, optional, default=1 Hz.
        Exponential line broadening,
        If it is not a quantity with units,
        it is assumed to be a broadening expressed in Hz.
    shifted : float or `quantity`, optional, default=0 us
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
        
     # is it a shifted broadening?
    if not isinstance(shifted, Quantity):
        # we default to microsecond units
        shifted = shifted * ur.us
    if shifted.magnitude < 0.:
        shifted = 0. * ur.us

    def func(x, lb, shifted) :
        e = np.ones_like(x)
        if lb.magnitude <= epsilon :
            return e
        else :
            if (x.unitless or x.dimensionless or
                    x.units.dimensionality != '[time]'):
                error_('em apply only to dimensions with [time] dimensionality\n'
                       'em processing was thus cancelled')
                return e
            units = x.units
            tc = 1. / lb
            tc = tc.to(units)
            shifted = shifted.to(units)
            e = np.pi * np.abs(x - shifted) / tc
            return np.exp(-e.data)
    
    # call the generic apodization function
    out, apodcurve = apodize(dataset, func, (lb, shifted), inv=inv, rev=rev, inplace=inplace, dim=dim, **kwargs)

    # Should we return the apodization
    if kwargs.pop('retfunc', False) :
        apodcurve = type(out)(apodcurve, coords=[out.coords(out.dims[-1])])  # make a dataset from the ndarray apodcurve
        return out, apodcurve
    
    return out


# ======================================================================================================================
if __name__ == '__main__' : # pragma: no cover
    pass