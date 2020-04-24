# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ['em', 'gm', "sp", "sine", "sinm", "qsin"]
__dataset_methods__ = __all__

# ======================================================================================================================
# Third party imports
# ======================================================================================================================
import numpy as np

# ======================================================================================================================
# Local imports
# ======================================================================================================================
from ...units import ur, Quantity
from ...utils import epsilon, docstrings
from .. import general_preferences, error_

# ======================================================================================================================
# Private module methods
# ======================================================================================================================
@docstrings.get_sectionsf('apodize', sections=['Other Parameters','Returns'])
@docstrings.dedent
def _apodize(dataset, method, apod, **kwargs):
    """
    
    Calculate an apodization window function using the given `method` and the `apod` parameters
    
    Parameters
    ----------
    dataset : |NDDataset| or |NDPanel|.
        Input dataset or panel
    method : str
        The method of apodization
    apod : tuple of float or |Quantity|
        The apodization parameters to be used depending on the apodization method

    Other Parameters
    ----------------
    dim : str or int, keyword parameter, optional, default='x'.
        Specify on which dimension to apply the apodization method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    inv : bool, keyword parameter, optional, default=False.
        True for inverse apodization.
    rev : bool, keyword parameter, optional, default=False.
        True to reverse the apodization before applying it to the data.
    inplace : bool, keyword parameter, optional, default=False.
        True if we make the transform inplace.  If False, the function return a new dataset
    retfunc : bool, keyword parameter, optional, default=False
        True to return the apodization array along with the apodized object
    
    Returns
    -------
    out : |NDDataset| or |NDPanel|.
        The apodized dataset
    apod_arr : ndarray,
        The apodization array only if 'retfunc' is True.

    """

    # output dataset inplace (by default) or not
    if not kwargs.pop('inplace', False):
        new = dataset.copy()  # copy to be sure not to modify this dataset
    else:
        new = dataset
        
    # On which axis do we want to apodize? (get axis from arguments)
    axis, dim = dataset.get_axis(**kwargs, negative_axis=True)
    
    # The last dimension is always the dimension on which we apply the apodization window.
    # If needed, we swap the dimensions to be sure to be in this situation
    
    swaped = False
    if axis != -1:
        new.swapaxes(axis, -1, inplace=True)  # must be done in  place
        swaped = True
    
    x = new.coords[dim]
    if (x.unitless or x.dimensionless or
            x.units.dimensionality != '[time]'):
        error_('em apply only to dimensions with [time] dimensionality\n'
               'em processing was thus cancelled')
        apod_arr = 1.
    else:
        # compute the apodization function
        name = method.__module__.split('.')[-1]
        new.history = f'{name} apodization performed on dimension {dim} with parameters:' + str(apod)
        apod_arr = method(x, *apod)
    
        if kwargs.pop('rev', False):
            apod_arr = apod_arr[::-1]  # reverse apodization
        
        if kwargs.pop('inv', False):
            apod_arr = 1. / apod_arr  # invert apodization
    
    # if we are in NMR we have an additional complication due to the mode
    # of acquisition (sequential mode when ['QSEQ','TPPI','STATES-TPPI'])
    # TODO: CHECK IF THIS WORK WITH 2D DATA - IMPORTANT - CHECK IN PARTICULAR IF SWAPING ALSO SWAP METADATA (NOT SURE FOR NOW)
    iscomplex = new.is_complex
    isquaternion = new.is_quaternion
    encoding = new.meta.encoding[-1]
    # TODO: handle this eventual complexity
    
    new._data *= apod_arr
    
    # restore original data order if it was swaped
    if swaped:
        new.swapaxes(axis, -1, inplace=True)  # must be done inplace
    
   
    
    # TODO: improve display of apod parameters
    
    return new, apod_arr
    
    # shifted = args.shifted  # float(kargs.get('top', 0.0))
    # k_shifted = args.k_shifted
    #
    # if k_shifted and (axis == -1 or axis == 1) and dataset.is_2d:
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
    #         ratio = k_ratio(dataset, p, q)
    #     shifted = np.abs(ratio)
    #     if args.verbose:
    #         print('k_ratio: %.3f' % ratio)
    #
    # par.LB = lb
    #
    # kargs['lb'] = lb / dataset.get_multiplicator(axis)
    # kargs['shifted'] = shifted
    # kargs['states'] = True if 'STATES' in par.encoding else False
    #
    
    #
    # if axis == 0:
    #     # transpose temporarily the data for indirect dimension ft
    #     datv = datv.T
    #
    # dataset.data = datv
    # dataset.history.append('Exponential apodization lb:%.2f' % par.LB)


# ======================================================================================================================
# Public module methods
# ======================================================================================================================
@docstrings.dedent
def em(dataset, lb=1, shifted=0, **kwargs):
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
    dataset : |NDDataset| or |NDPanel|.
        Input dataset or panel
    lb : float or |Quantity|, optional, default=1 Hz.
        Exponential line broadening,
        If it is not a quantity with units,
        it is assumed to be a broadening expressed in Hz.
    shifted : float or `quantity`, optional, default=0 us
        Shift the data time origin by this amount. If it is not a quantity
        it is assumed to be expressed in the data units of the last
        dimension.
        
    Other Parameters
    ----------------
    %(apodize.other_parameters)s
    
    Returns
    -------
    %(apodize.returns)s
    
    See Also
    --------
    gm, sp, sine, sinm, qsin
    
    """
    
    # what's the line broadening ?
    if not isinstance(lb, Quantity):
        # we default to Hz units
        lb = lb * ur.Hz
    
    # is it a shifted broadening?
    if not isinstance(shifted, Quantity):
        # we default to microsecond units
        shifted = shifted * ur.us
    if shifted.magnitude < epsilon:
        shifted = 0. * ur.us
    
    def func(x, lb, shifted):
        e = np.ones_like(x)
        if lb.magnitude <= epsilon:
            return e
        else:
            units = x.units
            tc = 1. / lb
            tc = tc.to(units)
            shifted = shifted.to(units)
            e = np.pi * np.abs(x - shifted) / tc
            return np.exp(-e.data)
    
    # Call the generic apodization function
    out, apodcurve = _apodize(dataset, func, (lb, shifted), **kwargs)
    
    # Should we return the apodization array?
    if kwargs.pop('retfunc', False):
        apodcurve = type(out)(apodcurve, coords=[out.coords(out.dims[-1])])  # make a dataset from the ndarray apodcurve
        return out, apodcurve
    
    return out

# ======================================================================================================================
@docstrings.dedent
def gm(dataset, gb=1, lb=0, shifted=0, **kwargs):
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
    lb : float or `quantity`, optional, default=0 Hz.
        inverse exponential width.
        If it is not a quantity with units,
        it is assumed to be a broadening expressed in Hz.
    gb : float or `quantity`, optional, default=1 Hz.
        Gaussian broadening width.
        If it is not a quantity with units,
        it is assumed to be a broadening expressed in Hz.
    shifted : float or `quantity`, optional, default=0 us
        Shift the data time origin by this amount. If it is not a quantity
        it is assumed to be expressed in the data units of the last
        dimension.
        
    Other Parameters
    ----------------
    %(apodize.other_parameters)s
    
    Returns
    -------
    %(apodize.returns)s

    See Also
    --------
    em, sp, sine, sinm, qsin
    
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
        if abs(lb.magnitude) and abs(gb.magnitude) <= epsilon:
            return g
        else:
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
    out, apodcurve = _apodize(dataset, func, (gb, lb, shifted), **kwargs)
    
    if kwargs.pop('retfunc', False):
        apodcurve = type(out)(apodcurve, coords=[out.coords(out.dims[-1])])  # make a dataset from the ndarray apodcurve
        return out, apodcurve
    
    return out


# ======================================================================================================================
@docstrings.dedent
def sp(dataset, ssb=1, pow=1, **kwargs):
    r"""
    Calculate apodization with a Sine window multiplication.
    
    For multidimensional NDDataset or NDPanels,
    the apodization is by default performed on the last dimension.

    Functional form of apodization window (cfBruker TOPSPIN manual):

    .. math::
        sp(t) = \sin(\frac{(\pi - \phi) t }{\text{aq}} + \phi)^{pow}
        
    where :math:`0 < t < \text{aq}` and  :math:`\phi = \pi ⁄ \text{sbb}` when :math:`\text{ssb} \ge 2` or
    :math:`\phi = 0` when :math:`\text{ssb} < 2`
    
    :math:`\text{aq}` is an acquisition status parameter and :math:`\text{ssb}` is a processing parameter (see the
    `ssb` parameter definition below) and :math:`\text{pow}` is an exponent equal to 1 for a sine bell window
    or 2 for a squared sine bell window.
    
    Parameters
    ----------
    dataset : |NDDataset| or |NDPanel|.
        Dataset we want to apodize using Sine Bell or Squared Sine Bell window multiplication
    sbb : int or float, optional, default=1.
        This processing parameter mimics the behaviour of the SSB parameter on bruker TOPSPIN software:
        Typical values are 1 for a pure sine function and 2 for a pure cosine function.
        Values greater than 2 give a mixed sine/cosine function. Note that all values smaller than 2, for example 0,
        have the same effect as :math:`\text{ssb}=1`, namely a pure sine function.
    pow : enum [1,2], optional, default=1.
        exponent value - If pow=2 a Squared Sine Bell window multiplication is performed.

    Other Parameters
    ----------------
    %(apodize.other_parameters)s
    
    Returns
    -------
    %(apodize.returns)s
    
    See Also
    --------
    em, gm, sine, sinm, qsin

    """
    
    # ssb
    ssb = kwargs.pop('ssb', ssb)
    if ssb < 1.:
        ssb = 1.
    
    # pow
    pow = kwargs.pop('pow', pow)
    pow = 2 if int(pow) % 2 == 0 else 1
    
    # func
    
    def func(x, ssb, pow):
        aq = (x.data[-1] - x.data[0])
        t = x.data / aq
        if ssb < 2:
            phi = 0.
        else:
            phi = np.pi / ssb
        return np.sin((np.pi - phi) * t + phi) ** pow
    
    # call the generic apodization function
    out, apodcurve = _apodize(dataset, func, (ssb, pow), **kwargs)
    
    if kwargs.pop('retfunc', False):
        apodcurve = type(out)(apodcurve, coords=[out.coords(out.dims[-1])])  # make a dataset from the ndarray apodcurve
        return out, apodcurve
    return out


# ======================================================================================================================
def sine(dataset, *args, **kwargs):
    """
    Strictly equivalent to :meth:`sp`.
    
    See Also
    --------
    em, gm, sp, sinm, qsin
    
    """
    return sp(dataset, *args, **kwargs)


# ======================================================================================================================
def sinm(dataset, ssb=1, **kwargs):
    """
    Equivalent to :meth:`sp`, with pow = 1 (sine bell apodization window).
    
    See Also
    --------
    em, gm, sp, sine, qsin
    
    """
    return sp(dataset, ssb=ssb, pow=1, **kwargs)


# ======================================================================================================================
def qsin(dataset, ssb=1, **kwargs):
    """
    Equivalent to :meth:`sp`, with pow = 2 (squared sine bell apodization window).
    
    See Also
    --------
    em, gm, sp, sine, sinm

    """
    return sp(dataset, ssb=ssb, pow=2, **kwargs)

