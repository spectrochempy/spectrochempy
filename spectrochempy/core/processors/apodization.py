# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================

__all__ = [
    "em",
    "gm",
    "sp",
    "sine",
    "sinm",
    "qsin",
    "general_hamming",
    "hamming",
    "hann",
    "triang",
    "bartlett",
    "blackmanharris",
]
__dataset_methods__ = __all__

import functools
import numpy as np
from scipy.signal import windows

from spectrochempy.units import Quantity
from spectrochempy.utils import EPSILON
from spectrochempy.core import error_

pi = np.pi


# ======================================================================================================================
# Decorators
# ======================================================================================================================


def _apodize_method(**units):
    # Decorator to set units of parameters according to dataset units

    def decorator_apodize_method(method):
        @functools.wraps(method)
        def wrapper(dataset, **kwargs):

            # what to return
            retapod = kwargs.pop("retapod", False)
            dryrun = kwargs.pop("dryrun", False)
            # is_nmr = dataset.origin.lower() in ["topspin", ]
            is_ir = dataset.origin.lower() in ["omnic", "opus"]

            # On which axis do we want to apodize? (get axis from arguments)
            axis, dim = dataset.get_axis(**kwargs, negative_axis=True)

            # output dataset inplace (by default) or not
            if not kwargs.pop("inplace", False) and not dryrun:
                new = dataset.copy()  # copy to be sure not to modify this dataset
            else:
                new = dataset

            # The last dimension is always the dimension on which we apply the apodization window.
            # If needed, we swap the dimensions to be sure to be in this situation
            swaped = False
            if axis != -1:
                new.swapdims(axis, -1, inplace=True)  # must be done in  place
                swaped = True

            # Get the coordinates for the last dimension
            x = new.coordset[dim]
            if hasattr(x, "_use_time_axis"):
                store = x._use_time_axis
                x._use_time_axis = True  # we need to have dimentionless or time units

            # check if the dimensionality is compatible with this kind of functions
            if x.unitless or x.dimensionless or x.units.dimensionality == "[time]":

                # Set correct units for parameters
                dunits = dataset.coordset[dim].units
                apod = {}
                for key, default_units in units.items():

                    if key not in kwargs or default_units is None:
                        continue

                    par = kwargs[key]

                    if isinstance(par, str):
                        par = Quantity(par)

                    if not isinstance(par, Quantity):
                        # set to default units
                        par *= Quantity(1.0, default_units)

                    apod[key] = par
                    if par.dimensionality == 1 / dunits.dimensionality:
                        kwargs[key] = 1.0 / (1.0 / par).to(dunits)
                    else:
                        kwargs[key] = par.to(dunits)

                    kwargs[key] = kwargs[key].magnitude

                # Call to the apodize function
                # ----------------------------

                # now call the method with unitless parameters
                if is_ir:
                    # we must apodize at the top of the interferogram.
                    zpd = int(np.argmax(new.data, -1))
                    dist2end = x.size - zpd
                    apod_arr = method(np.empty(2 * dist2end), **kwargs)
                    apod_arr = apod_arr[-x.size :]
                else:
                    apod_arr = method(x.data, **kwargs)

                if kwargs.pop("rev", False):
                    apod_arr = apod_arr[::-1]  # reverse apodization

                if kwargs.pop("inv", False):
                    apod_arr = 1.0 / apod_arr  # invert apodization

                if not dryrun:
                    new.history = (
                        f"`{method.__name__}` apodization performed on dimension `{dim}` "
                        f"with parameters: {apod}"
                    )

                # Apply?
                if not dryrun:
                    new._data *= apod_arr

            else:  # not (x.unitless or x.dimensionless or x.units.dimensionality != '[time]')
                error_(
                    "This method apply only to dimensions with [time] or [dimensionless] dimensionality.\n"
                    "Apodization processing was thus cancelled"
                )
                apod_arr = 1.0

            # restore original data order if it was swaped
            if swaped:
                new.swapdims(axis, -1, inplace=True)  # must be done inplace

            if hasattr(x, "_use_time_axis"):
                new.x._use_time_axis = store

            if retapod:
                apodcurve = type(new)(apod_arr, coordset=[x])
                return new, apodcurve
            else:
                return new

        return wrapper

    return decorator_apodize_method


# ======================================================================================================================
# Public module methods
# ======================================================================================================================


@_apodize_method(lb="Hz", shifted="us")
def em(dataset, lb=1, shifted=0, **kwargs):
    r"""
    Calculate exponential apodization.

    For multidimensional NDDataset,
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
    dataset : Dataset
        Input dataset.
    lb : float or |Quantity|, optional, default=1 Hz
        Exponential line broadening,
        If it is not a quantity with units, it is assumed to be a broadening expressed in Hz.
    shifted : float or `quantity`, optional, default=0 us
        Shift the data time origin by this amount. If it is not a quantity
        it is assumed to be expressed in the data units of the last
        dimension.

    Returns
    -------
    apodized
        Dataset.
    apod_arr
        The apodization array only if 'retapod' is True.

    Other Parameters
    ----------------
    dim : str or int, keyword parameter, optional, default='x'
        Specify on which dimension to apply the apodization method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    inv : bool, keyword parameter, optional, default=False
        True for inverse apodization.
    rev : bool, keyword parameter, optional, default=False
        True to reverse the apodization before applying it to the data.
    inplace : bool, keyword parameter, optional, default=False
        True if we make the transform inplace.  If False, the function return a new dataset.
    retapod : bool, keyword parameter, optional, default=False
        True to return the apodization array along with the apodized object.

    See Also
    --------
    gm, sp, sine, sinm, qsin, hamming, triang, bartlett, blackmanharris
    """

    # units are set by a decorator

    x = dataset

    e = np.ones_like(x)

    if abs(lb) <= EPSILON:
        return e
    if shifted < EPSILON:
        shifted = 0.0

    tc = 1.0 / lb
    xs = pi * np.abs(x - shifted)
    e = xs / tc

    return np.exp(-e)


# ..............................................................................
@_apodize_method(gb="Hz", lb="Hz", shifted="us")
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
    dataset : ndarray
        Dataset we want to apodize using an Lorentz Multiplication.
    lb : float or `quantity`, optional, default=0 Hz
        Inverse exponential width.
        If it is not a quantity with units,
        it is assumed to be a broadening expressed in Hz.
    gb : float or `quantity`, optional, default=1 Hz
        Gaussian broadening width.
        If it is not a quantity with units,
        it is assumed to be a broadening expressed in Hz.
    shifted : float or `quantity`, optional, default=0 us
        Shift the data time origin by this amount. If it is not a quantity
        it is assumed to be expressed in the data units of the last
        dimension.

    Returns
    -------
    apodized
        Dataset.
    apod_arr
        The apodization array only if 'retapod' is True.

    Other Parameters
    ----------------
    dim : str or int, keyword parameter, optional, default='x'
        Specify on which dimension to apply the apodization method. If `dim` is specified as an integer it is
        equivalent to the usual `axis` numpy parameter.
    inv : bool, keyword parameter, optional, default=False
        True for inverse apodization.
    rev : bool, keyword parameter, optional, default=False
        True to reverse the apodization before applying it to the data.
    inplace : bool, keyword parameter, optional, default=False
        True if we make the transform inplace.  If False, the function return a new dataset
    retapod : bool, keyword parameter, optional, default=False
        True to return the apodization array along with the apodized object.

    See Also
    --------
    em, sp, sine, sinm, qsin, hamming, triang, bartlett, blackmanharris
    """

    x = dataset

    g = np.ones_like(x)

    if abs(lb) <= EPSILON and abs(gb) <= EPSILON:
        return g
    if shifted < EPSILON:
        shifted = 0.0

    xs = pi * np.abs(x - shifted)

    if abs(lb) > EPSILON:
        tc1 = 1.0 / lb
        e = x / tc1
    else:
        e = np.zeros_like(x)
    if abs(gb) > EPSILON:
        tc2 = 1.0 / gb
        g = 0.6 * xs / tc2
    else:
        g = np.zeros_like(x)

    return np.exp(e - g ** 2)


# ..............................................................................
@_apodize_method(ssb=None, pow=None)
def sp(dataset, ssb=1, pow=1, **kwargs):
    r"""
    Calculate apodization with a Sine window multiplication.

    For multidimensional NDDataset,
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
    dataset : Dataset
        Dataset we want to apodize using Sine Bell or Squared Sine Bell window multiplication.
    sbb : int or float, optional, default=1
        This processing parameter mimics the behaviour of the SSB parameter on bruker TOPSPIN software:
        Typical values are 1 for a pure sine function and 2 for a pure cosine function.
        Values greater than 2 give a mixed sine/cosine function. Note that all values smaller than 2, for example 0,
        have the same effect as :math:`\text{ssb}=1`, namely a pure sine function.
    pow : enum [1,2], optional, default=1
        Exponent value - If pow=2 a Squared Sine Bell window multiplication is performed.

    Other Parameters
    ----------------
    dim : str or int, keyword parameter, optional, default='x'
        Specify on which dimension to apply the apodization method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    inv : bool, keyword parameter, optional, default=False
        True for inverse apodization.
    rev : bool, keyword parameter, optional, default=False
        True to reverse the apodization before applying it to the data.
    inplace : bool, keyword parameter, optional, default=False
        True if we make the transform inplace.  If False, the function return a new dataset.
    retapod : bool, keyword parameter, optional, default=False
        True to return the apodization array along with the apodized object.

    Returns
    -------
    apodized
        Dataset.
    apod_arr
        The apodization array only if 'retapod' is True.

    See Also
    --------
    em, gm, sine, sinm, qsin, hamming, triang, bartlett, blackmanharris
    """

    x = dataset

    # ssb
    if ssb < 1.0:
        ssb = 1.0

    # pow
    pow = 2 if int(pow) % 2 == 0 else 1

    aq = x[-1] - x[0]
    t = x / aq
    if ssb < 2:
        phi = 0.0
    else:
        phi = np.pi / ssb
    return np.sin((np.pi - phi) * t + phi) ** pow


# ..............................................................................
def sine(dataset, *args, **kwargs):
    """
    Strictly equivalent to :meth:`sp`.

    See Also
    --------
    em, gm, sp, sinm, qsin, hamming, triang, bartlett, blackmanharris
    """
    return sp(dataset, *args, **kwargs)


# ..............................................................................
def sinm(dataset, ssb=1, **kwargs):
    """
    Equivalent to :meth:`sp`, with pow = 1 (sine bell apodization window).

    See Also
    --------
    em, gm, sp, sine, qsin, hamming, triang, bartlett, blackmanharris
    """
    return sp(dataset, ssb=ssb, pow=1, **kwargs)


# ..............................................................................
def qsin(dataset, ssb=1, **kwargs):
    """
    Equivalent to :meth:`sp`, with pow = 2 (squared sine bell apodization window).

    See Also
    --------
    em, gm, sp, sine, sinm, hamming, triang, bartlett, blackmanharris
    """
    return sp(dataset, ssb=ssb, pow=2, **kwargs)


# ..............................................................................
@_apodize_method(alpha=None)
def general_hamming(dataset, alpha, **kwargs):
    r"""
    Calculate generalized Hamming apodization.

    For multidimensional NDDataset,
    the apodization is by default performed on the last dimension.

    The data in the last dimension MUST be time-domain or dimensionless,
    otherwise an error is raised.

    Functional form of apodization window :

    .. math:: w(n) = \alpha - \left(1 - \alpha\right) \cos\left(\frac{2\pi{n}}{M-1}\right)
              \qquad 0 \leq n \leq M-1

    where M is the number of point of the input dataset.

    Both the common Hamming window and Hann window are special cases of the
    generalized Hamming window with :math:`\alpha` = 0.54 and :math:`\alpha` =
    0.5, respectively

    Parameters
    ----------
    dataset : array.
        Input dataset.
    alpha : float
        The window coefficient, :math:`\alpha`.

    Other Parameters
    ----------------
    dim : str or int, keyword parameter, optional, default='x'
        Specify on which dimension to apply the apodization method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    inv : bool, keyword parameter, optional, default=False
        True for inverse apodization.
    rev : bool, keyword parameter, optional, default=False
        True to reverse the apodization before applying it to the data.
    inplace : bool, keyword parameter, optional, default=False
        True if we make the transform inplace.  If False, the function return a new dataset.
    retapod : bool, keyword parameter, optional, default=False
        True to return the apodization array along with the apodized object.

    Returns
    -------
    apodized
        Dataset.
    apod_arr
        The apodization array only if 'retapod' is True.

    See Also
    --------
    gm, sp, sine, sinm, qsin, hamming, triang, bartlett, blackmanharris
    """

    x = dataset

    return windows.general_hamming(len(x), alpha, sym=True)


def hamming(dataset, **kwargs):
    """
    Calculate generalized Hamming (== Happ-Genzel) apodization.

    For multidimensional NDDataset,
    the apodization is by default performed on the last dimension.

    The data in the last dimension MUST be time-domain or dimensionless,
    otherwise an error is raised.

    Functional form of apodization window :

    .. math::
        w(n) = \alpha - (1 - \alpha)\cos(\frac{2\pi n}{M-1})\qquad 0\leq n\leq M-1

    where M is the number of point of the input dataset and :math:`\alpha` = 0.54.

    Parameters
    ----------
    dataset : array
        Input dataset.
    alpha : float
        The window coefficient, :math:`\alpha`.

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
    retapod : bool, keyword parameter, optional, default=False
        True to return the apodization array along with the apodized object

    Returns
    -------
    apodized
        Dataset.
    apod_arr
        The apodization array only if 'retapod' is True.

    See Also
    --------
    general_hamming, hann
    """

    return general_hamming(dataset, alpha=0.54)


def hann(dataset, **kwargs):
    """
    Return a Hann window.

    For multidimensional NDDataset,
    the apodization is by default performed on the last dimension.

    The data in the last dimension MUST be time-domain or dimensionless,
    otherwise an error is raised.

    Functional form of apodization window :

    .. math::
       w(n) = \alpha - (1 - \alpha) \cos(\frac{2\pi{n}}{M-1}) \qquad 0 \leq n \leq M-1

    where M is the number of point of the input dataset and :math:`\alpha` = 0.5

    Parameters
    ----------
    dataset : array.
        Input dataset.
    alpha : float
        The window coefficient, :math:`\alpha`.

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
    retapod : bool, keyword parameter, optional, default=False
        True to return the apodization array along with the apodized object

    Returns
    -------
    apodized
        Dataset.
    apod_arr
        The apodization array only if 'retapod' is True.

    See Also
    --------
    general_hamming, hamming
    """
    return general_hamming(dataset, alpha=0.5)


@_apodize_method()
def triang(dataset, **kwargs):
    r"""
    Calculate triangular apodization with non-null extremities and maximum value normalized to 1.

    For multidimensional NDDataset,
    the apodization is by default performed on the last dimension.

    The data in the last dimension MUST be time-domain or dimensionless,
    otherwise an error is raised.

    Parameters
    ----------
    dataset : array
        Input dataset.

    Returns
    -------
    apodized
        Dataset
    apod_arr
        The apodization array only if 'retapod' is True.

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
    retapod : bool, keyword parameter, optional, default=False
        True to return the apodization array along with the apodized object

    See Also
    --------
    gm, sp, sine, sinm, qsin, hamming, bartlett, blackmanharris
    """

    x = dataset

    return x * windows.triang(len(x), sym=True)


@_apodize_method()
def bartlett(dataset, **kwargs):
    """
    Calculate Bartlett apodization (triangular window with end points at zero).

    For multidimensional NDDataset,
    the apodization is by default performed on the last dimension.

    The data in the last dimension MUST be time-domain or dimensionless,
    otherwise an error is raised.

    The Bartlett window is defined as

    .. math::
       w(n) = \frac{2}{M-1} (\frac{M-1}{2} - |n - \frac{M-1}{2}|)

    where M is the number of point of the input dataset.

    Parameters
    ----------
    dataset : Dataset
        Input dataset.

    Returns
    -------
    apodized
        dataset.
    apod_arr
        The apodization array only if 'retapod' is True.

    Other Parameters
    ----------------
    dim : str or int, keyword parameter, optional, default='x'
        Specify on which dimension to apply the apodization method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    inv : bool, keyword parameter, optional, default=False
        True for inverse apodization.
    rev : bool, keyword parameter, optional, default=False
        True to reverse the apodization before applying it to the data.
    inplace : bool, keyword parameter, optional, default=False
        True if we make the transform inplace.  If False, the function return a new dataset.
    retapod : bool, keyword parameter, optional, default=False
        True to return the apodization array along with the apodized object.

    See Also
    --------
    triang : A triangular window that does not touch zero at the ends.
    """

    x = dataset

    return x * windows.bartlett(len(x), sym=True)


@_apodize_method()
def blackmanharris(dataset, **kwargs):
    """
    Calculate a minimum 4-term Blackman-Harris apodization.

    For multidimensional NDDataset,
    the apodization is by default performed on the last dimension.

    The data in the last dimension MUST be time-domain or dimensionless,
    otherwise an error is raised.

    Parameters
    ----------
    dataset : dataset
        Input dataset.

    Returns
    -------
    apodized
        dataset.
    apod_arr
        The apodization array only if 'retapod' is True.
    **kwargs
        Additional keyword parameters (see Other Parameters)

    Other Parameters
    ----------------
    dim : str or int, keyword parameter, optional, default='x'
        Specify on which dimension to apply the apodization method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    inv : bool, keyword parameter, optional, default=False
        True for inverse apodization.
    rev : bool, keyword parameter, optional, default=False
        True to reverse the apodization before applying it to the data.
    inplace : bool, keyword parameter, optional, default=False
        True if we make the transform inplace.  If False, the function return a new datase
    retapod : bool, keyword parameter, optional, default=False
        True to return the apodization array along with the apodized object.
    """

    x = dataset

    return x * windows.blackmanharris(len(x), sym=True)
