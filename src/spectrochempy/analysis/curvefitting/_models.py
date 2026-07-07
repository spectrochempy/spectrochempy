# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Module holding the definitions of all the various models."""

from functools import wraps

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import Quantity

__all__ = [
    "polynomial",
    "polynomialbaseline",
    "gaussian",
    "gaussianmodel",
    "lorentzian",
    "lorentzianmodel",
    "voigt",
    "voigtmodel",
    "asymmetricvoigt",
    "asymmetricvoigtmodel",
    "sigmoid",
    "sigmoidmodel",
]


def make_units_compatibility(func):
    """Take into account the input features (units, type...)."""

    def _convert_to_units(arg, x_units):
        if isinstance(arg, Quantity):
            arg.ito(x_units)  # eventually convert units and rescale
        # set units to those of x
        elif x_units is not None:
            arg = arg * x_units
        else:
            # do not take into account unit of arg
            return arg
        return arg.magnitude

    @wraps(func)
    def wrapper(cls, xinput, *args, **kwargs):
        returntype = "ndarray"
        x = xinput.copy()

        x_units = None
        if hasattr(xinput, "units"):
            x_units = xinput.units
            if isinstance(xinput, Coord):
                x = xinput.data
                returntype = "NDDataset"
            else:
                x = xinput.m

        # get args or their equivalent in kwargs and eventually convert units.
        newargs = []

        for index, param in enumerate(cls.args):
            newargs.append(kwargs.get(param, args[index] if len(args) > index else 0))

        for index, arg in enumerate(newargs):
            # adapt units
            if cls.args[index] in ["width", "pos"]:
                # implicit units: those of x else rescale
                newargs[index] = _convert_to_units(arg, x_units)

        ampl_units = None
        if hasattr(newargs[0], "units"):
            ampl_units = newargs[0].units
            newargs[0] = newargs[0].m

        extra_kwargs = {k: v for k, v in kwargs.items() if k not in cls.args}
        _data = func(cls, x, *newargs, **extra_kwargs)

        if returntype == "NDDataset":
            res = NDDataset(_data, units=ampl_units)
            res.x = Coord(xinput)
            res.name = cls.__class__.__name__.split("model")[0]
            res.title = "intensity"

        else:
            res = _data
            if ampl_units:
                res = res * ampl_units

        return res

    return wrapper


def _evaluate_model(model, x, name=None, **kwargs):
    """Evaluate a 1D model with ergonomic top-level helpers."""
    if isinstance(x, NDDataset):
        if x.ndim != 1:
            raise ValueError("Shape helpers expect a 1D abscissa")
        coord = getattr(x, x.dims[0], None)
        x = coord.copy() if coord is not None else Coord(x.data, units=x.units)

    result = model().f(x, **kwargs)
    if isinstance(result, NDDataset) and name is not None:
        result.name = name
    return result


def polynomial(x, offset=0.0, slope=0.0, ampl=1.0, **kwargs):
    """
    Return a 1D polynomial profile.

    This helper evaluates a centered polynomial using the same parameterization
    as the ``polynomialbaseline`` fitting shape, while also allowing an
    optional constant offset and linear slope for convenient signal
    construction.

    Parameters
    ----------
    x : array-like, Coord or NDDataset
        Abscissa values.
    offset : float, optional
        Constant offset added to the profile.
    slope : float, optional
        Linear slope applied around the centre of *x*.
    ampl : float, optional
        Global scaling factor applied to the polynomial terms.
    **kwargs
        Polynomial coefficients such as ``c_2=...``, ``c_3=...``, etc.
    """
    result = _evaluate_model(
        polynomialbaseline,
        x,
        name="polynomial",
        ampl=ampl,
        **kwargs,
    )

    if offset == 0.0 and slope == 0.0:
        return result

    xcoord = x
    if isinstance(xcoord, NDDataset):
        if xcoord.ndim != 1:
            raise ValueError("Shape helpers expect a 1D abscissa")
        coord = getattr(xcoord, xcoord.dims[0], None)
        xcoord = (
            coord.copy()
            if coord is not None
            else Coord(xcoord.data, units=xcoord.units)
        )

    xvalues = xcoord.data if isinstance(xcoord, Coord) else np.asarray(xcoord)

    xcentered = xvalues - xvalues[int(len(xvalues) / 2)]
    linear = offset + slope * xcentered
    return result + linear


def gaussian(x, ampl=1.0, pos=0.0, width=1.0, normalized=True, **kwargs):
    """
    Return a Gaussian profile.

    Parameters
    ----------
    x : array-like or Coord
        Abscissa values.
    ampl : float, optional
        Amplitude. When *normalized* is ``True`` (default) this scales the
        area under the curve. When *normalized* is ``False`` this is the peak
        height.
    pos : float, optional
        Position of the peak centre.
    width : float, optional
        Full width at half maximum (FWHM).
    normalized : bool, optional
        If ``True`` (default) the profile is normalized so that its integral
        is approximately *ampl*. If ``False`` the peak value is exactly
        *ampl*.
    **kwargs
        Additional keyword arguments forwarded to the model evaluator.
    """
    return _evaluate_model(
        gaussianmodel,
        x,
        name="gaussian",
        ampl=ampl,
        pos=pos,
        width=width,
        normalized=normalized,
        **kwargs,
    )


def lorentzian(x, ampl=1.0, pos=0.0, width=1.0, normalized=True, **kwargs):
    """
    Return a Lorentzian profile.

    Parameters
    ----------
    x : array-like or Coord
        Abscissa values.
    ampl : float, optional
        Amplitude. When *normalized* is ``True`` (default) this scales the
        area under the curve. When *normalized* is ``False`` this is the peak
        height.
    pos : float, optional
        Position of the peak centre.
    width : float, optional
        Full width at half maximum (FWHM).
    normalized : bool, optional
        If ``True`` (default) the profile is normalized so that its integral
        is approximately *ampl*. If ``False`` the peak value is exactly
        *ampl*.
    **kwargs
        Additional keyword arguments forwarded to the model evaluator.
    """
    return _evaluate_model(
        lorentzianmodel,
        x,
        name="lorentzian",
        ampl=ampl,
        pos=pos,
        width=width,
        normalized=normalized,
        **kwargs,
    )


def voigt(x, ampl=1.0, pos=0.0, width=1.0, ratio=0.5, normalized=True, **kwargs):
    """
    Return a Voigt profile.

    Parameters
    ----------
    x : array-like or Coord
        Abscissa values.
    ampl : float, optional
        Amplitude. When *normalized* is ``True`` (default) this scales the
        area under the curve. When *normalized* is ``False`` this is the peak
        height.
    pos : float, optional
        Position of the peak centre.
    width : float, optional
        Full width at half maximum (FWHM).
    ratio : float, optional
        Ratio of Gaussian to Lorentzian character (0 = pure Lorentzian,
        1 = pure Gaussian).
    normalized : bool, optional
        If ``True`` (default) the profile is normalized so that its integral
        is approximately *ampl*. If ``False`` the peak value is exactly
        *ampl*.
    **kwargs
        Additional keyword arguments forwarded to the model evaluator.
    """
    return _evaluate_model(
        voigtmodel,
        x,
        name="voigt",
        ampl=ampl,
        pos=pos,
        width=width,
        ratio=ratio,
        normalized=normalized,
        **kwargs,
    )


def asymmetricvoigt(
    x, ampl=1.0, pos=0.0, width=1.0, ratio=0.5, asym=0.0, normalized=True, **kwargs
):
    """
    Return an asymmetric Voigt profile.

    Parameters
    ----------
    x : array-like or Coord
        Abscissa values.
    ampl : float, optional
        Amplitude. When *normalized* is ``True`` (default) this scales the
        area under the curve. When *normalized* is ``False`` this is the peak
        height.
    pos : float, optional
        Position of the peak centre.
    width : float, optional
        Full width at half maximum (FWHM).
    ratio : float, optional
        Ratio of Gaussian to Lorentzian character.
    asym : float, optional
        Asymmetry parameter.
    normalized : bool, optional
        If ``True`` (default) the profile is normalized so that its integral
        is approximately *ampl*. If ``False`` the peak value is exactly
        *ampl*.
    **kwargs
        Additional keyword arguments forwarded to the model evaluator.
    """
    return _evaluate_model(
        asymmetricvoigtmodel,
        x,
        name="asymmetricvoigt",
        ampl=ampl,
        pos=pos,
        width=width,
        ratio=ratio,
        asym=asym,
        normalized=normalized,
        **kwargs,
    )


def sigmoid(x, ampl=1.0, pos=0.0, asym=1.0, **kwargs):
    """Return a sigmoid profile."""
    return _evaluate_model(
        sigmoidmodel, x, name="sigmoid", ampl=ampl, pos=pos, asym=asym, **kwargs
    )


############
#          #
#    1D    #
#          #
############


# ======================================================================================
# PolynomialBaseline
# ======================================================================================
class polynomialbaseline:
    r"""
    Arbitrary-degree polynomial (degree limited to 10, however).

    As a linear baseline is automatically calculated, this polynom is always of
    greater or equal to order 2 (parabolic function).

    .. math::
        f(x) = ampl * \sum_{i=2}^{max} c_i*x^i
    """

    type = "1D"
    args = ["ampl"]
    args.extend([f"c_{i}" for i in range(2, 11)])

    script = """
    MODEL: baseline%(id)d\nshape: polynomialbaseline
    # This polynom starts at the order 2
    # as a linear baseline is additionally fitted automatically
    # parameters must be in the form c_i where i is an integer as shown below
    $ ampl: %(scale).3g, 0.0, None
    $ c_2: 1.0, None, None
    * c_3: 0.0, None, None
    * c_4: 0.0, None, None
    # etc...
    """

    @make_units_compatibility
    def f(self, x, ampl, *c_, **kwargs):
        c = [0.0, 0.0]
        c.extend(c_)
        return ampl * np.polyval(np.array(tuple(c))[::-1], x - x[int(x.size / 2)])


# #===============================================================================
# # Gaussian2DModel
# #===============================================================================
# class gaussian2dmodel(object):
#    r"""
#    Two dimensional Gaussian model (*not* normalized - peak value is 1).
#
#    .. math::
#        A e^{\frac{-(x-\iso_x)^2}{2 \gb_x^2}} e^{\frac{-(y-\iso_y)^2}{2 \gb_y^2}}
#
#    """
#    args = ['amp','gbx','gby','posx','posy']
#    def f(self, xy, amp, gbx, gby, posx, posy, **kargs):
#        gbx = float(gbx)
#        gby = float(gby)
#        x,y = xy
#        xo = x-posx
#        xdenom = 2*gbx*gbx
#        yo = y-posy
#        ydenom = 2*gby*gby
#        return amp*np.exp(-xo*xo/xdenom-yo*yo/ydenom)
# ======================================================================================
# ======================================================================================
# GaussianModel
# ======================================================================================
class gaussianmodel:
    r"""
    1D Gaussian function.

    When *normalized* is ``True`` (default) the function returns a normalized
    profile whose integral is approximately *ampl*:

    .. math::
        f(x) = \frac{ampl}{\sqrt{2 \pi \sigma^2} } \exp({\frac{-(x-pos)^2}{2 \sigma^2}})

    where :math:`\sigma = \frac{width}{2.3548}` .

    When *normalized* is ``False`` the peak height is exactly *ampl* and the
    normalization factor is omitted.
    """

    type = "1D"
    args = ["ampl", "pos", "width"]

    script = """
    MODEL: line%(id)d\nshape: gaussianmodel
    $ ampl: %(ampl).3f, 0.0, None
    $ width: %(width).3f, 0.0, None
    $ pos: %(pos).3f, %(poslb).3f, %(poshb).3f
    """

    @make_units_compatibility
    def f(self, x, ampl, pos, width, **kwargs):
        gb = width / 2.3548
        tsq = (x - pos) * 2**-0.5 / gb
        normalized = kwargs.get("normalized", True)
        if normalized:
            w = np.exp(-tsq * tsq) * (2 * np.pi) ** -0.5 / gb
            w = w * abs(x[1] - x[0])
        else:
            w = np.exp(-tsq * tsq)
        return ampl * w


# ======================================================================================
# LorentzianModel
# ======================================================================================
class lorentzianmodel:
    r"""
    A standard Lorentzian function (also known as the Cauchy distribution).

    When *normalized* is ``True`` (default) the function returns a normalized
    profile whose integral is approximately *ampl*:

    .. math::
        f(x) = \frac{ampl * \lambda}{\pi [(x-pos)^2+ \lambda^2]}

    where :math:`\lambda = \frac{width}{2}` .

    When *normalized* is ``False`` the peak height is exactly *ampl* and the
    normalization factor is omitted.
    """

    type = "1D"
    args = ["ampl", "pos", "width"]

    script = """
    MODEL: line%(id)d\nshape: lorentzianmodel
    $ ampl: %(ampl).3f, 0.0, None
    $ width: %(width).3f, 0.0, None
    $ pos: %(pos).3f, %(poslb).3f, %(poshb).3f
    """

    @make_units_compatibility
    def f(self, x, ampl, pos, width, **kargs):
        lb = width / 2.0
        w = lb / np.pi / (x * x - 2 * x * pos + pos * pos + lb * lb)
        normalized = kargs.get("normalized", True)
        w = w * abs(x[1] - x[0]) if normalized else w * np.pi * lb
        return ampl * w


# ======================================================================================
# VoigtModel
# ======================================================================================
class voigtmodel:
    """
    A Voigt model constructed as the convolution of a :class:`GaussianModel` and a :class:`LorentzianModel` .

    Commonly used for spectral line fitting.
    """

    type = "1D"
    args = ["ampl", "pos", "width", "ratio"]

    script = """
    MODEL: line%(id)d\nshape: voigtmodel
    $ ampl: %(ampl).3f, 0.0, None
    $ width: %(width).3f, 0.0, None
    $ pos: %(pos).3f, %(poslb).3f, %(poshb).3f
    $ ratio: 0.1, 0.0, 1.0
    """

    # @make_units_compatibility
    # def f(self, x, ampl, pos, width, ratio, **kargs):
    #     from scipy.special import wofz
    #
    #     gb = ratio * width / 2.3548
    #     lb = (1.0 - ratio) * width / 2.0
    #     if ratio < 1.0e-16:
    #         return lorentzianmodel().f(x, ampl, pos, lb * 2.0, **kargs)
    #     else:
    #         w = wofz(((x - pos) + 1.0j * lb) * 2 ** -0.5 / gb)
    #         w = w.real * (2.0 * np.pi) ** -0.5 / gb
    #         w = w * abs(x[1] - x[0])
    #         return ampl * w

    @staticmethod
    def f(x, ampl, pos, width, ratio, **kargs):
        return asymmetricvoigtmodel().f(x, ampl, pos, width, ratio, asym=0.0, **kargs)


# ======================================================================================
# Asymmetric Voigt Model
# ======================================================================================
class asymmetricvoigtmodel:
    """
    An asymmetric Voigt model.

    A. L. Stancik and E. B. Brauns, Vibrational Spectroscopy, 2008, 47, 66-69.

    When *normalized* is ``True`` (default) the function returns a normalized
    profile whose integral is approximately *ampl*. When *normalized* is
    ``False`` the peak height is exactly *ampl* and the normalization factor
    is omitted.
    """

    type = "1D"
    args = ["ampl", "pos", "width", "ratio", "asym"]

    script = """
        MODEL: line%(id)d\nshape: voigtmodel
        $ ampl: %(ampl).3f, 0.0, None
        $ width: %(width).3f, 0.0, None
        $ pos: %(pos).3f, %(poslb).3f, %(poshb).3f
        $ ratio: 0.1, 0.0, 1.0
        $ asym: 0.1, 0.0, 1.0
        """

    @make_units_compatibility
    def f(self, x, ampl, pos, width, ratio, asym, **kargs):
        from scipy.special import wofz

        g = width
        if asym > 0.0:
            # sigmoid variation of the width
            g = 2.0 * sigmoidmodel().f(x, width, pos, asym)
        gb = ratio * g / 2.3548
        lb = (1.0 - ratio) * g / 2.0
        if ratio < 1.0e-16:
            return lorentzianmodel().f(x, ampl, pos, lb * 2.0, **kargs)
        w = wofz(((x - pos) + 1.0j * lb) * 2**-0.5 / gb)
        w = w.real * (2.0 * np.pi) ** -0.5 / gb
        normalized = kargs.get("normalized", True)
        if normalized:
            w = w * abs(x[1] - x[0])
        else:
            peak = np.max(w)
            if peak > 0:
                w = w / peak
        return ampl * w


# ======================================================================================
# Sigmoid Model
# ======================================================================================


class sigmoidmodel:
    r"""
    A Sigmoid function.

    .. math::
        f(x) = \frac{1.}{1 + \exp(\lambda (x-pos))}
    """

    type = "1D"
    args = ["ampl", "pos", "asym"]

    script = """
    MODEL: line%(id)d\nshape: sigmoidmodel
    $ ampl: %(ampl).3f, 0.0, None
    $ pos: %(pos).3f, %(poslb).3f, %(poshb).3f
    $ asym: %(asym).3f, 0.0, None
    """

    @make_units_compatibility
    def f(self, x, ampl, pos, asym, **kargs):
        w = 1.0 / (1.0 + np.exp(asym * (x - pos) / ampl))
        return ampl * w


# ======================================================================================
# User defined model
# ======================================================================================
class usermodel:
    """Base class for user defined models."""

    type = "1D"
    args = []

    @staticmethod
    def f():
        raise NotImplementedError("This is a base class for user defined models")
