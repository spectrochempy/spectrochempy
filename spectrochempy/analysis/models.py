# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
This module holds the definitions all the various models.
"""
from functools import wraps
import numpy as np

from spectrochempy.core.units import Quantity
from spectrochempy.core.dataset.coord import Coord, LinearCoord
from spectrochempy.core.dataset.nddataset import NDDataset

__all__ = [
    "polynomialbaseline",
    "gaussianmodel",
    "lorentzianmodel",
    "voigtmodel",
    "asymmetricvoigtmodel",
    "sigmoidmodel",
]


def make_units_compatibility(func):
    """
    Decorator to take into account the input features (units, type...)
    """

    def _convert_to_units(arg, x_units):

        if isinstance(arg, Quantity):
            arg.ito(x_units)  # eventually convert units and rescale
        # set units to those of x
        else:
            if x_units is not None:
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

        print(newargs)
        _data = func(cls, x, *newargs)

        if returntype == "NDDataset":
            res = NDDataset(_data, units=ampl_units)
            res.x = LinearCoord(xinput)
            res.name = cls.__class__.__name__.split("model")[0]
            res.title = "intensity"

        else:
            res = _data
            if ampl_units:
                res = res * ampl_units

        return res

    return wrapper


############
#          #
#    1D    #
#          #
############

# ======================================================================================================================
# PolynomialBaseline
# ======================================================================================================================


class polynomialbaseline(object):
    """
    Arbitrary-degree polynomial (degree limited to 10, however).

    As a linear baseline is automatically calculated, this polynom is always of
    greater or equal to order 2 (parabolic function).

    .. math::
        f(x) = ampl * \\sum_{i=2}^{max} c_i*x^i
    """

    type = "1D"
    args = ["ampl"]
    args.extend(["c_%d" % i for i in range(2, 11)])

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
    def f(self, x, ampl, *c_, **kargs):
        c = [0.0, 0.0]
        c.extend(c_)
        return ampl * np.polyval(np.array(tuple(c))[::-1], x - x[int(x.size / 2)])


# ======================================================================================================================
# #===============================================================================
# # Gaussian2DModel
# #===============================================================================
# class gaussian2dmodel(object):
#    """
#    Two dimensional Gaussian model (*not* normalized - peak value is 1).
#
#    .. math::
#        A e^{\\frac{-(x-\\iso_x)^2}{2 \\gb_x^2}} e^{\\frac{-(y-\\iso_y)^2}{2 \\gb_y^2}}
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
# ======================================================================================================================


# ======================================================================================================================
# GaussianModel
# ======================================================================================================================
class gaussianmodel(object):
    """
    Normalized 1D gaussian function.

    .. math::
        f(x) = \\frac{ampl}{\\sqrt{2 \\pi \\sigma^2} } \\exp({\\frac{-(x-pos)^2}{2 \\sigma^2}})

    where :math:`\\sigma = \\frac{width}{2.3548}`.
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
    def f(self, x, ampl, pos, width, **kargs):
        gb = width / 2.3548
        tsq = (x - pos) * 2 ** -0.5 / gb
        w = np.exp(-tsq * tsq) * (2 * np.pi) ** -0.5 / gb
        w = w * abs(x[1] - x[0])
        return ampl * w


# ======================================================================================================================
# LorentzianModel
# ======================================================================================================================
class lorentzianmodel(object):
    """
    A standard Lorentzian function (also known as the Cauchy distribution).

    .. math::
        f(x) = \\frac{ampl * \\lambda}{\\pi [(x-pos)^2+ \\lambda^2]}

    where :math:`\\lambda = \\frac{width}{2}`.
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
        w = w * abs(x[1] - x[0])
        return ampl * w


# ======================================================================================================================
# VoigtModel
# ======================================================================================================================
class voigtmodel(object):
    """
    A Voigt model constructed as the convolution of a :class:`GaussianModel` and
    a :class:`LorentzianModel`.

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

    def f(self, x, ampl, pos, width, ratio, **kargs):

        return asymmetricvoigtmodel().f(x, ampl, pos, width, ratio, asym=0.0)


# ======================================================================================================================
# Asymmetric Voigt Model
# ======================================================================================================================


class asymmetricvoigtmodel(object):
    """
    An asymmetric Voigt model.

    A. L. Stancik and E. B. Brauns, Vibrational Spectroscopy, 2008, 47, 66-69.
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
        else:
            w = wofz(((x - pos) + 1.0j * lb) * 2 ** -0.5 / gb)
            w = w.real * (2.0 * np.pi) ** -0.5 / gb
            w = w * abs(x[1] - x[0])
            return ampl * w


class sigmoidmodel(object):
    """
    A Sigmoid function.

    .. math::
        f(x) = \\frac{1.}{1 + \\exp(\\lambda (x-pos))}
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


if __name__ == "__main__":
    pass
