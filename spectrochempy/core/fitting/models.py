# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
This module holds the definitions all the various models.
"""

__all__ = []

import numpy as np


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

    args = ["ampl"]
    args.extend(["c_%d" % i for i in range(2, 11)])

    script = """
    MODEL: baseline%(id)d\nshape: polynomialbaseline
    # This polynom starts at the order 2
    # as a linear baseline is additionnaly fitted automatically
    # parameters must be in the form c_i where i is an integer as shown below
    $ ampl: %(scale).3g, 0.0, None
    $ c_2: 1.0, None, None
    * c_3: 0.0, None, None
    * c_4: 0.0, None, None
    # etc...
    """

    def f(self, x, ampl, *c_, **kargs):
        c = [0.0, 0.0]
        c.extend(c_)
        return ampl * np.polyval(np.array(tuple(c))[::-1], x - x[x.size / 2])


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

    args = ["ampl", "width", "pos"]
    script = """
    MODEL: line%(id)d\nshape: gaussianmodel
    $ ampl: %(ampl).3f, 0.0, None
    $ width: %(width).3f, 0.0, None
    $ pos: %(pos).3f, %(poslb).3f, %(poshb).3f
    """

    def f(self, x, ampl, width, pos, **kargs):
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

    args = ["ampl", "width", "pos"]
    script = """
    MODEL: line%(id)d\nshape: lorentzianmodel
    $ ampl: %(ampl).3f, 0.0, None
    $ width: %(width).3f, 0.0, None
    $ pos: %(pos).3f, %(poslb).3f, %(poshb).3f
    """

    def f(self, x, ampl, width, pos, **kargs):
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

    args = ["ampl", "width", "ratio", "pos"]
    script = """
    MODEL: line%(id)d\nshape: voigtmodel
    $ ampl: %(ampl).3f, 0.0, None
    $ width: %(width).3f, 0.0, None
    $ pos: %(pos).3f, %(poslb).3f, %(poshb).3f
    $ ratio: 0.1, 0.0, 1.0
    """

    def f(self, x, ampl, width, ratio, pos, **kargs):
        from scipy.special import wofz

        gb = ratio * width / 2.3548
        lb = (1.0 - ratio) * width / 2.0
        if gb < 1.0e-16:
            return lorentzianmodel().f(x, ampl, lb * 2.0, pos, **kargs)
        else:
            w = wofz(((x - pos) + 1.0j * lb) * 2 ** -0.5 / gb)
            w = w.real * (2.0 * np.pi) ** -0.5 / gb
            w = w * abs(x[1] - x[0])
            return ampl * w


# ======================================================================================================================
# Assymetric Voigt Model
# ======================================================================================================================
class assymvoigtmodel(object):
    """
    An assymetric Voigt model.

    A. L. Stancik and E. B. Brauns, Vibrational Spectroscopy, 2008, 47, 66-69.
    """

    args = ["ampl", "width", "ratio", "assym", "pos"]

    script = """
        MODEL: line%(id)d\nshape: voigtmodel
        $ ampl: %(ampl).3f, 0.0, None
        $ width: %(width).3f, 0.0, None
        $ pos: %(pos).3f, %(poslb).3f, %(poshb).3f
        $ ratio: 0.1, 0.0, 1.0
        $ assym: 0.1, 0.0, 1.0
        """

    def lorentz(self, x, g_, pos):
        w = (2.0 / (np.pi * g_)) / (1.0 + 4.0 * ((x - pos) / g_) ** 2)
        return w

    def gaussian(self, x, g_, pos):
        a = np.sqrt(4.0 * np.log(2.0) / np.pi) / g_
        b = -4.0 * np.log(2.0) * ((x - pos) / g_) ** 2
        w = a * np.exp(b)
        return w

    def g(self, x, width, a):
        return

    def f(self, x, ampl, width, ratio, assym, pos, **kargs):
        g = 2.0 * width / (1.0 + np.exp(assym * (x - pos)))
        # sigmoid variation of width
        w = ratio * self.lorentz(x, g, pos) + (1.0 - ratio) * self.gaussian(x, g, pos)
        w = w * abs(x[1] - x[0])
        return ampl * w


#################
#               #
#    GENERAL    #
#               #
#################

# ======================================================================================================================
# getmodel
# ======================================================================================================================
def getmodel(x, y=None, modelname=None, par=None, **kargs):
    """
    Get the model for a given x vector.

    Parameters
    -----------
    x : ndarray
        Array of frequency where to evaluate the model values returned by the
        f function.
    y : ndarray or None
        None for 1D, or index for the second dimension.
    modelname : str
        Name of the model class to use.
    par : :class:`Parameters` instance
        Parameter to pass to the f function.
    kargs : any
        Keywords arguments to pass the the f function.

    Returns
    -------
    ndarray : float
        An array containing the calculated model.
    """
    model = par.model[modelname]
    modelcls = globals()[model]

    # take an instance of the model
    a = modelcls()

    # get the parameters for the given model
    args = []
    for p in a.args:
        try:
            args.append(par["%s_%s" % (p, modelname)])
        except KeyError as e:
            if p.startswith("c_"):
                # probably the end of the list
                # due to a limited polynomial degree
                pass
            else:
                raise ValueError(e.message)

    x = np.array(x, dtype=np.float64)
    if y is not None:
        y = np.array(y, dtype=np.float64)

    if y is None:
        return a.f(x, *args, **kargs)
    else:
        return a.f(x, y, *args, **kargs)


if __name__ == "__main__":
    pass
