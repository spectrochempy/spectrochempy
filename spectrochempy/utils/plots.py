# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import matplotlib as mpl
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import numpy as np

from spectrochempy.core.dataset.meta import Meta

__all__ = [
    "cmyk2rgb",
    "NBlack",
    "NRed",
    "NBlue",
    "NGreen",
    "figure",
    "show",
    "get_figure",  # Plotly specific
    "get_plotly_figure",
    "colorscale",
    "make_attr",
    "make_label",
]


# ............................................................................
# color conversion function
def cmyk2rgb(C, M, Y, K):
    """
    CMYK to RGB conversion.

    C,M,Y,K are given in percent.
    The R,G,B values are returned in the range of 0..1.
    """
    C, Y, M, K = C / 100.0, Y / 100.0, M / 100.0, K / 100.0

    # The red (R) color is calculated from the cyan (C) and black (K) colors:
    R = (1.0 - C) * (1.0 - K)

    # The green color (G) is calculated from the magenta (M) and black (K) colors:
    G = (1.0 - M) * (1.0 - K)

    # The blue color (B) is calculated from the yellow (Y) and black (K) colors:
    B = (1.0 - Y) * (1.0 - K)

    return (R, G, B)


# Constants
# ------------------------------------------------------------------

# For color blind people, it is safe to use only 4 colors in graphs:
# see http://jfly.iam.u-tokyo.ac.jp/color/ichihara_etal_2008.pdf
#   Black CMYK=0,0,0,0
#   Red CMYK= 0, 77, 100, 0 %
#   Blue CMYK= 100, 30, 0, 0 %
#   Green CMYK= 85, 0, 60, 10 %
NBlack = (0, 0, 0)
NRed = cmyk2rgb(0, 77, 100, 0)
NBlue = cmyk2rgb(100, 30, 0, 0)
NGreen = cmyk2rgb(85, 0, 60, 10)


# .............................................................................
def figure(preferences=Meta(), **kwargs):
    """
    Method to open a new figure.

    Parameters
    ----------
    Kwargs : any
        Keywords arguments to be passed to the matplotlib figure constructor.
    Preferences : Meta dictionary
        Per object saved plot configuration.
    """
    return get_figure(preferences=preferences, **kwargs)


# .............................................................................
def show():
    """
    Method to force the `matplotlib` figure display.
    """
    from spectrochempy import NO_DISPLAY

    if not NO_DISPLAY:

        if get_figure(clear=False):
            plt.show(block=True)


# .............................................................................
def get_figure(**kwargs):
    """
    Get the figure where to plot.

    Parameters
    ----------
    clear : bool
        If False the last used figure is returned.
    figsize : 2-tuple of floats, default: rcParams["figure.figsize"] (default: [6.4, 4.8])
        Figure dimension (width, height) in inches.
    dpi : float, default: rcParams["figure.dpi"] (default: 100.0)
        Dots per inch.
    facecolor : default: rcParams["figure.facecolor"] (default: 'white')
        The figure patch facecolor.
    edgecolor : default: preferences.figure_edgecolor (default: 'white')
        The figure patch edge color.
    frameon : bool, default: preferences.figure_frameon (default: True)
        If False, suppress drawing the figure background patch.
    tight_layout : bool or dict, default: preferences.figure.autolayout (default: False)
        If False use subplotpars. If True adjust subplot parameters using tight_layout with default padding.
        When providing a dict containing the keys pad, w_pad, h_pad, and rect,
        the default tight_layout paddings will be overridden.
    constrained_layout : bool, default: preferences.figure_constrained_layout (default: False)
        If True use constrained layout to adjust positioning of plot elements.
        Like tight_layout, but designed to be more flexible. See Constrained Layout Guide for examples.
    preferences : Meta object,
        Per object plot configuration.

    Returns
    -------
    matplotlib figure instance
    """

    n = plt.get_fignums()

    clear = kwargs.get("clear", True)

    if not n or clear:
        # create a figure
        prefs = kwargs.pop("preferences", None)

        figsize = kwargs.get("figsize", prefs.figure_figsize)
        dpi = kwargs.get("dpi", prefs.figure_dpi)
        facecolor = kwargs.get("facecolor", prefs.figure_facecolor)
        edgecolor = kwargs.get("edgecolor", prefs.figure_edgecolor)
        frameon = kwargs.get("frameon", prefs.figure_frameon)
        tight_layout = kwargs.get("autolayout", prefs.figure_autolayout)

        # get the current figure (or the last used)
        fig = plt.figure(figsize=figsize)

        fig.set_dpi(dpi)
        fig.set_frameon(frameon)
        try:
            fig.set_edgecolor(edgecolor)
        except ValueError:
            fig.set_edgecolor(eval(edgecolor))
        try:
            fig.set_facecolor(facecolor)
        except ValueError:
            try:
                fig.set_facecolor(eval(facecolor))
            except ValueError:
                fig.set_facecolor("#" + eval(facecolor))
        fig.set_dpi(dpi)
        fig.set_tight_layout(tight_layout)

        return fig

    # a figure already exists - if several we take the last
    return plt.figure(n[-1])


# FOR PLOTLY
# .............................................................................
def get_plotly_figure(clear=True, fig=None, **kwargs):
    """
    Get the figure where to plot.

    Parameters
    ----------
    clear : bool
        If False the figure provided in the `fig` parameters is used.
    fig : plotly figure
        If provided, and clear is not True, it will be used for plotting
    kwargs : any
        Keywords arguments to be passed to the plotly figure constructor.

    Returns
    -------
    Plotly figure instance
    """

    if clear or fig is None:
        # create a figure
        return go.Figure()

    # a figure already exists - if several we take the last
    return fig


class colorscale:
    def normalize(self, vmin, vmax, cmap="viridis", rev=False, offset=0):
        """ """
        if rev:
            cmap = cmap + "_r"
        _colormap = plt.get_cmap(cmap)

        _norm = mpl.colors.Normalize(vmin=vmin - offset, vmax=vmax - offset)
        self.scalarMap = mpl.cm.ScalarMappable(norm=_norm, cmap=_colormap)

    def rgba(self, z, offset=0):
        c = np.array(self.scalarMap.to_rgba(z.squeeze() - offset))
        c[0:3] *= 255
        c[0:3] = np.round(c[0:3].astype("uint16"), 0)
        return f"rgba{tuple(c)}"


colorscale = colorscale()


# ............................................................................
def make_label(ss, lab="<no_axe_label>", use_mpl=True):
    """
    Make a label from title and units.
    """

    if ss is None:
        return lab

    if ss.title:
        label = ss.title  # .replace(' ', r'\ ')
    else:
        label = lab

    if "<untitled>" in label:
        label = "values"

    if use_mpl:
        if ss.units is not None and str(ss.units) not in [
            "dimensionless",
            "absolute_transmittance",
        ]:
            units = r"/\ {:~L}".format(ss.units)
            units = units.replace("%", r"\%")
        else:
            units = ""
        label = r"%s $\mathrm{%s}$" % (label, units)
    else:
        if ss.units is not None and str(ss.units) != "dimensionless":
            units = r"{:~H}".format(ss.units)
        else:
            units = ""
        label = r"%s / %s" % (label, units)

    return label


def make_attr(key):
    name = "M_%s" % key[1]
    k = r"$\mathrm{%s}$" % name

    if "P" in name:
        m = "o"
        c = NBlack
    elif "A" in name:
        m = "^"
        c = NBlue
    elif "B" in name:
        m = "s"
        c = NRed

    if "400" in key:
        f = "w"
        s = ":"
    else:
        f = c
        s = "-"

    return m, c, k, f, s
