# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
import textwrap

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.axes as maxes
import mpl_toolkits.mplot3d.axes3d as maxes3D

import plotly.graph_objects as go
import numpy as np

from spectrochempy.core.dataset.meta import Meta
from spectrochempy.units import remove_args_units

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
    "plot_method",
]


@maxes.subplot_class_factory
class _Axes(maxes.Axes):
    """
    Subclass of matplotlib Axes class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def draw(self, renderer):
        #    # with plt.rc_context({"something": self.xxx}):
        return super().draw(renderer)

    @remove_args_units
    def plot(self, *args, **kwargs):
        return super().plot(*args, **kwargs)

    @remove_args_units
    def errorbar(self, *args, **kwargs):
        return super().errorbar(*args, **kwargs)

    @remove_args_units
    def scatter(self, *args, **kwargs):
        return super().scatter(*args, **kwargs)

    @remove_args_units
    def plot_date(self, *args, **kwargs):
        return super().plot_date(*args, **kwargs)

    @remove_args_units
    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)

    @remove_args_units
    def loglog(self, *args, **kwargs):
        return super().loglog(*args, **kwargs)

    @remove_args_units
    def semilogx(self, *args, **kwargs):
        return super().semilogx(*args, **kwargs)

    @remove_args_units
    def semilogy(self, *args, **kwargs):
        return super().semilogy(*args, **kwargs)

    @remove_args_units
    def fill_between(self, *args, **kwargs):
        return super().fill_between(*args, **kwargs)

    @remove_args_units
    def fill_betweenx(self, *args, **kwargs):
        return super().fill_betweenx(*args, **kwargs)

    @remove_args_units
    def bar(self, *args, **kwargs):
        return super().bar(*args, **kwargs)

    @remove_args_units
    def barh(self, *args, **kwargs):
        return super().barh(*args, **kwargs)

    @remove_args_units
    def bar_label(self, *args, **kwargs):
        return super().bar_label(*args, **kwargs)

    @remove_args_units
    def stem(self, *args, **kwargs):
        return super().stem(*args, **kwargs)

    @remove_args_units
    def eventplot(self, *args, **kwargs):
        return super().eventplot(*args, **kwargs)

    @remove_args_units
    def pie(self, *args, **kwargs):
        return super().pie(*args, **kwargs)

    @remove_args_units
    def stackplot(self, *args, **kwargs):
        return super().stackplot(*args, **kwargs)

    @remove_args_units
    def broken_barh(self, *args, **kwargs):
        return super().broken_barh(*args, **kwargs)

    @remove_args_units
    def vlines(self, *args, **kwargs):
        return super().vlines(*args, **kwargs)

    @remove_args_units
    def hlines(self, *args, **kwargs):
        return super().hlines(*args, **kwargs)

    @remove_args_units
    def fill(self, *args, **kwargs):
        return super().fill(*args, **kwargs)

    @remove_args_units
    def axhline(self, *args, **kwargs):
        return super().axhline(*args, **kwargs)

    @remove_args_units
    def axhspan(self, *args, **kwargs):
        return super().axhspan(*args, **kwargs)

    @remove_args_units
    def axvline(self, *args, **kwargs):
        return super().axvline(*args, **kwargs)

    @remove_args_units
    def axvspan(self, *args, **kwargs):
        return super().axvspan(*args, **kwargs)

    @remove_args_units
    def axline(self, *args, **kwargs):
        return super().axline(*args, **kwargs)

    @remove_args_units
    def acorr(self, *args, **kwargs):
        return super().acorr(*args, **kwargs)

    @remove_args_units
    def angle_spectrum(self, *args, **kwargs):
        return super().angle_spectrum(*args, **kwargs)

    @remove_args_units
    def cohere(self, *args, **kwargs):
        return super().cohere(*args, **kwargs)

    @remove_args_units
    def csd(self, *args, **kwargs):
        return super().csd(*args, **kwargs)

    @remove_args_units
    def magnitude_spectrum(self, *args, **kwargs):
        return super().magnitude_spectrum(*args, **kwargs)

    @remove_args_units
    def phase_spectrum(self, *args, **kwargs):
        return super().phase_spectrum(*args, **kwargs)

    @remove_args_units
    def psd(self, *args, **kwargs):
        return super().psd(*args, **kwargs)

    @remove_args_units
    def specgram(self, *args, **kwargs):
        return super().specgram(*args, **kwargs)

    @remove_args_units
    def xcorr(self, *args, **kwargs):
        return super().xcorr(*args, **kwargs)

    @remove_args_units
    def boxplot(self, *args, **kwargs):
        return super().boxplot(*args, **kwargs)

    @remove_args_units
    def violinplot(self, *args, **kwargs):
        return super().violinplot(*args, **kwargs)

    @remove_args_units
    def violin(self, *args, **kwargs):
        return super().violin(*args, **kwargs)

    @remove_args_units
    def bxp(self, *args, **kwargs):
        return super().bxp(*args, **kwargs)

    @remove_args_units
    def hexbin(self, *args, **kwargs):
        return super().hexbin(*args, **kwargs)

    @remove_args_units
    def hist(self, *args, **kwargs):
        return super().hist(*args, **kwargs)

    @remove_args_units
    def hist2d(self, *args, **kwargs):
        return super().hist2d(*args, **kwargs)

    @remove_args_units
    def stairs(self, *args, **kwargs):
        return super().stairs(*args, **kwargs)

    @remove_args_units
    def contour(self, *args, **kwargs):
        return super().contour(*args, **kwargs)

    @remove_args_units
    def contourf(self, *args, **kwargs):
        return super().contourf(*args, **kwargs)

    @remove_args_units
    def imshow(self, *args, **kwargs):
        return super().imshow(*args, **kwargs)

    @remove_args_units
    def matshow(self, *args, **kwargs):
        return super().matshow(*args, **kwargs)

    @remove_args_units
    def pcolor(self, *args, **kwargs):
        return super().pcolor(*args, **kwargs)

    @remove_args_units
    def pcolorfast(self, *args, **kwargs):
        return super().pcolorfast(*args, **kwargs)

    @remove_args_units
    def pcolormesh(self, *args, **kwargs):
        return super().pcolormesh(*args, **kwargs)

    @remove_args_units
    def spy(self, *args, **kwargs):
        return super().spy(*args, **kwargs)

    @remove_args_units
    def tripcolor(self, *args, **kwargs):
        return super().tripcolor(*args, **kwargs)

    @remove_args_units
    def triplot(self, *args, **kwargs):
        return super().triplot(*args, **kwargs)

    @remove_args_units
    def tricontour(self, *args, **kwargs):
        return super().tricontour(*args, **kwargs)

    @remove_args_units
    def tricontourf(self, *args, **kwargs):
        return super().tricontourf(*args, **kwargs)

    @remove_args_units
    def annotate(self, *args, **kwargs):
        return super().annotate(*args, **kwargs)

    @remove_args_units
    def text(self, *args, **kwargs):
        return super().text(*args, **kwargs)

    @remove_args_units
    def table(self, *args, **kwargs):
        return super().table(*args, **kwargs)

    @remove_args_units
    def arrow(self, *args, **kwargs):
        return super().arrow(*args, **kwargs)

    @remove_args_units
    def set_xlim(self, *args, **kwargs):
        return super().set_xlim(*args, **kwargs)

    @remove_args_units
    def set_ylim(self, *args, **kwargs):
        return super().set_ylim(*args, **kwargs)


class _Axes3D(maxes3D.Axes3D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @remove_args_units
    def plot_surface(self, *args, **kwargs):
        return super().plot_surface(*args, **kwargs)


def plot_method(type, doc):
    """
    Decorator to to select a plot method from the function name
    """

    def decorator_plot_method(func):

        method = func.__name__.split("plot_")[-1]

        def wrapper(dataset, *args, **kwargs):

            if dataset.ndim < 2:
                from spectrochempy.core.plotters.plot1d import plot_1D

                _ = kwargs.pop("method", None)
                return plot_1D(dataset, *args, method=method, **kwargs)

            if kwargs.get("use_plotly", False):
                return dataset.plotly(method=method, **kwargs)
            else:
                return getattr(dataset, f"plot_{type}")(*args, method=method, **kwargs)

        wrapper.__doc__ = f"""
{textwrap.dedent(func.__doc__).strip()}

Parameters
----------
dataset : |NDDataset|
    The dataset to plot.
**kwargs : dic, optional
    Additional keywords parameters.
    See Other Parameters.

Other Parameters
----------------
{doc.strip()}

See Also
--------
plot_1D
plot_pen
plot_bar
plot_scatter_pen
plot_multiple
plot_2D
plot_stack
plot_map
plot_image
plot_3D
plot_surface
plot_waterfall
multiplot
""".replace(
            f"\nplot_{method}", ""
        )

        return wrapper

    return decorator_plot_method


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

    if NO_DISPLAY:
        plt.close("all")
    else:
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
