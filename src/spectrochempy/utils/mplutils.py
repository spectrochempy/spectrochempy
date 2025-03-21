# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import matplotlib.axes as maxes
import mpl_toolkits.mplot3d.axes3d as maxes3D  # noqa: N812
from matplotlib import pyplot as plt

__all__ = ["show"]


@maxes.subplot_class_factory
class _Axes(maxes.Axes):  # pragma: no cover
    """Subclass of matplotlib Axes class."""

    from spectrochempy.core.units import remove_args_units

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def draw(self, renderer):
    #    #    # with plt.rc_context({"something": self.xxx}):
    #    return super().draw(renderer)

    def _implements(self, type=None):
        if type is None:
            return "_Axes"
        return type == "_Axes"

    def __repr__(self):
        return "<Matplotlib Axes object>"

    def __str__(self):
        return self.__repr__()

    def _repr_html_(self):
        # Suppress text output in notebooks
        return ""

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
        """Plot a spy."""
        return super().spy(*args, **kwargs)

    @remove_args_units
    def tripcolor(self, *args, **kwargs):
        """Plot a tripcolor."""
        return super().tripcolor(*args, **kwargs)

    @remove_args_units
    def triplot(self, *args, **kwargs):
        """Plot a triplot."""
        return super().triplot(*args, **kwargs)

    @remove_args_units
    def tricontour(self, *args, **kwargs):
        """Plot a tricontour."""
        return super().tricontour(*args, **kwargs)

    @remove_args_units
    def tricontourf(self, *args, **kwargs):
        """Plot a tricontourf."""
        return super().tricontourf(*args, **kwargs)

    @remove_args_units
    def annotate(self, *args, **kwargs):
        """Add an annotation to the axes."""
        return super().annotate(*args, **kwargs)

    @remove_args_units
    def text(self, *args, **kwargs):
        """Add text to the axes."""
        return super().text(*args, **kwargs)

    @remove_args_units
    def table(self, *args, **kwargs):
        """Add a table to the axes."""
        return super().table(*args, **kwargs)

    @remove_args_units
    def arrow(self, *args, **kwargs):
        """Add an arrow to the axes."""
        return super().arrow(*args, **kwargs)

    @remove_args_units
    def set_xlim(self, *args, **kwargs):
        """Set the x-axis limits."""
        return super().set_xlim(*args, **kwargs)

    @remove_args_units
    def set_ylim(self, *args, **kwargs):
        """Set the y-axis limits."""
        return super().set_ylim(*args, **kwargs)


class _Axes3D(maxes3D.Axes3D):  # pragma: no cover
    """Subclass of matplotlib Axes3D class."""

    from spectrochempy.core.units import remove_args_units

    def __init__(self, *args, **kwargs):
        """Initialize the 3D axes."""
        super().__init__(*args, **kwargs)

    @remove_args_units
    def plot_surface(self, *args, **kwargs):
        """Plot a surface."""
        return super().plot_surface(*args, **kwargs)


def figure(preferences=None, **kwargs):
    """
    Open a new figure.

    Parameters
    ----------
    Kwargs : any
        Keywords arguments to be passed to the matplotlib figure constructor.
    Preferences : Meta dictionary
        Per object saved plot configuration.

    """
    from spectrochempy.utils.meta import Meta

    if preferences is None:
        preferences = Meta()
    return get_figure(preferences=preferences, **kwargs)


def show():
    """Force the `matplotlib` figure display."""
    from spectrochempy import NO_DISPLAY

    if NO_DISPLAY:
        plt.close("all")
    elif get_figure(clear=False):
        plt.show(block=True)


def get_figure(**kwargs):
    """
    Get the figure where to plot.

    Parameters
    ----------
    clear : bool
        If False the last used figure is returned.
    figsize : 2-tuple of floats, default: rcParams["figure.figsize"])
        Figure dimension (width, height) in inches.
    dpi : float, default: rcParams["figure.dpi"] (default: 100.0)
        Dots per inch.
    facecolor : default: rcParams["figure.facecolor"] (default: 'white')
        The figure patch facecolor.
    edgecolor : default: preferences.figure_edgecolor (default: 'white')
        The figure patch edge color.
    frameon : bool, default: preferences.figure_frameon (default: True)
        If False, suppress drawing the figure background patch.
    tight_layout : bool or dict, default: preferences.figure.autolayout
        If False use subplotpars. If True adjust subplot parameters using tight_layout
        with default padding. When providing a dict containing the keys pad, w_pad,
        h_pad, and rect, the default tight_layout paddings will be overridden.
    constrained_layout : bool, default: preferences.figure_constrained_layout
        If True use constrained layout to adjust positioning of plot elements.
        Like tight_layout, but designed to be more flexible.
        See Constrained Layout Guide for examples.
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
        if prefs is None:
            return None

        figsize = kwargs.get("figsize", prefs.figure_figsize)
        dpi = int(kwargs.get("dpi", prefs.figure_dpi))
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
            fig.set_edgecolor(eval(edgecolor))  # noqa: S307
        try:
            fig.set_facecolor(facecolor)
        except ValueError:
            try:
                fig.set_facecolor(eval(facecolor))  # noqa: S307
            except ValueError:
                fig.set_facecolor("#" + eval(facecolor))  # noqa: S307
        fig.set_dpi(dpi)
        fig.set_tight_layout(tight_layout)

        return fig

    # a figure already exists - if several we take the last
    return plt.figure(n[-1])


# class colorscale:
#     def normalize(self, vmin, vmax, cmap="viridis", rev=False, offset=0):
#         """Normalize the color scale based on the given parameters."""
#         if rev:
#             cmap = cmap + "_r"
#         _colormap = plt.get_cmap(cmap)

#         _norm = mpl.colors.Normalize(vmin=vmin - offset, vmax=vmax - offset)
#         self.scalarMap = mpl.cm.ScalarMappable(norm=_norm, cmap=_colormap)

#     def rgba(self, z, offset=0):
#         """Return the rgba color for the given value."""
#         c = np.array(self.scalarMap.to_rgba(z.squeeze() - offset))
#         c[0:3] *= 255
#         c[0:3] = np.round(c[0:3].astype("uint16"), 0)
#         return f"rgba{tuple(c)}"


# colorscale = colorscale()


def make_label(ss, lab="<no_axe_label>", use_mpl=True):
    """Make a label from title and units."""
    from pint import __version__

    pint_version = int(__version__.split(".")[1])

    if ss is None:
        return lab

    label = ss.title if ss.title else lab  # .replace(' ', r'\ ')

    if "<untitled>" in label:
        label = "values"

    if use_mpl:
        if ss.units is not None and str(ss.units) not in [
            "dimensionless",
            "absolute_transmittance",
        ]:
            units = rf"/\ {ss.units:~L}"
            if pint_version < 24:
                units = units.replace("%", r"\%")
        else:
            units = ""
        label = rf"{label} $\mathrm{{{units}}}$"
    else:
        if ss.units is not None and str(ss.units) != "dimensionless":
            units = rf"{ss.units:~H}"
        else:
            units = ""
        label = rf"{label} / {units}"

    return label


# FOR PLOTLY
#


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
    from spectrochempy.utils.optional import import_optional_dependency

    go = import_optional_dependency("plotly.graph_objects", errors="ignore")

    if go is None:
        raise ImportError("Plotly is not installed. Uee pip or conda to install it")

    if clear or fig is None:
        # create a figure
        return go.Figure()

    # a figure already exists - if several we take the last
    return fig
