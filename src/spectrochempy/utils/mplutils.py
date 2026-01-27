# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Matplotlib utilities used across SpectroChemPy.

- Custom Axes classes supporting pint quantities
- get_figure helper with lazy Matplotlib initialization
"""

from contextlib import suppress

import matplotlib.axes as maxes
import mpl_toolkits.mplot3d.axes3d as maxes3D  # noqa: N812

from spectrochempy.core.plotters._mpl_setup import ensure_mpl_setup

__all__ = [
    "show",
    "get_figure",
    "make_label",
    "get_plotly_figure",
    "_Axes",
    "_Axes3D",
]

plt = None  # will be initialized lazily


@maxes.subplot_class_factory
class _Axes(maxes.Axes):  # pragma: no cover
    """Subclass of matplotlib Axes class."""

    from spectrochempy.core.units import remove_args_units

    def _implements(self, type=None):
        if type is None:
            return "_Axes"
        return type == "_Axes"

    def __repr__(self):
        return "<Matplotlib Axes object>"

    def __str__(self):
        return self.__repr__()

    def _repr_html_(self):
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
    def contour(self, *args, **kwargs):
        return super().contour(*args, **kwargs)

    @remove_args_units
    def contourf(self, *args, **kwargs):
        return super().contourf(*args, **kwargs)

    @remove_args_units
    def imshow(self, *args, **kwargs):
        return super().imshow(*args, **kwargs)

    @remove_args_units
    def set_xlim(self, *args, **kwargs):
        return super().set_xlim(*args, **kwargs)

    @remove_args_units
    def set_ylim(self, *args, **kwargs):
        return super().set_ylim(*args, **kwargs)


class _Axes3D(maxes3D.Axes3D):  # pragma: no cover
    """Subclass of matplotlib Axes3D class."""

    from spectrochempy.core.units import remove_args_units

    @remove_args_units
    def plot_surface(self, *args, **kwargs):
        return super().plot_surface(*args, **kwargs)


def show():
    """Force the matplotlib figure display."""
    ensure_mpl_setup()

    global plt
    if plt is None:
        import matplotlib.pyplot as plt  # noqa: F401

        globals()["plt"] = plt

    from spectrochempy import NO_DISPLAY
    from spectrochempy.utils.mplutils import get_figure

    if NO_DISPLAY:
        plt.close("all")
        return

    fig = get_figure(clear=False)
    if fig:
        plt.show(block=True)


def get_figure(**kwargs):
    """Return a Matplotlib figure for plotting (pyplot-free)."""
    ensure_mpl_setup()

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from spectrochempy.application.preferences import preferences as _global_prefs

    prefs = kwargs.pop("preferences", None) or _global_prefs

    figsize = kwargs.get("figsize") or getattr(prefs, "figure_figsize", None)
    dpi = kwargs.get("dpi") or getattr(prefs, "figure_dpi", 100)

    try:
        dpi = int(dpi)
    except Exception:
        dpi = 100

    facecolor = kwargs.get("facecolor", getattr(prefs, "figure_facecolor", "white"))
    edgecolor = kwargs.get("edgecolor", getattr(prefs, "figure_edgecolor", "white"))
    frameon = kwargs.get("frameon", getattr(prefs, "figure_frameon", True))
    tight_layout = kwargs.get("autolayout", getattr(prefs, "figure_autolayout", False))

    fig = Figure(figsize=figsize, dpi=dpi, frameon=frameon)
    FigureCanvasAgg(fig)

    with suppress(Exception):
        fig.set_edgecolor(facecolor)

    with suppress(Exception):
        fig.set_edgecolor(edgecolor)

    with suppress(Exception):
        fig.set_tight_layout(tight_layout)

    return fig


def figure(*args, **kwargs):
    """Backward-compatible alias for get_figure."""
    return get_figure(*args, **kwargs)


def make_label(ss, lab="<no_axe_label>", use_mpl=True):
    """Make a label from title and units."""
    from pint import __version__

    pint_version = int(__version__.split(".")[1])

    if ss is None:
        return lab

    label = ss.title if ss.title else lab

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


def get_plotly_figure(clear=True, fig=None, **kwargs):
    """Get a Plotly figure for plotting."""
    from spectrochempy.utils.optional import import_optional_dependency

    go = import_optional_dependency("plotly.graph_objects", errors="ignore")

    if go is None:
        raise ImportError("Plotly is not installed. Use pip or conda to install it")

    if clear or fig is None:
        return go.Figure()

    return fig
