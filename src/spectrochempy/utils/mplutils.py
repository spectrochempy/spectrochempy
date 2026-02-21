# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Matplotlib utilities used across SpectroChemPy.

Responsibilities:
- Custom Axes classes supporting pint quantities
- Figure factory (headless-safe)
- Explicit, non-invasive figure display helper
"""

from contextlib import suppress

__all__ = [
    "show",
    "get_figure",
    "figure",  # backward compatibility
    "make_label",
    "get_plotly_figure",
    "_Axes",
    "_Axes3D",
]


# ----------------------------------------------------------------------
# Lazy loading: matplotlib is only imported when plotting functions are called
# ----------------------------------------------------------------------


def __getattr__(name):
    """Lazily import matplotlib classes when first accessed."""
    if name == "_Axes":
        import matplotlib.axes as maxes

        @maxes.subplot_class_factory
        class _Axes(maxes.Axes):  # pragma: no cover
            """Subclass of matplotlib Axes class supporting pint quantities."""

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

        return _Axes

    elif name == "_Axes3D":
        import mpl_toolkits.mplot3d.axes3d as maxes3D

        class _Axes3D(maxes3D.Axes3D):  # pragma: no cover
            """Subclass of matplotlib Axes3D supporting pint quantities."""

            from spectrochempy.core.units import remove_args_units

            @remove_args_units
            def plot_surface(self, *args, **kwargs):
                return super().plot_surface(*args, **kwargs)

        return _Axes3D

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# -----------------------------------------------------------------------------
# Figure handling
# -----------------------------------------------------------------------------


def get_figure(**kwargs):
    """
    Return a Matplotlib figure.

    - Uses pyplot in all modes (figures are tracked by pyplot.gcf())
    - Agg backend is set by matplotlib.use() in test environments
    - Does NOT trigger application initialization
    """
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

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, dpi=dpi, frameon=frameon)

    with suppress(Exception):
        fig.set_facecolor(facecolor)

    with suppress(Exception):
        fig.set_edgecolor(edgecolor)

    with suppress(Exception):
        fig.set_tight_layout(tight_layout)

    _apply_window_position(fig, prefs)

    return fig


def _apply_window_position(fig, prefs):
    """Apply window position preference for TkAgg backend."""
    import matplotlib

    backend = matplotlib.get_backend().lower()
    if "tkagg" not in backend:
        return

    window_position = getattr(prefs, "figure_window_position", None)
    if window_position is None:
        return

    try:
        import matplotlib.pyplot as plt

        manager = plt.get_current_fig_manager()
        x, y = window_position
        manager.window.wm_geometry(f"+{x}+{y}")
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Backward compatibility
# -----------------------------------------------------------------------------

figure = get_figure

# -----------------------------------------------------------------------------
# Backward compatibility
# -----------------------------------------------------------------------------


def show():
    """
    Force display of existing Matplotlib figures.

    - Never creates figures
    - Safe in scripts, IDEs, and notebooks
    - Respects non-interactive backends (Agg, template)
    - In interactive mode, figures display automatically - no show() needed
    """
    import matplotlib

    from spectrochempy import NO_DISPLAY

    if NO_DISPLAY:
        return

    if matplotlib.is_interactive():
        # In interactive mode, figures display automatically
        # Calling plt.show(block=True) would clear figure tracking
        return

    import matplotlib.pyplot as plt

    if plt.get_fignums():
        plt.show(block=True)


# -----------------------------------------------------------------------------
# Misc helpers
# -----------------------------------------------------------------------------


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
