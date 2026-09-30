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
- Shared figure saving, driven by the ``savefig`` preferences
"""

from contextlib import suppress
from os import PathLike

__all__ = [
    "show",
    "get_figure",
    "figure",  # backward compatibility
    "make_label",
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

    if name == "_Axes3D":
        import mpl_toolkits.mplot3d.axes3d as maxes3d

        class _Axes3D(maxes3d.Axes3D):  # pragma: no cover
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


def _setup_axes(ax=None, *, clear=True, projection=None):
    """
    Create or prepare a Matplotlib Axes for plotting.

    This helper centralises the common figure/axes lifecycle used by
    composite plot functions.  It avoids duplicating the *ax=None* /
    *clear* pattern across ``plot_score``, ``plot_scree``,
    ``plot_compare``, ``plot_merit``, and ``plot_baseline``.

    Parameters
    ----------
    ax : Axes or None
        If *None*, a new figure and subplot are created.
        If provided, the axes may be cleared depending on *clear*.
    clear : bool
        Only meaningful when *ax* is provided.
        If *True* (default), clear the axes (``ax.clear()``).
        If *False*, leave existing artists untouched.
    projection : str or None
        Matplotlib projection type passed to ``add_subplot`` when
        creating new axes (e.g. ``"3d"``).

    Returns
    -------
    Axes
    """
    if ax is None:
        fig = get_figure()
        ax = fig.add_subplot(111, projection=projection)
    elif clear:
        ax.clear()
    return ax


def _maybe_show(do_show=True):
    """Display the current figure if *do_show* is *True*."""
    if do_show:
        show()


def _resolve_save_figure(target):
    """
    Return the Matplotlib figure carrying a complete plot.

    Plotting functions return an axes, a figure, or - for composite two-panel
    layouts - a tuple of axes belonging to a single figure. This helper maps any
    of those results to the figure that must be written to disk, so that saving
    never depends on which figure happens to be globally active.

    Parameters
    ----------
    target : `~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`, or sequence of Axes
        Result of a plotting call.

    Returns
    -------
    `~matplotlib.figure.Figure`
        Figure that holds the whole plot.

    Raises
    ------
    TypeError
        If *target* is not a figure, an axes, or a sequence of axes.
    ValueError
        If a sequence of axes is given whose members belong to different figures.
    """
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    if isinstance(target, Figure):
        return target

    if isinstance(target, Axes):
        return target.figure

    if isinstance(target, (tuple, list)):
        figures = []
        for item in target:
            figure = _resolve_save_figure(item)
            if not any(figure is known for known in figures):
                figures.append(figure)
        if len(figures) > 1:
            raise ValueError(
                "A single output file cannot hold panels belonging to different "
                "figures. Use one `output` per figure.",
            )
        if not figures:
            raise ValueError("No figure to save.")
        return figures[0]

    raise TypeError(
        f"Cannot determine the figure to save from a {type(target).__name__} object.",
    )


def _save_figure(fig, output):
    """
    Save *fig* to *output* using the SpectroChemPy ``savefig`` preferences.

    This is the single place where SpectroChemPy writes figure files. It only
    forwards the ``savefig`` preferences to :meth:`~matplotlib.figure.Figure.savefig`
    so that formats, extensions, and overwrite behaviour stay those of
    Matplotlib: the file is written exactly at *output*, an existing file is
    overwritten, and missing parent directories are reported rather than
    created.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        Figure to write.
    output : str or `pathlib.Path`
        Destination file. ``str`` and :class:`pathlib.Path` are both accepted,
        following the SpectroChemPy path conventions.

    Raises
    ------
    TypeError
        If *output* is neither a string nor a path-like object.
    OSError
        If the file cannot be written. The original error is chained.
    """
    from spectrochempy.application.preferences import preferences as prefs
    from spectrochempy.utils.file import pathclean

    if not isinstance(output, (str, PathLike)):
        raise TypeError(
            f"`output` must be a str or a pathlib.Path, not {type(output).__name__}.",
        )

    path = pathclean(str(output))

    # The `savefig` preferences are declared as text traits, while Matplotlib
    # expects a number for `dpi` and `None`, "tight", or a Bbox for
    # `bbox_inches`. Translate the legacy values once, here.
    dpi = prefs.savefig_dpi
    if isinstance(dpi, str) and dpi != "figure":
        dpi = float(dpi)

    bbox_inches = prefs.savefig_bbox
    if bbox_inches == "standard":
        # "standard" means the full figure, which is the Matplotlib default.
        bbox_inches = None

    transparent = prefs.savefig_transparent

    savefig_options = {
        "dpi": dpi,
        "bbox_inches": bbox_inches,
        "pad_inches": prefs.savefig_pad_inches,
        "transparent": transparent,
    }
    if not transparent:
        # Matplotlib already makes the whole figure transparent when
        # `transparent=True`; an explicit background color would defeat it.
        savefig_options["facecolor"] = prefs.savefig_facecolor
        savefig_options["edgecolor"] = prefs.savefig_edgecolor

    # Without a suffix Matplotlib cannot infer a format: fall back on the
    # configured default format rather than renaming the requested file.
    if not path.suffix:
        savefig_options["format"] = prefs.savefig_format

    try:
        fig.savefig(path, **savefig_options)
    except Exception as exc:
        message = f"Could not save the figure to '{path}': {exc}"
        error_type = type(exc) if isinstance(exc, OSError) else OSError
        raise error_type(message) from exc


def _finalize_plot(target, *, show=True, output=None):
    """
    Shared final step of a plotting call: optional saving, then optional display.

    Both the dataset plotting backend and the composite plotters close their
    plot with this helper, so ``show`` and ``output`` behave identically
    everywhere. Saving happens before any blocking display so that the file is
    on disk even when ``show=True`` keeps a window open.

    Parameters
    ----------
    target : `~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`, or sequence of Axes
        Completed plot, as returned by the plotting function.
    show : bool, optional, default: True
        Whether SpectroChemPy should perform its explicit display step.
    output : str or `pathlib.Path`, optional
        Destination file for the whole figure. When given, the figure is saved
        once the plot is complete - after all panels, legends, and colorbars.
    """
    if output is not None:
        _save_figure(_resolve_save_figure(target), output)

    _maybe_show(show)


def _apply_window_position(fig, prefs):
    """Apply window position preference for TkAgg backend."""
    import matplotlib

    backend = matplotlib.get_backend().lower()
    if "tkagg" not in backend:
        return

    window_position = getattr(prefs, "figure_window_position", None)
    if window_position is None:
        return

    with suppress(Exception):
        import matplotlib.pyplot as plt

        manager = plt.get_current_fig_manager()
        x, y = window_position
        manager.window.wm_geometry(f"+{x}+{y}")


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

    has_display_units = ss.units is not None and str(ss.units) not in [
        "dimensionless",
        "absolute_transmittance",
    ]

    if use_mpl:
        if has_display_units:
            units = rf"/\ {ss.units:~L}"
            if pint_version < 24:
                units = units.replace("%", r"\%")
            label = rf"{label} $\mathrm{{{units}}}$"
    else:
        if has_display_units:
            units = rf"{ss.units:~H}"
            label = rf"{label} / {units}"

    return label
