# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Matplotlib backend for SpectroChemPy.

Current responsibility split:

- `_methods.py` normalizes plotting vocabulary;
- `_kwargs.py` normalizes plotting keyword arguments;
- this backend selects the plotter and owns the final `show` and `output`
  steps for the main dataset plotting path;
- `plot1d.py`, `plot2d.py`, and `plot3d.py` create matplotlib artists.
"""

from importlib import import_module
from typing import Any

from spectrochempy.plotting._kwargs import normalize_plot_kwargs
from spectrochempy.plotting._methods import get_default_method_for_ndim
from spectrochempy.plotting._methods import get_dispatch_method_key
from spectrochempy.plotting._methods import normalize_backend_method
from spectrochempy.plotting.plot_setup import lazy_ensure_mpl_config
from spectrochempy.plotting.profile import ensure_plot_profile_loaded
from spectrochempy.utils.mplutils import _finalize_plot

# Track which aliases we've warned about (warn once per session)
_WARNED_ALIASES = set()


# Canonical dispatch key -> (module, renderer name, renderer method string).
# The dispatcher calls the renderer with the method already set, while the
# public shortcuts (plot_pen, plot_map, ...) delegate through this same
# lifecycle step. Storing names keeps this module free of any import of the
# plot modules and prevents the shortcuts from recursing into the dispatcher.
_PLOT_FUNCTIONS = {
    "pen": ("spectrochempy.plotting.plot1d", "plot_1D", "pen"),
    "scatter": ("spectrochempy.plotting.plot1d", "plot_1D", "scatter"),
    "scatter_pen": (
        "spectrochempy.plotting.plot1d",
        "plot_1D",
        "scatter_pen",
    ),
    "bar": ("spectrochempy.plotting.plot1d", "plot_1D", "bar"),
    "multiple": ("spectrochempy.plotting.plot1d", "plot_multiple", None),
    "lines": ("spectrochempy.plotting.plot2d", "plot_2D", "lines"),
    "contour": ("spectrochempy.plotting.plot2d", "plot_2D", "contour"),
    "contourf": ("spectrochempy.plotting.plot2d", "plot_2D", "contourf"),
    "surface": ("spectrochempy.plotting.plot3d", "plot_3D", "surface"),
    "waterfall": ("spectrochempy.plotting.plot2d", "plot_2D", "waterfall"),
}


def _get_plot_function(method: str):
    """Return ``(renderer_callable, renderer_method)`` for a dispatch key."""
    entry = _PLOT_FUNCTIONS.get(get_dispatch_method_key(method))
    if entry is None:
        return None

    module_name, func_name, renderer_method = entry
    func = getattr(import_module(module_name), func_name)
    return func, renderer_method


def plot_dataset_impl(
    dataset: Any,
    method: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Implement dataset plotting using matplotlib.

    Parameters
    ----------
    dataset : NDDataset
        The dataset to plot.
    method : str, optional
        Plotting method (e.g., "pen", "lines", "surface").
        If None, method is chosen based on data dimensionality.
    **kwargs
        Additional arguments passed to the plotting function.
        ``show`` controls whether SpectroChemPy performs its explicit display
        step after plotting. In notebook environments, figures may still
        render inline without that explicit call.
        ``output`` is the destination file of the finished figure.

    Returns
    -------
    Any
        The matplotlib axes.
    """
    # Ensure environment is set up BEFORE importing matplotlib.
    # In notebooks/nbsphinx, setup_environment() activates %matplotlib inline
    # which switches the backend.  If we import matplotlib first with the
    # default backend (Agg) and *then* switch, the existing figures are
    # closed by plt.switch_backend() -> plt.close('all'), and we lose the
    # plot.  See note in envsetup.py for details.
    from spectrochempy.application.application import _get_environment

    _get_environment()

    # Initialize matplotlib lazily
    lazy_ensure_mpl_config()
    kwargs = normalize_plot_kwargs(kwargs)

    # Initialize plot profile lazily (loads defaults into PlotPreferences)
    ensure_plot_profile_loaded()

    # Determine default method based on dimensionality
    if method is None:
        method = get_default_method_for_ndim(dataset._squeeze_ndim)

    # NORMALIZE METHOD - Convert legacy names to canonical BEFORE dispatch
    method = normalize_backend_method(method, warned_aliases=_WARNED_ALIASES)

    # Get the standalone plot function
    plot_func_data = _get_plot_function(method) if method else None
    if plot_func_data is None:
        from spectrochempy.utils._logging import error_

        error_(
            NameError,
            f"The specified plotter for method `{method}` was not found!",
        )
        raise OSError

    # Handle the lifecycle parameters owned by this layer
    show = kwargs.pop("show", True)
    output = kwargs.pop("output", None)

    # Call the renderer with the method already set, so the public shortcuts
    # can delegate here without recursing into each other
    plot_func, render_method = plot_func_data
    if render_method is None:
        ax = plot_func(dataset, **kwargs)
    else:
        ax = plot_func(dataset, method=render_method, **kwargs)

    # Save and/or display the completed figure
    _finalize_plot(ax, show=show, output=output)

    return ax
