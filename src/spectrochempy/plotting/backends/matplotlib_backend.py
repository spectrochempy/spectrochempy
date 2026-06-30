# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Matplotlib backend for SpectroChemPy.

This module provides the matplotlib implementation of plotting functions,
wrapping the plotting functions from the spectrochempy.plotting module.
"""

from typing import Any

from spectrochempy.plotting._methods import get_default_method_for_ndim
from spectrochempy.plotting._methods import get_dispatch_method_key
from spectrochempy.plotting._methods import normalize_backend_method
from spectrochempy.plotting.plot_setup import lazy_ensure_mpl_config
from spectrochempy.plotting.profile import ensure_plot_profile_loaded
from spectrochempy.utils.mplutils import show as mpl_show

# Track which aliases we've warned about (warn once per session)
_WARNED_ALIASES = set()


# Mapping of method names to standalone plot functions
_PLOT_FUNCTIONS = {}


def _get_plot_function(method: str):
    """Lazily get the plot function for a given method."""
    if method not in _PLOT_FUNCTIONS:
        # Import all plot modules to populate the mapping
        from spectrochempy.plotting import plot1d
        from spectrochempy.plotting import plot2d
        from spectrochempy.plotting import plot3d

        _PLOT_FUNCTIONS.update(
            {
                # Canonical 1D methods
                "pen": plot1d.plot_pen,
                "scatter": plot1d.plot_scatter,
                "bar": plot1d.plot_bar,
                "multiple": plot1d.plot_multiple,
                "scatter_pen": plot1d.plot_scatter_pen,
                # Canonical 2D methods
                "lines": plot2d.plot_lines,
                "contour": plot2d.plot_contour,
                "contourf": plot2d.plot_contourf,
                # Legacy 2D methods (deprecated, point to canonical functions)
                "stack": plot2d.plot_stack,
                "map": plot2d.plot_map,
                "image": plot2d.plot_image,
                # 3D methods
                "surface": plot3d.plot_surface,
                "waterfall": plot3d.plot_waterfall,
            }
        )

    return _PLOT_FUNCTIONS.get(get_dispatch_method_key(method))


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

    Returns
    -------
    Any
        The matplotlib axes.
    """
    # Initialize matplotlib lazily
    lazy_ensure_mpl_config()

    # Initialize plot profile lazily (loads defaults into PlotPreferences)
    ensure_plot_profile_loaded()

    # Determine default method based on dimensionality
    if method is None:
        method = get_default_method_for_ndim(dataset._squeeze_ndim)

    # NORMALIZE METHOD - Convert legacy names to canonical BEFORE dispatch
    method = normalize_backend_method(method, warned_aliases=_WARNED_ALIASES)

    # Get the standalone plot function
    plot_func = _get_plot_function(method) if method else None
    if plot_func is None:
        from spectrochempy.utils._logging import error_

        error_(
            NameError,
            f"The specified plotter for method `{method}` was not found!",
        )
        raise OSError

    # Handle show parameter
    show = kwargs.pop("show", True)

    # Call the standalone plot function with dataset as first argument
    ax = plot_func(dataset, **kwargs)

    # Show the figure if requested
    if show:
        mpl_show()

    return ax
