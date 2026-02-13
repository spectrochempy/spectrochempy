# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Matplotlib backend for SpectroChemPy.

This module provides the matplotlib implementation of plotting functions,
wrapping the plotting functions from the spectrochempy.plot module.
"""

from typing import Any, Optional

from spectrochempy.plot.plot_setup import lazy_ensure_mpl_config
from spectrochempy.utils.mplutils import show as mpl_show

# Mapping of method names to standalone plot functions
_PLOT_FUNCTIONS = {}


def _get_plot_function(method: str):
    """Lazily get the plot function for a given method."""
    if method not in _PLOT_FUNCTIONS:
        # Import all plot modules to populate the mapping
        from spectrochempy.plot import plot1d, plot2d, plot3d

        _PLOT_FUNCTIONS.update(
            {
                "pen": plot1d.plot_pen,
                "scatter": plot1d.plot_scatter,
                "bar": plot1d.plot_bar,
                "multiple": plot1d.plot_multiple,
                "scatter_pen": plot1d.plot_scatter_pen,
                "stack": plot2d.plot_stack,
                "map": plot2d.plot_map,
                "image": plot2d.plot_image,
                "surface": plot3d.plot_surface,
                "waterfall": plot3d.plot_waterfall,
                # Handle '+' in method names by replacing with '_'
                "scatter_pen".replace("+", "_"): plot1d.plot_scatter_pen,
            }
        )

    return _PLOT_FUNCTIONS.get(method.replace("+", "_") if method else method)


def plot_dataset_impl(
    dataset: Any,
    method: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Implementation of dataset plotting using matplotlib.

    Parameters
    ----------
    dataset : NDDataset
        The dataset to plot.
    method : str, optional
        Plotting method (e.g., "pen", "stack", "surface").
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

    # Determine default method based on dimensionality
    if method is None:
        if dataset._squeeze_ndim == 1:
            method = "pen"
        elif dataset._squeeze_ndim == 2:
            method = "stack"
        elif dataset._squeeze_ndim == 3:
            method = "surface"

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
