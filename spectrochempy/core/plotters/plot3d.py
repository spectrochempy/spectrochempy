# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
This module should be able to handle a large set of plot types related to 3D plots.
"""

__all__ = ["plot_surface", "plot_waterfall"]

__dataset_methods__ = __all__

from spectrochempy.utils import plot_method, add_docstring

_PLOT3D_DOC = """
ax : |Axes| instance. Optional
    The axe where to plot. The default is the current axe or to create a new one if is None.
figsize : tuple, optional
    The figure size expressed as a tuple (w,h) in inch.
fontsize : int, optional
    The font size in pixels, default is 10 (or read from preferences).
autolayout : `bool`, optional, default=True
    if True, layout will be set automatically.
dpi : [ None | scalar > 0]
    The resolution in dots per inch. If None it will default to the
    value savefig.dpi in the matplotlibrc file.
colorbar :
method : str [optional among ``surface``, ``waterfall``, ...]
    The type of plot,
style : str, optional, default='notebook'
    Matplotlib stylesheet (use `available_style` to get a list of available
    styles for plotting
reverse : `bool` or None [optional, default=None
    In principle, coordinates run from left to right, except for wavenumbers
    (e.g., FTIR spectra) or ppm (e.g., NMR), that spectrochempy
    will try to guess. But if reverse is set, then this is the
    setting which will be taken into account.
x_reverse : `bool` or None [optional, default=None
y_reverse : `bool` or None [optional, default=None
"""

# ======================================================================================================================
# nddataset plot3D functions
# ======================================================================================================================


@plot_method("2D", _PLOT3D_DOC)
def plot_surface(dataset, **kwargs):
    """
    Plot a 2D dataset as a a 3D-surface.

    Alias of plot_3D (with `method` argument set to ``surface``).
    """
    return


@plot_method("2D", _PLOT3D_DOC)
def plot_waterfall(dataset, **kwargs):
    """
    Plot a 2D dataset as a a 3D-waterfall plot.

    Alias of plot_2D (with `method` argument set to ``waterfall``).
    """
    return


@add_docstring(_PLOT3D_DOC)
def plot_3D(dataset, method="surface", **kwargs):
    """
    Plot of 2D array as 3D plot

    Parameters
    ----------
    dataset : |NDDataset|
        The dataset to plot.
    method : ['surface', 'waterfall'] , optional
        The method of plot of the dataset, which will determine the plotter to use. Default is stack.
    **kwargs : dic, optional
        Additional keywords parameters.
        See Other Parameters.

    Other Parameters
    ----------------
    {0}

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
    plot_surface
    plot_waterfall
    multiplot
    """
    from spectrochempy.core.plotters.plot2d import plot_2D

    return plot_2D(dataset, method=method, **kwargs)


# TODO: complete 3D method
