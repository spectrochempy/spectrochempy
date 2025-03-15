# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Module to handle a large set of plot types related to 3D plots."""

__all__ = ["plot_3D", "plot_surface", "plot_waterfall"]

__dataset_methods__ = __all__


from spectrochempy.core.dataset.arraymixins.ndplot import (
    NDPlot,  # noqa: F401 # for the docstring to be determined it necessary to import NDPlot
)
from spectrochempy.utils.docutils import docprocess


# ======================================================================================
# nddataset plot3D functions
# ======================================================================================
@docprocess.dedent
def plot_surface(dataset, **kwargs):
    """
    Plot a 2D dataset as a a 3D-surface.

    Parameters
    ----------
    %(plot.parameters.no_method)s

    Other Parameters
    ----------------
    %(plot.other_parameters)s

    Returns
    -------
    %(plot.returns)s

    See Also
    --------
    plot
    plot_2D
    plot_3D
    plot_stack
    plot_map
    plot_image
    plot_waterfall
    """
    return plot_3D(dataset, method="surface", **kwargs)


@docprocess.dedent
def plot_waterfall(dataset, **kwargs):
    """
    Plot a 2D dataset as a a 3D-waterfall plot.

    Parameters
    ----------
    %(plot.parameters.no_method)s

    Other Parameters
    ----------------
    %(plot.other_parameters)s

    Returns
    -------
    %(plot.returns)s

    See Also
    --------
    plot
    plot_2D
    plot_3D
    plot_stack
    plot_map
    plot_image
    plot_surface

    """
    return plot_3D(dataset, method="waterfall", **kwargs)


@docprocess.dedent
def plot_3D(dataset, method="surface", **kwargs):
    """
    Plot of 2D array as 3D plot.

    Parameters
    ----------
    %(plot.parameters)s

    Other Parameters
    ----------------
    %(plot.other_parameters)s

    Returns
    -------
    %(plot.returns)s

    See Also
    --------
    plot
    plot_2D
    plot_stack
    plot_map
    plot_image
    plot_surface
    plot_waterfall

    """
    from spectrochempy.core.plotters.plot2d import plot_2D

    return plot_2D(dataset, method=method, **kwargs)


# TODO: complete 3D method
