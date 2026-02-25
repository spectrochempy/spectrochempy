# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Plotters."""

__all__ = [
    "plot_2D",
    "plot_contour",
    "plot_contourf",
    "plot_image",
    "plot_lines",
    "plot_map",
    "plot_stack",
]
__dataset_methods__ = __all__


from contextlib import suppress

import numpy as np
from matplotlib.ticker import MaxNLocator

from spectrochempy.application.preferences import preferences
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.plotting._colorbar_utils import _apply_colorbar_tick_policy
from spectrochempy.plotting._render import render_lines
from spectrochempy.plotting._style import resolve_2d_colormap
from spectrochempy.plotting._style import resolve_line_style
from spectrochempy.plotting._style import resolve_stack_colors
from spectrochempy.utils.mplutils import make_label


# ======================================================================================
# Helper functions for aspect ratio control
# ======================================================================================


def _can_enforce_equal_aspect(dataset):
    """
    Check if equal aspect ratio can be enforced for a 2D dataset.

    Returns True if both plotted dimensions have compatible units
    (identical units or both have no units).
    """
    if dataset is None:
        return False

    try:
        dims = dataset.dims
        if dims is None or len(dims) < 2:
            return False

        dim_y = dims[-2]
        dim_x = dims[-1]

        y_coord = dataset.coord(dim_y)
        x_coord = dataset.coord(dim_x)

        if x_coord is None or y_coord is None:
            return False

        x_units = getattr(x_coord, "units", None)
        y_units = getattr(y_coord, "units", None)

        if x_units is None and y_units is None:
            return True

        if x_units is None or y_units is None:
            return False

        if x_units == y_units:
            return True

        return False
    except Exception:
        return False


def _handle_3d_aspect(ax, dataset, **kwargs):
    """
    Handle aspect ratio for 3D surface plots.

    Parameters
    ----------
    ax : matplotlib Axes
        The 3D axes.
    dataset : NDDataset
        The dataset being plotted.
    **kwargs
        Additional keyword arguments including equal_aspect.
    """
    import warnings as _warn_module

    equal_aspect = kwargs.get("equal_aspect", "xy")

    if not equal_aspect:
        return

    try:
        x_coord = getattr(dataset, "x", None)
        y_coord = getattr(dataset, "y", None)
        z_data = dataset.masked_data

        if x_coord is None or y_coord is None or z_data is None:
            _warn_module.warn(
                "equal_aspect: Cannot determine axis ranges. Using default aspect.",
                UserWarning,
                stacklevel=2,
            )
            return

        x_data = x_coord.data
        y_data = y_coord.data

        x_range = float(np.nanmax(x_data) - np.nanmin(x_data))
        y_range = float(np.nanmax(y_data) - np.nanmin(y_data))
        z_range = float(np.nanmax(z_data) - np.nanmin(z_data))

        x_units = getattr(x_coord, "units", None)
        y_units = getattr(y_coord, "units", None)

        max_range = max(x_range, y_range)

        if equal_aspect == "xy":
            if x_units is not None and y_units is not None and x_units != y_units:
                _warn_module.warn(
                    "equal_aspect='xy' ignored: X and Y units are incompatible.",
                    UserWarning,
                    stacklevel=2,
                )
                return

            if max_range > 0:
                ax.set_box_aspect(
                    (x_range / max_range, y_range / max_range, z_range / max_range)
                )

        elif equal_aspect == "xyz":
            if x_units is not None and y_units is not None:
                has_same_units = x_units == y_units
            else:
                has_same_units = x_units is None and y_units is None

            if not has_same_units:
                _warn_module.warn(
                    "equal_aspect='xyz' ignored: X, Y, Z units are not all compatible. "
                    "Falling back to 'xy'.",
                    UserWarning,
                    stacklevel=2,
                )
                if max_range > 0:
                    ax.set_box_aspect(
                        (
                            x_range / max_range,
                            y_range / max_range,
                            z_range / max_range,
                        )
                    )
                return

            ax.set_box_aspect((1, 1, 1))

    except Exception as e:
        _warn_module.warn(
            f"equal_aspect: Could not set aspect ratio ({e}). Using default.",
            UserWarning,
            stacklevel=2,
        )


# ======================================================================================
# nddataset plot2D functions - Canonical API
# ======================================================================================


def plot_lines(dataset, **kwargs):
    """
    Plot a 2D dataset as a stack plot with lines.

    This is the canonical geometry-based name for stack plots.

    Parameters
    ----------
    dataset : :class:~spectrochempy.ddataset.nddataset.NDDataset
        Source of data to plot.
    method : str, optional, default: auto-detected from data dimensionality
        Name of plotting method to use. If None, method is chosen based on data
        dimensionality.

        2D plotting methods:

        - `lines` : Stacked plot (canonical)
        - `contour` : Contour plot
        - `contourf` : Filled contour (image-like) plot
        - `surface` : Surface plot
        - `waterfall` : Waterfall plot

    **kwargs
        Additional matplotlib / plotting keyword arguments.

    Other Parameters
    ----------------
    ax : Axe, optional
        Axe where to plot. If not specified, create a new one.
    clear : bool, optional, default: True
        If false, hold the current figure and ax until a new plot is performed.
    color or c : color, optional, default: auto
        color of the line.
    colorbar : bool, optional, default: True
        Show colorbar (2D plots only).
    commands : str,
        matplotlib commands to be executed.
    data_only : bool, optional, default: False
        Only the plot is done. No addition of axes or label specifications.
    dpi : int, optional
        the number of pixel per inches.
    figsize : tuple, optional, default is (3.4, 1.7)
        figure size.
    fontsize : int, optional
        The font size in pixels, default is 10 (or read from preferences).
    imag : bool, optional, default: False
        Show imaginary component for complex data. By default the real component is
        displayed.
    linestyle or ls : str, optional, default: auto
        line style definition.
    linewidth or lw : float, optional, default: auto
        line width.
    marker, m: str, optional, default: auto
        marker type for scatter plot. If marker != "" then the scatter type of plot is chosen automatically.
    markeredgecolor or mec: color, optional
    markeredgewidth or mew: float, optional
    markerfacecolor or mfc: color, optional
    markersize or ms: float, optional
    markevery: None or int
    modellinestyle or modls : str
        line style of the model.
    offset : float
        offset of the model individual lines.
    output : str,
        name of the file to save the figure.
    palette : {"auto", "categorical", "continuous"} or str or list, optional
        Color palette for stack plot (plot_lines only).

        - "auto" (default): detect from dataset semantics.
        - "categorical": use matplotlib color cycle.
        - "continuous": use sequential colormap.
        - colormap name: continuous mapping using that colormap.
        - list of colors: explicit categorical colors.
    plot_model : Bool,
        plot model data if available.
    plottitle: bool, optional, default: False
        Use the name of the dataset as title. Works only if title is not defined
    projections : bool, optional, default: False
        Show projections on the axes (2D plots only).
    reverse : bool or None [optional, default=None/False
        In principle, coordinates run from left to right,
        except for wavenumbers
        (e.g., FTIR spectra) or ppm (e.g., NMR), that spectrochempy
        will try to guess. But if reverse is set, then this is the
        setting which will be taken into account.
    show_complex : bool, optional, default: False
        Show both real and imaginary component for complex data.
        By default only the real component is displayed.
    show_mask: bool, optional
        Should we display the mask using colored area.
    show_z : bool, optional, default: True
        should we show the vertical axis.
    show_zero : bool, optional
        show the zero basis.
    style : str, optional, default: scp.preferences.style (scpy)
        Matplotlib stylesheet (use available_style to get a list of available
        styles for plotting.
    title : str
        Title of the plot (or subplot) axe.
    transposed : bool, optional, default: False
        Transpose the data before plotting (2D plots only).
    twinx : :class:~matplotlib.axes.Axes instance, optional, default: None
        If this is not None, then a twin axes will be created with a
        common x dimension.
    uselabel_x: bool, optional
        use x coordinate label as x tick labels
    vshift : float, optional
        vertically shift the line from its baseline.
    xlim : tuple, optional
        limit on the horizontal axis.
    xlabel : str, optional
        label on the horizontal axis.
    x_reverse : bool, optional, default: False
        reverse the x axis. Equivalent to reverse.
    ylabel or zlabel : str, optional
        label on the vertical axis.
    ylim or zlim : tuple, optional
        limit on the vertical axis.
    y_reverse : bool, optional, default: False
        reverse the y axis (2D plot only).

    Returns
    -------
    Matplolib Axes or None
        The matplotlib axes containing the plot if successful, None otherwise.

    See Also
    --------
    plot
    plot_2D
    plot_contour
    plot_contourf
    plot_surface
    plot_waterfall
    plot_stack
    """
    return plot_2D(dataset, method="lines", **kwargs)


def plot_contour(dataset, **kwargs):
    """
    Plot a 2D dataset as a contour plot.

    This is the canonical geometry-based name for map (contour) plots.

    Parameters
    ----------
    dataset : :class:~spectrochempy.ddataset.nddataset.NDDataset
        Source of data to plot.
    method : str, optional, default: auto-detected from data dimensionality
        Name of plotting method to use. If None, method is chosen based on data
        dimensionality.

        2D plotting methods:

        - `lines` : Stacked plot
        - `contour` : Contour plot (canonical)
        - `contourf` : Filled contour (image-like) plot
        - `surface` : Surface plot
        - `waterfall` : Waterfall plot

    **kwargs
        Additional matplotlib / plotting keyword arguments.

    Other Parameters
    ----------------
    ax : Axe, optional
        Axe where to plot. If not specified, create a new one.
    clear : bool, optional, default: True
        If false, hold the current figure and ax until a new plot is performed.
    color or c : color, optional, default: auto
        color of the line.
    colorbar : bool, optional, default: True
        Show colorbar (2D plots only).
    commands : str,
        matplotlib commands to be executed.
    data_only : bool, optional, default: False
        Only the plot is done. No addition of axes or label specifications.
    dpi : int, optional
        the number of pixel per inches.
    figsize : tuple, optional, default is (3.4, 1.7)
        figure size.
    fontsize : int, optional
        The font size in pixels, default is 10 (or read from preferences).
    imag : bool, optional, default: False
        Show imaginary component for complex data. By default the real component is
        displayed.
    linestyle or ls : str, optional, default: auto
        line style definition.
    linewidth or lw : float, optional, default: auto
        line width.
    marker, m: str, optional, default: auto
        marker type for scatter plot. If marker != "" then the scatter type of plot is chosen automatically.
    markeredgecolor or mec: color, optional
    markeredgewidth or mew: float, optional
    markerfacecolor or mfc: color, optional
    markersize or ms: float, optional
    markevery: None or int
    modellinestyle or modls : str
        line style of the model.
    offset : float
        offset of the model individual lines.
    output : str,
        name of the file to save the figure.
    plot_model : Bool,
        plot model data if available.
    plottitle: bool, optional, default: False
        Use the name of the dataset as title. Works only if title is not defined
    projections : bool, optional, default: False
        Show projections on the axes (2D plots only).
    reverse : bool or None [optional, default=None/False
        In principle, coordinates run from left to right,
        except for wavenumbers
        (e.g., FTIR spectra) or ppm (e.g., NMR), that spectrochempy
        will try to guess. But if reverse is set, then this is the
        setting which will be taken into account.
    show_complex : bool, optional, default: False
        Show both real and imaginary component for complex data.
        By default only the real component is displayed.
    show_mask: bool, optional
        Should we display the mask using colored area.
    show_z : bool, optional, default: True
        should we show the vertical axis.
    show_zero : bool, optional
        show the zero basis.
    style : str, optional, default: scp.preferences.style (scpy)
        Matplotlib stylesheet (use available_style to get a list of available
        styles for plotting.
    title : str
        Title of the plot (or subplot) axe.
    transposed : bool, optional, default: False
        Transpose the data before plotting (2D plots only).
    twinx : :class:~matplotlib.axes.Axes instance, optional, default: None
        If this is not None, then a twin axes will be created with a
        common x dimension.
    uselabel_x: bool, optional
        use x coordinate label as x tick labels
    vshift : float, optional
        vertically shift the line from its baseline.
    xlim : tuple, optional
        limit on the horizontal axis.
    xlabel : str, optional
        label on the horizontal axis.
    x_reverse : bool, optional, default: False
        reverse the x axis. Equivalent to reverse.
    ylabel or zlabel : str, optional
        label on the vertical axis.
    ylim or zlim : tuple, optional
        limit on the vertical axis.
    y_reverse : bool, optional, default: False
        reverse the y axis (2D plot only).
    equal_aspect : bool, optional, default: False
        If True and X/Y units are compatible, enforce metric scaling
        (square pixels). This ensures 1 unit in X equals 1 unit in Y.

    Returns
    -------
    Matplolib Axes or None
        The matplotlib axes containing the plot if successful, None otherwise.

    See Also
    --------
    plot
    plot_2D
    plot_lines
    plot_contourf
    plot_surface
    plot_waterfall
    plot_map
    """
    return plot_2D(dataset, method="contour", **kwargs)


def plot_contourf(dataset, **kwargs):
    """
    Plot a 2D dataset as a filled contour (image-like) plot.

    This is the canonical geometry-based name for image plots.
    Uses high-resolution filled contours (500 levels by default).

    Parameters
    ----------
    dataset : :class:~spectrochempy.ddataset.nddataset.NDDataset
        Source of data to plot.
    method : str, optional, default: auto-detected from data dimensionality
        Name of plotting method to use. If None, method is chosen based on data
        dimensionality.

        2D plotting methods:

        - `lines` : Stacked plot
        - `contour` : Contour plot
        - `contourf` : Filled contour (image-like) plot (canonical)
        - `surface` : Surface plot
        - `waterfall` : Waterfall plot

    **kwargs
        Additional matplotlib / plotting keyword arguments.

    Other Parameters
    ----------------
    ax : Axe, optional
        Axe where to plot. If not specified, create a new one.
    clear : bool, optional, default: True
        If false, hold the current figure and ax until a new plot is performed.
    color or c : color, optional, default: auto
        color of the line.
    colorbar : bool, optional, default: True
        Show colorbar (2D plots only).
    commands : str,
        matplotlib commands to be executed.
    data_only : bool, optional, default: False
        Only the plot is done. No addition of axes or label specifications.
    dpi : int, optional
        the number of pixel per inches.
    figsize : tuple, optional, default is (3.4, 1.7)
        figure size.
    fontsize : int, optional
        The font size in pixels, default is 10 (or read from preferences).
    imag : bool, optional, default: False
        Show imaginary component for complex data. By default the real component is
        displayed.
    linestyle or ls : str, optional, default: auto
        line style definition.
    linewidth or lw : float, optional, default: auto
        line width.
    marker, m: str, optional, default: auto
        marker type for scatter plot. If marker != "" then the scatter type of plot is chosen automatically.
    markeredgecolor or mec: color, optional
    markeredgewidth or mew: float, optional
    markerfacecolor or mfc: color, optional
    markersize or ms: float, optional
    markevery: None or int
    modellinestyle or modls : str
        line style of the model.
    offset : float
        offset of the model individual lines.
    output : str,
        name of the file to save the figure.
    plot_model : Bool,
        plot model data if available.
    plottitle: bool, optional, default: False
        Use the name of the dataset as title. Works only if title is not defined
    projections : bool, optional, default: False
        Show projections on the axes (2D plots only).
    reverse : bool or None [optional, default=None/False
        In principle, coordinates run from left to right,
        except for wavenumbers
        (e.g., FTIR spectra) or ppm (e.g., NMR), that spectrochempy
        will try to guess. But if reverse is set, then this is the
        setting which will be taken into account.
    show_complex : bool, optional, default: False
        Show both real and imaginary component for complex data.
        By default only the real component is displayed.
    show_mask: bool, optional
        Should we display the mask using colored area.
    show_z : bool, optional, default: True
        should we show the vertical axis.
    show_zero : bool, optional
        show the zero basis.
    style : str, optional, default: scp.preferences.style (scpy)
        Matplotlib stylesheet (use available_style to get a list of available
        styles for plotting.
    title : str
        Title of the plot (or subplot) axe.
    transposed : bool, optional, default: False
        Transpose the data before plotting (2D plots only).
    twinx : :class:~matplotlib.axes.Axes instance, optional, default: None
        If this is not None, then a twin axes will be created with a
        common x dimension.
    uselabel_x: bool, optional
        use x coordinate label as x tick labels
    vshift : float, optional
        vertically shift the line from its baseline.
    xlim : tuple, optional
        limit on the horizontal axis.
    xlabel : str, optional
        label on the horizontal axis.
    x_reverse : bool, optional, default: False
        reverse the x axis. Equivalent to reverse.
    ylabel or zlabel : str, optional
        label on the vertical axis.
    ylim or zlim : tuple, optional
        limit on the vertical axis.
    y_reverse : bool, optional, default: False
        reverse the y axis (2D plot only).
    equal_aspect : bool, optional, default: False
        If True and X/Y units are compatible, enforce metric scaling
        (square pixels). This ensures 1 unit in X equals 1 unit in Y.

    Returns
    -------
    Matplolib Axes or None
        The matplotlib axes containing the plot if successful, None otherwise.

    See Also
    --------
    plot
    plot_2D
    plot_lines
    plot_contour
    plot_surface
    plot_waterfall
    plot_image
    """
    return plot_2D(dataset, method="contourf", **kwargs)


# ======================================================================================
# nddataset plot2D functions - Legacy wrappers (deprecated)
# ======================================================================================


def plot_stack(dataset, **kwargs):
    """
    Plot a 2D dataset as a stack plot.

    Parameters
    ----------
    dataset : :class:~spectrochempy.ddataset.nddataset.NDDataset
        Source of data to plot.
    method : str, optional, default: auto-detected from data dimensionality
        Name of plotting method to use. If None, method is chosen based on data
        dimensionality.

        2D plotting methods:

        - `stack` : Stacked plot
        - `map` : Contour plot
        - `image` : Image plot
        - `surface` : Surface plot
        - `waterfall` : Waterfall plot

    **kwargs
        Additional matplotlib / plotting keyword arguments.

    Other Parameters
    ----------------
    ax : Axe, optional
        Axe where to plot. If not specified, create a new one.
    clear : bool, optional, default: True
        If false, hold the current figure and ax until a new plot is performed.
    color or c : color, optional, default: auto
        color of the line.
    colorbar : bool, optional, default: True
        Show colorbar (2D plots only).
    commands : str,
        matplotlib commands to be executed.
    data_only : bool, optional, default: False
        Only the plot is done. No addition of axes or label specifications.
    dpi : int, optional
        the number of pixel per inches.
    figsize : tuple, optional, default is (3.4, 1.7)
        figure size.
    fontsize : int, optional
        The font size in pixels, default is 10 (or read from preferences).
    imag : bool, optional, default: False
        Show imaginary component for complex data. By default the real component is
        displayed.
    linestyle or ls : str, optional, default: auto
        line style definition.
    linewidth or lw : float, optional, default: auto
        line width.
    marker, m: str, optional, default: auto
        marker type for scatter plot. If marker != "" then the scatter type of plot is chosen automatically.
    markeredgecolor or mec: color, optional
    markeredgewidth or mew: float, optional
    markerfacecolor or mfc: color, optional
    markersize or ms: float, optional
    markevery: None or int
    modellinestyle or modls : str
        line style of the model.
    offset : float
        offset of the model individual lines.
    output : str,
        name of the file to save the figure.
    palette : str or list, optional, default: None
        Color palette for stack plot. If None, auto-detect based on dataset.
        If "continuous": use continuous colormap (viridis).
        If "categorical": use matplotlib default color cycle.
        If colormap name: use that colormap.
        If list/tuple of colors: use as explicit categorical colors.
        Auto-detection uses continuous colormap only when y coordinate is numeric,
        strictly monotonic, and number of spectra > 6.
    plot_model : Bool,
        plot model data if available.
    plottitle: bool, optional, default: False
        Use the name of the dataset as title. Works only if title is not defined
    projections : bool, optional, default: False
        Show projections on the axes (2D plots only).
    reverse : bool or None [optional, default=None/False
        In principle, coordinates run from left to right,
        except for wavenumbers
        (e.g., FTIR spectra) or ppm (e.g., NMR), that spectrochempy
        will try to guess. But if reverse is set, then this is the
        setting which will be taken into account.
    show_complex : bool, optional, default: False
        Show both real and imaginary component for complex data.
        By default only the real component is displayed.
    show_mask: bool, optional
        Should we display the mask using colored area.
    show_z : bool, optional, default: True
        should we show the vertical axis.
    show_zero : bool, optional
        show the zero basis.
    style : str, optional, default: scp.preferences.style (scpy)
        Matplotlib stylesheet (use available_style to get a list of available
        styles for plotting.
    title : str
        Title of the plot (or subplot) axe.
    transposed : bool, optional, default: False
        Transpose the data before plotting (2D plots only).
    twinx : :class:~matplotlib.axes.Axes instance, optional, default: None
        If this is not None, then a twin axes will be created with a
        common x dimension.
    uselabel_x: bool, optional
        use x coordinate label as x tick labels
    vshift : float, optional
        vertically shift the line from its baseline.
    xlim : tuple, optional
        limit on the horizontal axis.
    xlabel : str, optional
        label on the horizontal axis.
    x_reverse : bool, optional, default: False
        reverse the x axis. Equivalent to reverse.
    ylabel or zlabel : str, optional
        label on the vertical axis.
    ylim or zlim : tuple, optional
        limit on the vertical axis.
    y_reverse : bool, optional, default: False
        reverse the y axis (2D plot only).

    Returns
    -------
    Matplolib Axes or None
        The matplotlib axes containing the plot if successful, None otherwise.

    See Also
    --------
    plot
    plot_2D
    plot_map
    plot_image
    plot_surface
    plot_waterfall
    """
    return plot_2D(dataset, method="stack", **kwargs)


def plot_map(dataset, **kwargs):
    """
    Plot a 2D dataset as a contoured map.

    Parameters
    ----------
    dataset : :class:~spectrochempy.ddataset.nddataset.NDDataset
        Source of data to plot.
    method : str, optional, default: auto-detected from data dimensionality
        Name of plotting method to use. If None, method is chosen based on data
        dimensionality.

        1D plotting methods:

        - `pen` : Solid line plot
        - `bar` : Bar graph
        - `scatter` : Scatter plot
        - `scatter+pen` : Scatter plot with solid line

        2D plotting methods:

        - `stack` : Stacked plot
        - `map` : Contour plot
        - `image` : Image plot
        - `surface` : Surface plot
        - `waterfall` : Waterfall plot

    **kwargs
        Additional matplotlib / plotting keyword arguments.

    Other Parameters
    ----------------
    ax : Axe, optional
        Axe where to plot. If not specified, create a new one.
    clear : bool, optional, default: True
        If false, hold the current figure and ax until a new plot is performed.
    color or c : color, optional, default: auto
        color of the line.
    colorbar : bool, optional, default: True
        Show colorbar (2D plots only).
    commands : str,
        matplotlib commands to be executed.
    data_only : bool, optional, default: False
        Only the plot is done. No addition of axes or label specifications.
    dpi : int, optional
        the number of pixel per inches.
    figsize : tuple, optional, default is (3.4, 1.7)
        figure size.
    fontsize : int, optional
        The font size in pixels, default is 10 (or read from preferences).
    imag : bool, optional, default: False
        Show imaginary component for complex data. By default the real component is
        displayed.
    linestyle or ls : str, optional, default: auto
        line style definition.
    linewidth or lw : float, optional, default: auto
        line width.
    marker, m: str, optional, default: auto
        marker type for scatter plot. If marker != "" then the scatter type of plot is chosen automatically.
    markeredgecolor or mec: color, optional
    markeredgewidth or mew: float, optional
    markerfacecolor or mfc: color, optional
    markersize or ms: float, optional
    markevery: None or int
    modellinestyle or modls : str
        line style of the model.
    offset : float
        offset of the model individual lines.
    output : str,
        name of the file to save the figure.
    plot_model : Bool,
        plot model data if available.
    plottitle: bool, optional, default: False
        Use the name of the dataset as title. Works only if title is not defined
    projections : bool, optional, default: False
        Show projections on the axes (2D plots only).
    reverse : bool or None [optional, default=None/False
        In principle, coordinates run from left to right,
        except for wavenumbers
        (e.g., FTIR spectra) or ppm (e.g., NMR), that spectrochempy
        will try to guess. But if reverse is set, then this is the
        setting which will be taken into account.
    show_complex : bool, optional, default: False
        Show both real and imaginary component for complex data.
        By default only the real component is displayed.
    show_mask: bool, optional
        Should we display the mask using colored area.
    show_z : bool, optional, default: True
        should we show the vertical axis.
    show_zero : bool, optional
        show the zero basis.
    style : str, optional, default: scp.preferences.style (scpy)
        Matplotlib stylesheet (use available_style to get a list of available
        styles for plotting.
    title : str
        Title of the plot (or subplot) axe.
    transposed : bool, optional, default: False
        Transpose the data before plotting (2D plots only).
    twinx : :class:~matplotlib.axes.Axes instance, optional, default: None
        If this is not None, then a twin axes will be created with a
        common x dimension.
    uselabel_x: bool, optional
        use x coordinate label as x tick labels
    vshift : float, optional
        vertically shift the line from its baseline.
    xlim : tuple, optional
        limit on the horizontal axis.
    xlabel : str, optional
        label on the horizontal axis.
    x_reverse : bool, optional, default: False
        reverse the x axis. Equivalent to reverse.
    ylabel or zlabel : str, optional
        label on the vertical axis.
    ylim or zlim : tuple, optional
        limit on the vertical axis.
    y_reverse : bool, optional, default: False
        reverse the y axis (2D plot only).

    Returns
    -------
    Matplolib Axes or None
        The matplotlib axes containing the plot if successful, None otherwise.

    See Also
    --------
    plot
    plot_2D
    plot_stack
    plot_image
    plot_surface
    plot_waterfall
    """
    return plot_2D(dataset, method="map", **kwargs)


def plot_image(dataset, **kwargs):
    """
    Plot a 2D dataset as an image plot.

    Parameters
    ----------
    dataset : :class:~spectrochempy.ddataset.nddataset.NDDataset
        Source of data to plot.
    method : str, optional, default: auto-detected from data dimensionality
        Name of plotting method to use. If None, method is chosen based on data
        dimensionality.

        1D plotting methods:

        - `pen` : Solid line plot
        - `bar` : Bar graph
        - `scatter` : Scatter plot
        - `scatter+pen` : Scatter plot with solid line

        2D plotting methods:

        - `stack` : Stacked plot
        - `map` : Contour plot
        - `image` : Image plot
        - `surface` : Surface plot
        - `waterfall` : Waterfall plot

    **kwargs
        Additional matplotlib / plotting keyword arguments.

    Other Parameters
    ----------------
    ax : Axe, optional
        Axe where to plot. If not specified, create a new one.
    clear : bool, optional, default: True
        If false, hold the current figure and ax until a new plot is performed.
    color or c : color, optional, default: auto
        color of the line.
    colorbar : bool, optional, default: True
        Show colorbar (2D plots only).
    commands : str,
        matplotlib commands to be executed.
    data_only : bool, optional, default: False
        Only the plot is done. No addition of axes or label specifications.
    dpi : int, optional
        the number of pixel per inches.
    figsize : tuple, optional, default is (3.4, 1.7)
        figure size.
    fontsize : int, optional
        The font size in pixels, default is 10 (or read from preferences).
    imag : bool, optional, default: False
        Show imaginary component for complex data. By default the real component is
        displayed.
    linestyle or ls : str, optional, default: auto
        line style definition.
    linewidth or lw : float, optional, default: auto
        line width.
    marker, m: str, optional, default: auto
        marker type for scatter plot. If marker != "" then the scatter type of plot is chosen automatically.
    markeredgecolor or mec: color, optional
    markeredgewidth or mew: float, optional
    markerfacecolor or mfc: color, optional
    markersize or ms: float, optional
    markevery: None or int
    modellinestyle or modls : str
        line style of the model.
    offset : float
        offset of the model individual lines.
    output : str,
        name of the file to save the figure.
    plot_model : Bool,
        plot model data if available.
    plottitle: bool, optional, default: False
        Use the name of the dataset as title. Works only if title is not defined
    projections : bool, optional, default: False
        Show projections on the axes (2D plots only).
    reverse : bool or None [optional, default=None/False
        In principle, coordinates run from left to right,
        except for wavenumbers
        (e.g., FTIR spectra) or ppm (e.g., NMR), that spectrochempy
        will try to guess. But if reverse is set, then this is the
        setting which will be taken into account.
    show_complex : bool, optional, default: False
        Show both real and imaginary component for complex data.
        By default only the real component is displayed.
    show_mask: bool, optional
        Should we display the mask using colored area.
    show_z : bool, optional, default: True
        should we show the vertical axis.
    show_zero : bool, optional
        show the zero basis.
    style : str, optional, default: scp.preferences.style (scpy)
        Matplotlib stylesheet (use available_style to get a list of available
        styles for plotting.
    title : str
        Title of the plot (or subplot) axe.
    transposed : bool, optional, default: False
        Transpose the data before plotting (2D plots only).
    twinx : :class:~matplotlib.axes.Axes instance, optional, default: None
        If this is not None, then a twin axes will be created with a
        common x dimension.
    uselabel_x: bool, optional
        use x coordinate label as x tick labels
    vshift : float, optional
        vertically shift the line from its baseline.
    xlim : tuple, optional
        limit on the horizontal axis.
    xlabel : str, optional
        label on the horizontal axis.
    x_reverse : bool, optional, default: False
        reverse the x axis. Equivalent to reverse.
    ylabel or zlabel : str, optional
        label on the vertical axis.
    ylim or zlim : tuple, optional
        limit on the vertical axis.
    y_reverse : bool, optional, default: False
        reverse the y axis (2D plot only).

    Returns
    -------
    Matplolib Axes or None
        The matplotlib axes containing the plot if successful, None otherwise.

    See Also
    --------
    plot
    plot_2D
    plot_stack
    plot_map
    plot_surface
    plot_waterfall
    """
    return plot_2D(dataset, method="image", **kwargs)


def plot_2D(dataset, method=None, **kwargs):
    """
    Plot of 2D array.

    Parameters
    ----------
    dataset : :class:~spectrochempy.ddataset.nddataset.NDDataset
        Source of data to plot.
    method : str, optional, default: auto-detected from data dimensionality
        Name of plotting method to use. If None, method is chosen based on data
        dimensionality.

        1D plotting methods:

        - `pen` : Solid line plot
        - `bar` : Bar graph
        - `scatter` : Scatter plot
        - `scatter+pen` : Scatter plot with solid line

        2D plotting methods:

        - `stack` : Stacked plot
        - `map` : Contour plot
        - `image` : Image plot
        - `surface` : Surface plot
        - `waterfall` : Waterfall plot

    **kwargs
        Additional matplotlib / plotting keyword arguments.

    Other Parameters
    ----------------
    ax : Axe, optional
        Axe where to plot. If not specified, create a new one.
    clear : bool, optional, default: True
        If false, hold the current figure and ax until a new plot is performed.
    color or c : color, optional, default: auto
        color of the line.
    colorbar : bool, optional, default: True
        Show colorbar (2D plots only).
    commands : str,
        matplotlib commands to be executed.
    data_only : bool, optional, default: False
        Only the plot is done. No addition of axes or label specifications.
    dpi : int, optional
        the number of pixel per inches.
    figsize : tuple, optional, default is (3.4, 1.7)
        figure size.
    fontsize : int, optional
        The font size in pixels, default is 10 (or read from preferences).
    imag : bool, optional, default: False
        Show imaginary component for complex data. By default the real component is
        displayed.
    linestyle or ls : str, optional, default: auto
        line style definition.
    linewidth or lw : float, optional, default: auto
        line width.
    marker, m: str, optional, default: auto
        marker type for scatter plot. If marker != "" then the scatter type of plot is chosen automatically.
    markeredgecolor or mec: color, optional
    markeredgewidth or mew: float, optional
    markerfacecolor or mfc: color, optional
    markersize or ms: float, optional
    markevery: None or int
    modellinestyle or modls : str
        line style of the model.
    offset : float
        offset of the model individual lines.
    output : str,
        name of the file to save the figure.
    plot_model : Bool,
        plot model data if available.
    plottitle: bool, optional, default: False
        Use the name of the dataset as title. Works only if title is not defined
    projections : bool, optional, default: False
        Show projections on the axes (2D plots only).
    reverse : bool or None [optional, default=None/False
        In principle, coordinates run from left to right,
        except for wavenumbers
        (e.g., FTIR spectra) or ppm (e.g., NMR), that spectrochempy
        will try to guess. But if reverse is set, then this is the
        setting which will be taken into account.
    show_complex : bool, optional, default: False
        Show both real and imaginary component for complex data.
        By default only the real component is displayed.
    show_mask: bool, optional
        Should we display the mask using colored area.
    show_z : bool, optional, default: True
        should we show the vertical axis.
    show_zero : bool, optional
        show the zero basis.
    style : str, optional, default: scp.preferences.style (scpy)
        Matplotlib stylesheet (use available_style to get a list of available
        styles for plotting.
    title : str
        Title of the plot (or subplot) axe.
    transposed : bool, optional, default: False
        Transpose the data before plotting (2D plots only).
    twinx : :class:~matplotlib.axes.Axes instance, optional, default: None
        If this is not None, then a twin axes will be created with a
        common x dimension.
    uselabel_x: bool, optional
        use x coordinate label as x tick labels
    vshift : float, optional
        vertically shift the line from its baseline.
    xlim : tuple, optional
        limit on the horizontal axis.
    xlabel : str, optional
        label on the horizontal axis.
    x_reverse : bool, optional, default: False
        reverse the x axis. Equivalent to reverse.
    ylabel or zlabel : str, optional
        label on the vertical axis.
    ylim or zlim : tuple, optional
        limit on the vertical axis.
    y_reverse : bool, optional, default: False
        reverse the y axis (2D plot only).

    Returns
    -------
    Matplolib Axes or None
        The matplotlib axes containing the plot if successful, None otherwise.

    See Also
    --------
    plot
    plot_stack
    plot_map
    plot_image
    plot_surface
    plot_waterfall

    """

    from spectrochempy.plotting.plot_setup import lazy_ensure_mpl_config

    lazy_ensure_mpl_config()

    import matplotlib as mpl
    import matplotlib.backend_bases  # noqa: F401
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import ScalarFormatter

    # Get preferences
    # ----------------------------------------------------------------------------------
    prefs = preferences

    # Resolve plotting style(s) locally (no global rcParams / no prefs.style mutation)
    style = kwargs.pop("style", None)
    if style is None:
        style = getattr(prefs, "style", None) or ["scpy"]
    if isinstance(style, str):
        style = [style]

    # Resolve rc_overrides for font settings (will be applied in the style context)
    rc_overrides = prefs.set_latex_font(prefs.font.family)
    if rc_overrides is None:
        rc_overrides = {}

    # Apply unified style context for entire plotting operation
    # This includes: figure creation, colormap resolution, line style resolution, and rendering
    # Style context must wrap everything for matplotlib prop_cycle and image.cmap to work correctly
    with mpl.rc_context(rc_overrides):
        with plt.style.context(style):
            # Redirections ?
            # ----------------------------------------------------------------------------------
            # should we redirect the plotting to another method
            if dataset._squeeze_ndim < 2:
                return dataset.plot_1D(**kwargs)

            # if plotly execute plotly routine not this one
            if kwargs.get("use_plotly", prefs.use_plotly):
                return dataset.plotly(**kwargs)

            # do not display colorbar if it's not a surface plot
            # except if we have asked to d so

            # often we do need to plot only data when plotting on top of a previous plot
            data_only = kwargs.get("data_only", False)

            # Extract colorbar kwarg - default to False, do not pass to rendering
            colorbar = kwargs.pop("colorbar", False)

            # Get the data to plot
            # ---------------------------------------------------------------
            # if we want to plot the transposed dataset
            transposed = kwargs.get("transposed", False)
            if transposed:
                new = dataset.copy().T  # transpose dataset
                nameadd = ".T"
            else:
                new = dataset  # .copy()
                nameadd = ""
            new = new.squeeze()

            if kwargs.get("y_reverse", False):
                new = new[::-1]

            # Figure setup
            # ------------------------------------------------------------------------
            _figure_result = new._figure_setup(
                ndim=2,
                method=method,
                style=style,
                **kwargs,
            )
            # Handle both old (method string) and new (method, fig, ndaxes) return values
            if isinstance(_figure_result, tuple):
                method, fig, ndaxes = _figure_result
            else:
                # Fallback for any code that still uses old behavior
                method = _figure_result
                ndaxes = {}

            # Use ndaxes from figure_setup if available, otherwise try to get from figure
            if "main" in ndaxes:
                ax = ndaxes["main"]
            else:
                # Try to get axes from the figure
                if fig.get_axes():
                    ax = fig.get_axes()[0]
                    ax.name = "main"
                else:
                    # This shouldn't happen if _figure_setup worked correctly
                    ax = fig.add_subplot(1, 1, 1)
                    ax.name = "main"

            ax.name += nameadd

            # Resolve line/marker styles using centralized L1 function
            style_kwargs = resolve_line_style(
                dataset=new,
                geometry="line",
                kwargs=kwargs,
                prefs=prefs,
            )

            lw = style_kwargs["linewidth"]
            if lw == "auto":
                lw = prefs.lines_linewidth
            ls = style_kwargs["linestyle"]
            if ls == "auto":
                ls = prefs.lines_linestyle
            marker = style_kwargs["marker"]
            if marker == "auto":
                marker = None
            markersize = style_kwargs["markersize"]

            alpha = kwargs.get("calpha", prefs.contour_alpha)

            number_x_labels = kwargs.get("n_x_labels", prefs.number_of_x_labels)
            number_y_labels = kwargs.get("n_y_labels", prefs.number_of_y_labels)
            number_z_labels = kwargs.get("n_z_labels", prefs.number_of_z_labels)

            if method in ["waterfall"]:
                nxl = number_x_labels * 2
                nyl = number_z_labels * 2
            elif method in ["stack", "lines"]:
                nxl = number_x_labels
                nyl = number_z_labels
            else:
                nxl = number_x_labels
                nyl = number_y_labels

            ax.xaxis.set_major_locator(MaxNLocator(nbins=nxl))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=nyl))
            if method not in ["surface"]:
                ax.xaxis.set_ticks_position("bottom")
                ax.yaxis.set_ticks_position("left")

            # the next lines are to avoid multipliers in axis scale
            formatter = ScalarFormatter(useOffset=False)
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)

            # ------------------------------------------------------------------------
            # Set axis
            # ------------------------------------------------------------------------
            # set the abscissa axis
            # the actual dimension name is the last in the new.dims list
            dimx = new.dims[-1]
            x = getattr(new, dimx)
            if x is not None and x._implements("CoordSet"):
                # if several coords, take the default ones:
                x = x.default
            xsize = new.shape[-1]
            show_x_points = False
            if x is not None and hasattr(x, "show_datapoints"):
                show_x_points = x.show_datapoints
            if show_x_points:
                # remove data and units for display
                x = Coord.arange(xsize)

            discrete_data = False

            if x is not None and (not x.is_empty or x.is_labeled):
                xdata = x.data
                if not np.any(xdata) and x.is_labeled:
                    discrete_data = True
                    # take into account the fact that sometimes axis have just labels
                    xdata = range(1, len(x.labels) + 1)
            else:
                xdata = range(xsize)

            xl = [xdata[0], xdata[-1]]
            xl.sort()

            if xsize < number_x_labels + 1:
                # extend the axis so that the labels are not too close to the limits
                inc = abs(xdata[1] - xdata[0]) * 0.5
                xl = [xl[0] - inc, xl[1] + inc]

            if data_only:
                xl = ax.get_xlim()

            xlim = list(kwargs.get("xlim", xl))
            xlim.sort()
            xlim[-1] = min(xlim[-1], xl[-1])
            xlim[0] = max(xlim[0], xl[0])

            x_reverse_explicit = kwargs.get("x_reverse")
            reverse_explicit = kwargs.get("reverse")

            # Check if user explicitly passed x_reverse or reverse
            # If explicitly passed, use that value; otherwise use x.reversed
            if x_reverse_explicit is not None:
                x_reverse = x_reverse_explicit
            elif reverse_explicit is not None:
                x_reverse = reverse_explicit
            else:
                x_reverse = x.reversed if x else False

            # For x_reverse=True, don't reverse xlim - just invert axis after setting limits
            # For x_reverse=False (from x.reversed=False), reverse xlim and let matplotlib auto-handle
            # For x_reverse=False when x.reversed=True, we should NOT invert (user wants normal order)
            if x_reverse:
                # User wants inverted axis - don't reverse xlim, invert after
                pass
            elif x is not None and x.reversed:
                # x.reversed is True but user didn't explicitly override - auto-handle
                xlim.reverse()

            ax.set_xlim(xlim)

            # If x_reverse is True, explicitly invert the axis
            if x_reverse:
                ax.invert_xaxis()

            xscale = kwargs.get("xscale", "linear")
            ax.set_xscale(xscale)  # , nonpositive='mask')

            # set the ordinates axis
            # ------------------------------------------------------------------------
            # the actual dimension name is the second in the new.dims list
            dimy = new.dims[-2]
            y = getattr(new, dimy)
            if y is not None and y._implements("CoordSet"):
                # if several coords, take the default ones:
                y = y.default
            ysize = new.shape[-2]

            show_y_points = False
            if y is not None and hasattr(y, "show_datapoints"):
                show_y_points = y.show_datapoints
            if show_y_points:
                # remove data and units for display
                y = Coord.arange(ysize)

            if y is not None and (not y.is_empty or y.is_labeled):
                ydata = y.data

                if not np.any(ydata) and y.is_labeled:
                    ydata = range(1, len(y.labels) + 1)
            else:
                ydata = range(ysize)

            yl = [ydata[0], ydata[-1]]
            yl.sort()

            if ysize < number_y_labels + 1:
                # extend the axis so that the labels are not too close to the limits
                inc = abs(ydata[1] - ydata[0]) * 0.5
                yl = [yl[0] - inc, yl[1] + inc]

            if data_only:
                yl = ax.get_ylim()

            ylim = list(kwargs.get("ylim", yl))
            ylim.sort()
            ylim[-1] = min(ylim[-1], yl[-1])
            ylim[0] = max(ylim[0], yl[0])

            yscale = kwargs.get("yscale", "linear")
            ax.set_yscale(yscale)

            # z intensity (by default we plot real component of the data)
            # ------------------------------------------------------------------------
            if not kwargs.get("imag", False):
                zdata = new.real.masked_data
            else:
                zdata = (
                    new.imag.masked_data
                )  # TODO: quaternion case (3 imag.components)

            zlim = kwargs.get("zlim", (np.ma.min(zdata), np.ma.max(zdata)))

            if method in ["stack", "lines", "waterfall"]:
                # the z axis info
                # ---------------
                # zl = (np.min(np.ma.min(ys)), np.max(np.ma.max(ys)))
                amp = 0  # np.ma.ptp(zdata) / 50.
                zl = (np.min(np.ma.min(zdata) - amp), np.max(np.ma.max(zdata)) + amp)
                zlim = list(kwargs.get("zlim", zl))
                zlim.sort()
                z_reverse = kwargs.get("z_reverse", False)
                if z_reverse:
                    zlim.reverse()

                # set the limits
                # ---------------
                if yscale == "log" and min(zlim) <= 0:
                    # set the limits wrt smallest and largest strictly positive values
                    mi = np.amin(np.abs(zdata))
                    ma = np.amax(np.abs(zdata))
                    ax.set_ylim(
                        10 ** (int(np.log10(mi + (ma - mi) * 0.001)) - 1),
                        10 ** (int(np.log10(ma)) + 1),
                    )
                else:
                    ax.set_ylim(zlim)

            else:
                # the y axis info
                # ----------------
                if data_only:
                    ylim = ax.get_ylim()

                ylim = list(kwargs.get("ylim", ylim))
                ylim.sort()
                y_reverse_explicit = kwargs.get("y_reverse", False)
                y_reverse_auto = y.reversed if y else False

                # For explicit y_reverse, don't reverse ylim - just invert axis after setting limits
                # For auto y.reversed, reverse ylim and let matplotlib auto-handle the inversion
                if y_reverse_explicit:
                    ylim_original = list(ylim)  # Keep original order
                elif y_reverse_auto:
                    ylim.reverse()

                # set the limits
                # ----------------
                ax.set_ylim(ylim)

                # For explicit y_reverse kwarg, explicitly invert the axis
                # For auto y.reversed, matplotlib already handles inversion when ylim[0] > ylim[1]
                if y_reverse_explicit:
                    ax.invert_yaxis()

            # ------------------------------------------------------------------------
            # plot the dataset
            # ------------------------------------------------------------------------
            grid = kwargs.get("grid", prefs.axes_grid)
            ax.grid(grid)

            # Resolve colormap and normalization using unified helper
            # Priority: norm > cmap > cmap_mode > auto-detection > contrast_safe
            cmap = kwargs.get("cmap")
            cmap_mode = kwargs.get("cmap_mode", "auto")
            center = kwargs.get("center")
            norm = kwargs.get("norm")
            vmin = kwargs.get("vmin")
            vmax = kwargs.get("vmax")
            contrast_safe = kwargs.get("contrast_safe", True)
            min_contrast = kwargs.get("min_contrast", 1.5)
            diverging_margin = kwargs.get("diverging_margin", 0.05)

            # Get background color from axes
            try:
                facecolor = ax.get_facecolor()
                if facecolor and len(facecolor) > 0:
                    bg_rgba = facecolor[0]
                    background_rgb = (bg_rgba[0], bg_rgba[1], bg_rgba[2])
                else:
                    background_rgb = (1.0, 1.0, 1.0)
            except Exception:
                background_rgb = (1.0, 1.0, 1.0)

            # For image, map, surface methods, use the unified colormap resolution
            if method in ["map", "image", "contour", "contourf", "surface"]:
                geometry = method  # "map" -> "contour", "image" -> "image", "surface" -> "surface"
                # Style context is already active from the outer wrapper - no need for nested context
                cmap, norm = resolve_2d_colormap(
                    zdata,
                    cmap=cmap,
                    cmap_mode=cmap_mode,
                    center=center,
                    norm=norm,
                    vmin=vmin,
                    vmax=vmax,
                    contrast_safe=contrast_safe,
                    min_contrast=min_contrast,
                    background_rgb=background_rgb,
                    geometry=geometry,
                    diverging_margin=diverging_margin,
                    prefs=prefs,
                )
            else:
                # For non-image methods, use simple normalization
                if norm is None:
                    zmin, zmax = zlim
                    norm = Normalize(vmin=zmin, vmax=zmax)
                if cmap is None:
                    cmap = prefs.colormap
                elif isinstance(cmap, str):
                    cmap = plt.get_cmap(cmap)

            # Initialize mappable tracker for colorbar
            mappable = None

            if method in ["surface"]:
                # Ensure 3D axes
                if not hasattr(ax, "plot_surface"):
                    fig = ax.figure
                    fig.delaxes(ax)
                    ax = fig.add_subplot(111, projection="3d")
                    ndaxes["main"] = ax

                    # Re-apply axis inversions after recreating 3D axes
                    # (the inversions were set on the old 2D axes which was deleted)
                    x_reverse_explicit = kwargs.get("x_reverse")
                    reverse_explicit = kwargs.get("reverse")
                    if x_reverse_explicit is not None:
                        x_reverse = x_reverse_explicit
                    elif reverse_explicit is not None:
                        x_reverse = reverse_explicit
                    else:
                        x_reverse = x.reversed if x else False

                    if x_reverse:
                        ax.invert_xaxis()

                    y_reverse_explicit = kwargs.get("y_reverse", False)
                    if y_reverse_explicit:
                        ax.invert_yaxis()

                X, Y = np.meshgrid(xdata, ydata)
                Z = zdata.copy()

                # masker data not taken into account in surface plot
                Z[dataset.mask] = np.nan

                # Plot the surface.  #TODO : improve this (or remove it)

                antialiased = kwargs.get("antialiased", prefs.antialiased)
                rcount = kwargs.get("rcount", prefs.rcount)
                ccount = kwargs.get("ccount", prefs.ccount)
                ax.set_facecolor("w")
                mappable = ax.plot_surface(
                    X,
                    Y,
                    Z,
                    cmap=cmap,
                    linewidth=lw,
                    antialiased=antialiased,
                    rcount=rcount,
                    ccount=ccount,
                    edgecolor="k",
                    norm=norm,
                )

            if method in ["waterfall"]:
                _plot_waterfall(
                    ax, new, xdata, ydata, zdata, prefs, xlim, ylim, zlim, **kwargs
                )

            elif method in ["image", "contourf"]:
                if discrete_data:
                    method = "map"

                else:
                    kwargs["nlevels"] = 500
                    if not hasattr(new, "clevels") or new.clevels is None:
                        new.clevels = _get_clevels(zdata, prefs, **kwargs)
                    mappable = ax.contourf(
                        xdata, ydata, zdata, new.clevels, alpha=alpha
                    )
                    mappable.set_cmap(cmap)
                    mappable.set_norm(norm)

                    # For colorbar, create a ScalarMappable with the resolved norm
                    colorbar_mappable = ScalarMappable(norm=norm, cmap=cmap)
                    colorbar_mappable.set_array(zdata)

            elif method in ["map", "contour"]:
                if discrete_data:
                    _colormap = plt.get_cmap(cmap)
                    scalarMap = ScalarMappable(norm=norm, cmap=_colormap)
                    mappable = scalarMap

                    # marker = kwargs.get('marker', kwargs.get('m', None))
                    markersize = kwargs.get("markersize", kwargs.get("ms", 5.0))
                    # markevery = kwargs.get('markevery', kwargs.get('me', 1))

                    for i in ydata:
                        for j in xdata:
                            (li,) = ax.plot(
                                j, i, lw=lw, marker="o", markersize=markersize
                            )
                            li.set_color(scalarMap.to_rgba(zdata[i - 1, j - 1]))

                else:
                    # contour plot
                    # -------------
                    if not hasattr(new, "clevels") or new.clevels is None:
                        new.clevels = _get_clevels(zdata, prefs, **kwargs)

                    mappable = ax.contour(
                        xdata, ydata, zdata, new.clevels, linewidths=lw, alpha=alpha
                    )
                    mappable.set_cmap(cmap)
                    mappable.set_norm(norm)

                    # For continuous colorbar, create a ScalarMappable with the resolved norm
                    colorbar_mappable = ScalarMappable(norm=norm, cmap=cmap)
                    colorbar_mappable.set_array(zdata)

            elif method in ["stack", "lines"]:
                # stack plot
                # ----------
                # now plot the collection of lines
                # map colors - always use y-coordinate range (not data intensity)
                vmin, vmax = ylim
                norm = Normalize(vmin=vmin, vmax=vmax)

                # Initialize is_categorical (default True - no colorbar unless proven continuous)
                is_categorical = True

                # Get palette parameter for auto-detection
                palette = kwargs.pop("palette", None)

                # Check if user explicitly provided color or cmap (backward compatibility)
                explicit_color = kwargs.get("color")
                explicit_cmap = kwargs.get("colormap") or kwargs.get("cmap")

                if explicit_color is not None:
                    # User explicitly passed color - could be a single color or a list
                    if isinstance(explicit_color, (list)):
                        # Check if it's a list of colors or a single color wrapped in a list
                        # A list of colors would have color-like elements (strings, tuples)
                        if len(explicit_color) > 0:
                            first_elem = explicit_color[0]
                            if isinstance(first_elem, (list, tuple)) and len(
                                first_elem
                            ) in (
                                3,
                                4,
                            ):
                                # It's a list of color tuples - use as-is
                                colors = list(explicit_color)
                            elif len(explicit_color) == 1:
                                # Single color wrapped in list - treat as single color
                                colors = [explicit_color[0]]
                            else:
                                # Multiple color strings - use as-is for cycling
                                colors = list(explicit_color)
                        else:
                            colors = [explicit_color]
                    elif isinstance(explicit_color, tuple):
                        # Tuple - could be RGB/RGBA or just a single item - treat as single color
                        colors = [explicit_color]
                    else:
                        # Single color value (string, number, etc.)
                        colors = [explicit_color]
                    scalarMap = None
                    is_categorical = True  # Explicit colors are categorical
                elif explicit_cmap is not None:
                    # User explicitly passed colormap - use continuous mapping
                    _colormap = plt.get_cmap(
                        explicit_cmap if explicit_cmap != "Undefined" else "viridis"
                    )
                    scalarMap = ScalarMappable(norm=norm, cmap=_colormap)
                    colors = None
                    mappable = scalarMap
                    is_categorical = False
                else:
                    # Use auto-detection helper (L1)
                    contrast_safe = kwargs.get("contrast_safe", True)
                    min_contrast = kwargs.get("min_contrast", 1.5)
                    colors, is_categorical, mappable = resolve_stack_colors(
                        new,
                        palette=palette,
                        n=ysize,
                        geometry="line",
                        contrast_safe=contrast_safe,
                        min_contrast=min_contrast,
                        prefs=prefs,
                    )
                    if is_categorical:
                        # Categorical: use colors directly, no mappable
                        scalarMap = None
                    else:
                        # Continuous: use the mappable from L1, build ScalarMappable with colors
                        from matplotlib.colors import LinearSegmentedColormap

                        _colormap = LinearSegmentedColormap.from_list(
                            "stack_cmap", colors, N=256
                        )
                        scalarMap = ScalarMappable(norm=norm, cmap=_colormap)
                        # Update mappable with the new colormap
                        mappable = ScalarMappable(norm=norm, cmap=_colormap)
                        mappable.set_array(np.arange(ysize))
                        colors = None

                # we display the line in the reverse order, so that the last
                # are behind the first.

                clear = kwargs.get("clear", True)
                existing_lines = []
                if not clear and not transposed:
                    existing_lines.extend(ax.lines)  # keep the old lines

                # Pre-compute zorders (policy: later lines have lower zorder)
                n_lines = zdata.shape[0]
                zorders = [n_lines + 1 - i for i in range(n_lines)]

                # Pre-compute colors for each line
                fmt = kwargs.get("label_fmt", "{:.5f}")
                line_colors = []
                line_labels = []
                has_colors = colors is not None and len(colors) > 0
                for i in range(n_lines):
                    if scalarMap is not None:
                        line_colors.append(scalarMap.to_rgba(ydata[i]))
                    elif has_colors:
                        line_colors.append(colors[i % len(colors)])
                    else:
                        line_colors.append(None)
                    line_labels.append(fmt.format(ydata[i]))

                # Use render_lines for drawing
                new_lines = render_lines(
                    ax,
                    xdata,
                    zdata,
                    colors=line_colors,
                    linestyles=[ls] * n_lines,
                    linewidths=[lw] * n_lines,
                    markers=[marker] * n_lines,
                    markersizes=[markersize] * n_lines,
                    zorders=zorders,
                    labels=line_labels,
                    reverse=True,
                    picker=True,
                )

                # store the full set of lines (render_lines already added them to ax)
                new._ax_lines = existing_lines + new_lines

            if data_only or method in ["waterfall"]:
                # if data only (we will not set axes and labels
                # it was probably done already in a previous plot
                new._plot_resume(dataset, **kwargs)
                return ax

            # display a title
            # ------------------------------------------------------------------------
            title = kwargs.get("title")
            if title:
                ax.set_title(title)
            elif kwargs.get("plottitle", False):
                ax.set_title(new.name)

            # ----------------------------------------------------------------------------------
            # labels
            # ----------------------------------------------------------------------------------
            # x label
            xlabel = kwargs.get("xlabel")
            if show_x_points:
                xlabel = "data points"
            if not xlabel:
                xlabel = make_label(x, new.dims[-1])
            ax.set_xlabel(xlabel)

            uselabelx = kwargs.get("uselabel_x", False)
            if (
                x
                and x.is_labeled
                and (uselabelx or not np.any(x.data))
                and len(x.labels) < number_x_labels + 1
            ):
                # TODO refine this to use different orders of labels
                ax.set_xticks(xdata)
                ax.set_xticklabels(x.labels)

            # y label
            # ------------------------------------------------------------------------
            ylabel = kwargs.get("ylabel")
            if show_y_points:
                ylabel = "data points"
            if not ylabel:
                if method in ["stack", "lines"]:
                    ylabel = make_label(new, "values")

                else:
                    ylabel = make_label(y, new.dims[-2])
                    # y tick labels
                    uselabely = kwargs.get("uselabel_y", False)
                    if (
                        y
                        and y.is_labeled
                        and (uselabely or not np.any(y.data))
                        and len(y.labels) < number_y_labels
                    ):
                        # TODO refine this to use different orders of labels
                        ax.set_yticks(ydata)
                        ax.set_yticklabels(y.labels)

            # z label
            # ------------------------------------------------------------------------
            zlabel = kwargs.get("zlabel")
            if not zlabel:
                if method in ["stack", "lines"]:
                    zlabel = make_label(y, new.dims[-2])
                elif method in ["surface"]:
                    zlabel = make_label(new, "values")
                    ax.set_zlabel(zlabel)
                else:
                    zlabel = make_label(new, "z")

            # do we display the ordinate axis?
            if kwargs.get("show_y", True):
                ax.set_ylabel(ylabel)
            else:
                ax.set_yticks([])

            # Create colorbar if requested (L3 responsibility)
            # For map/image/surface: always show colorbar if requested
            # For stack: only show if continuous colormap (not categorical)
            if colorbar and mappable is not None:
                fig = ax.figure
                if method == "stack":
                    # Stack: only show colorbar for continuous, not categorical
                    # In stack mode, color represents y-coordinate (not intensity)
                    if not is_categorical:
                        if not hasattr(ax, "_scp_colorbar"):
                            # Semantic label: color represents y coordinate
                            y_coord_label = make_label(y, new.dims[-2])
                            cb = fig.colorbar(
                                mappable,
                                ax=ax,
                                location="right",
                                pad=0.02,
                                fraction=0.05,
                                aspect=30,
                            )
                            cb.set_label(y_coord_label)
                            _apply_colorbar_tick_policy(cb, norm, vmin=vmin, vmax=vmax)
                            ax._scp_colorbar = cb
                elif method in ["map", "image", "contour", "contourf", "surface"]:
                    # 2D plots: show colorbar (intensity)
                    if not hasattr(ax, "_scp_colorbar"):
                        # Use continuous colorbar mappable if available (for contour plots)
                        cb_mappable = locals().get("colorbar_mappable", mappable)
                        cb = fig.colorbar(
                            cb_mappable,
                            ax=ax,
                            location="right",
                            pad=0.02,
                            fraction=0.05,
                            aspect=30,
                        )
                        cb.set_label(zlabel)
                        _apply_colorbar_tick_policy(cb, norm, vmin=vmin, vmax=vmax)
                        ax._scp_colorbar = cb

            # do we display the zero line
            if kwargs.get("show_zero", False):
                ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)

            # Handle equal_aspect for 2D plots (contour, contourf, image, map)
            # Hierarchy: explicit kwarg > preference > default (False)
            if "equal_aspect" in kwargs:
                equal_aspect = kwargs["equal_aspect"]
            else:
                equal_aspect = prefs.image_equal_aspect

            if equal_aspect and method in ["contour", "contourf", "image", "map"]:
                if _can_enforce_equal_aspect(new):
                    ax.set_aspect("equal", adjustable="box")
                else:
                    import warnings as _warn_module

                    _warn_module.warn(
                        "equal_aspect=True ignored: X and Y units are incompatible or missing.",
                        UserWarning,
                        stacklevel=2,
                    )

            # Handle equal_aspect for 3D surface plots
            if method == "surface":
                _handle_3d_aspect(ax, new, **kwargs)

            new._plot_resume(dataset, **kwargs)

            return ax

        # ======================================================================================
        # Waterfall
        # ======================================================================================


def _plot_waterfall(ax, new, xdata, ydata, zdata, prefs, xlim, ylim, zlim, **kwargs):
    from spectrochempy.plotting.plot_setup import lazy_ensure_mpl_config

    lazy_ensure_mpl_config()

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    degazim = kwargs.get("azim", prefs.axes3d_azim)
    degelev = kwargs.get("elev", prefs.axes3d_elev)

    azim = np.deg2rad(degazim)
    elev = np.deg2rad(degelev)

    # transformation function Axes coordinates to Data coordinates
    def transA2D(x_, y_):
        return ax.transData.inverted().transform(ax.transAxes.transform((x_, y_)))

    # expansion in Axes coordinates
    xe, ze = np.sin(azim), np.sin(elev)

    incx, incz = transA2D(1 + xe, 1 + ze) - np.array((xlim[-1], zlim[-1]))

    def fx(y_):
        return (y_ - ydata[0]) * incx / (ydata[-1] - ydata[0])

    def fz(y_):
        return (y_ - ydata[0]) * incz / (ydata[-1] - ydata[0])

    zs = incz * 0.05
    base = zdata.min() - zs

    for i, row in enumerate(zdata):
        y = ydata[i]
        x = xdata + fx(y)
        z = row + fz(y)  # row.masked_data[0]
        ma = z.max()
        z2 = base + fz(y)
        line = mpl.lines.Line2D(x, z, color="k")
        line.set_label(f"{ydata[i]}")
        line.set_zorder(row.size + 1 - i)
        poly = plt.fill_between(
            x,
            z,
            z2,
            alpha=1,
            facecolors="w",
            edgecolors="0.85" if 0 < i < ydata.size - 1 else "k",
        )
        poly.set_zorder(row.size + 1 - i)
        with suppress(ValueError):
            ax.add_collection(poly)

        ax.add_line(line)

    (x0, y0), (x1, _) = transA2D(0, 0), transA2D(1 + xe + 0.15, 1 + ze)
    ax.set_xlim((x0, x1))
    ax.set_ylim((y0 - zs - 0.05, ma * 1.1))

    ax.set_facecolor("w")
    ax.vlines(
        x=xdata[-1] + incx,
        ymin=zdata.min() - zs + incz,
        ymax=ax.get_ylim()[-1],
        color="k",
    )
    ax.vlines(
        x=xdata[0] + incx,
        ymin=zdata.min() - zs + incz,
        ymax=ax.get_ylim()[-1],
        color="k",
    )
    ax.vlines(
        x=xdata[0],
        ymin=y0 - zs,
        ymax=ax.get_ylim()[-1] - incz,
        color="k",
        zorder=5000,
    )
    ax.vlines(
        x=xdata[0],
        ymin=y0 - zs,
        ymax=ax.get_ylim()[-1] - incz,
        color="k",
        zorder=5000,
    )

    x = [xdata[0], xdata[0] + incx, xdata[-1] + incx]
    z = [ax.get_ylim()[-1] - incz, ax.get_ylim()[-1], ax.get_ylim()[-1]]
    x2 = [xdata[0], xdata[-1], xdata[-1] + incx]
    z2 = [y0 - zs, y0 - zs, y0 - zs + incz]
    poly = plt.fill_between(x, z, z2, alpha=1, facecolors=".95", edgecolors="w")
    with suppress(ValueError):
        ax.add_collection(poly)
    poly = plt.fill_between(x2, z, z2, alpha=1, facecolors=".95", edgecolors="w")
    with suppress(ValueError):
        ax.add_collection(poly)
    line = mpl.lines.Line2D(x, np.array(z), color="k", zorder=50000)
    ax.add_line(line)
    line = mpl.lines.Line2D(x2, np.array(z2), color="k", zorder=50000)
    ax.add_line(line)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # xticks (xaxis)
    ticks = ax.get_xticks()
    newticks = []
    xt = sorted(xlim)
    for tick in ticks:
        if xt[0] <= tick <= xt[1]:
            newticks.append(tick)
    ax.set_xticks(newticks)

    # yticks (zaxis)
    ticks = ax.get_yticks()
    newticks = []
    zt = [y0, ax.get_ylim()[-1] - incz]
    for tick in ticks:
        if zt[0] <= tick <= zt[1]:
            newticks.append(tick)
    _ = ax.set_yticks(newticks)

    # make yaxis
    def ctx(x_):
        return (
            ax.transData.inverted().transform((x_, 0))
            - ax.transData.inverted().transform((0, 0))
        )[0]

    yt = list(np.linspace(ylim[0], ylim[-1], 5))
    for y in yt:
        xmin = xdata[-1] + fx(y)
        xmax = xdata[-1] + fx(y) + ctx(3.5)
        pos = y0 - zs + fz(y)
        ax.hlines(pos, xmin, xmax, zorder=50000)
        lab = ax.text(xmax + ctx(8), pos, f"{y:.0f}", va="center")

    # display a title
    # ------------------------------------------------------------------------
    title = kwargs.get("title")
    if title:
        ax.set_title(title)

    # ------------------------------------------------------------------------
    # labels
    # ------------------------------------------------------------------------
    # x label
    xlabel = kwargs.get("xlabel")
    if not xlabel:
        xlabel = make_label(new.x, "x")
    ax.set_xlabel(xlabel, x=(ax.bbox._bbox.x0 + ax.bbox._bbox.x1) / 2 - xe)

    # y label
    # ------------------------------------------------------------------------
    ylabel = kwargs.get("ylabel")
    if not ylabel:
        ylabel = make_label(new.y, "y")
    ym = (ylim[0] + ylim[1]) / 2
    x = xdata[-1] + fx(ym)
    z = y0 - zs + fz(ym)
    offset = prefs.font.size * (len(lab._text)) + 30
    iz = ax.transData.transform((0, incz + z))[1] - ax.transData.transform((0, z))[1]
    ix = ax.transData.transform((incx + x, 0))[0] - ax.transData.transform((x, 0))[0]
    angle = np.rad2deg(np.arctan(iz / ix))
    ax.annotate(
        ylabel,
        (x, z),
        xytext=(offset, 0),
        xycoords="data",
        textcoords="offset pixels",
        ha="center",
        va="center",
        rotation=angle,
    )

    # z label
    # ------------------------------------------------------------------------
    zlabel = kwargs.get("zlabel")
    if not zlabel:
        zlabel = make_label(new, "value")

    # do we display the z axis?
    if kwargs.get("show_z", True):
        ax.set_ylabel(zlabel, y=(ax.bbox._bbox.y0 + 1 - ze) / 2)
    else:
        ax.set_yticks([])


# ======================================================================================
# get clevels
# ======================================================================================
def _get_clevels(data, prefs, **kwargs):
    # Utility function to determine contours levels

    # contours
    maximum = data.max()
    minimum = data.min()

    nlevels = kwargs.get("nlevels", kwargs.get("nc", prefs.number_of_contours))
    start = kwargs.get("start", prefs.contour_start) * maximum
    negative = kwargs.get("negative", True)
    if negative < 0:
        negative = True

    c = np.arange(nlevels)
    cl = np.log(c + 1.0)
    clevel = cl * (maximum - start) / cl.max() + start

    # Only create negative levels if data actually contains negative values
    if negative and minimum < 0:
        clevelneg = -clevel
        clevelc = sorted(np.concatenate((clevelneg, clevel)))
    else:
        clevelc = clevel

    return clevelc
