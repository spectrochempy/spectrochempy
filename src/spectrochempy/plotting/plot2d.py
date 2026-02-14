# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Plotters."""

__all__ = [
    "plot_2D",
    "plot_map",
    "plot_stack",
    "plot_image",
]
__dataset_methods__ = __all__


from contextlib import suppress
from copy import copy as cpy

import numpy as np
import matplotlib

from spectrochempy.application.preferences import preferences
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.utils.mplutils import make_label

# ======================================================================================
# nddataset plot2D functions
# ======================================================================================


def _resolve_stack_colors(dataset, palette=None, n=None):
    """
    Resolve colors for stack plot with auto-detection.

    Parameters
    ----------
    dataset : NDDataset
        The 2D dataset being plotted.
    palette : str or list, optional
        If None: auto-detect based on dataset characteristics.
        If str:
            - "continuous": force continuous colormap (viridis)
            - "categorical": force categorical colors
            - colormap name: use that colormap
        If list: use as explicit categorical colors.
    n : int, optional
        Number of colors needed. If None, derived from dataset shape.

    Returns
    -------
    tuple
        (colors, is_categorical) where colors is a list of color values
        and is_categorical indicates whether to use discrete color cycling.
    """
    import matplotlib.pyplot as plt

    if n is None:
        n = dataset.shape[-2]

    # If palette is explicitly provided, use it
    if palette is not None:
        if palette == "continuous":
            cmap = plt.get_cmap("viridis")
            return cmap(np.linspace(0, 1, n)), False
        elif palette == "categorical":
            default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            colors = list(default_cycle)
            # Cycle if needed
            while len(colors) < n:
                colors.extend(default_cycle)
            return colors[:n], True
        elif isinstance(palette, (list, tuple)):
            # Explicit list of colors
            colors = list(palette)
            while len(colors) < n:
                colors.extend(palette)
            return colors[:n], True
        else:
            # Assume it's a colormap name
            cmap = plt.get_cmap(palette)
            return cmap(np.linspace(0, 1, n)), False

    # Auto-detection: determine if continuous or categorical
    # Use continuous colormap ONLY if ALL conditions are met:
    # - dataset has a y coordinate
    # - y is numeric
    # - y values are strictly monotonic

    use_continuous = False

    # Get y coordinate (second to last dimension)
    if dataset._squeeze_ndim >= 2:
        dimy = dataset.dims[-2]
        y = getattr(dataset, dimy)
        if y is not None:
            # Check if y is numeric
            if hasattr(y, "data") and y.data is not None:
                y_data = np.asarray(y.data)
                if np.issubdtype(y_data.dtype, np.number):
                    # Check if strictly monotonic
                    if len(y_data) > 1:
                        diffs = np.diff(y_data)
                        if np.all(diffs > 0) or np.all(diffs < 0):
                            # Strictly monotonic
                            use_continuous = True

    if use_continuous:
        cmap = plt.get_cmap("viridis")
        return cmap(np.linspace(0, 1, n)), False
    else:
        # Use categorical color cycle
        default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = list(default_cycle)
        # Cycle if needed
        while len(colors) < n:
            colors.extend(default_cycle)
        return colors[:n], True


def _relative_luminance(rgb):
    """
    Compute relative luminance of an sRGB color.

    Implements WCAG 2.1 relative luminance formula:
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B

    where R, G, B are linearized (gamma-corrected) sRGB values.

    Parameters
    ----------
    rgb : tuple
        RGB color tuple (r, g, b) with values in [0, 1].

    Returns
    -------
    float
        Relative luminance in [0, 1].
    """
    r, g, b = rgb[:3]

    def linearize(c):
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    r_lin = linearize(r)
    g_lin = linearize(g)
    b_lin = linearize(b)

    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def _contrast_ratio(rgb1, rgb2):
    """
    Compute WCAG 2.1 contrast ratio between two colors.

    Contrast ratio = (L1 + 0.05) / (L2 + 0.05)
    where L1 is the lighter color's luminance.

    Parameters
    ----------
    rgb1, rgb2 : tuple
        RGB color tuples (r, g, b) with values in [0, 1].

    Returns
    -------
    float
        Contrast ratio (>= 1).
    """
    l1 = _relative_luminance(rgb1)
    l2 = _relative_luminance(rgb2)

    lighter = max(l1, l2)
    darker = min(l1, l2)

    return (lighter + 0.05) / (darker + 0.05)


def _ensure_min_contrast(cmap, background_rgb, min_contrast=2.5, samples=256):
    """
    Trim colormap to ensure minimum contrast with background.

    Samples the colormap and finds the smallest interval [start, end]
    where all colors meet the minimum contrast threshold with the background.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        The colormap to potentially trim.
    background_rgb : tuple
        RGB tuple (r, g, b) of background color, values in [0, 1].
    min_contrast : float, optional, default: 2.5
        Minimum WCAG contrast ratio required (default 2.5 = AA large text).
    samples : int, optional, default: 256
        Number of samples to check across the colormap.

    Returns
    -------
    matplotlib.colors.Colormap
        Original colormap if no valid interval exists, otherwise
        a truncated colormap using LinearSegmentedColormap.from_list.
    """
    import matplotlib.colors as mcolors

    x = np.linspace(0, 1, samples)
    colors = cmap(x)

    contrasts = np.array([_contrast_ratio(c[:3], background_rgb) for c in colors])

    valid_mask = contrasts >= min_contrast

    if not np.any(valid_mask):
        return cmap

    valid_indices = np.where(valid_mask)[0]
    start_idx = valid_indices[0]
    end_idx = valid_indices[-1]

    if start_idx == 0 and end_idx == samples - 1:
        return cmap

    start_frac = x[start_idx]
    end_frac = x[end_idx]

    truncated_colors = cmap(np.linspace(start_frac, end_frac, 256))
    truncated_cmap = mcolors.LinearSegmentedColormap.from_list(
        "truncated", truncated_colors, N=256
    )

    return truncated_cmap


def _resolve_2d_colormap(
    data,
    cmap=None,
    cmap_mode="auto",
    center=None,
    norm=None,
    contrast_safe=True,
    min_contrast=2.5,
    background_rgb=None,
):
    """
    Resolve colormap and normalization for 2D plots with auto-detection.

    This function implements a consistent colormap API for all 2D plotting functions
    (plot_image, plot_contour, plot_surface). It provides:
    - Auto-detection of sequential vs diverging colormaps based on data
    - Centered normalization for bipolar data
    - Explicit overrides via parameters
    - Optional contrast safety to ensure visibility on background

    Priority order (strict):
        1. norm (if explicitly provided) -> use as-is, no auto-detection
        2. cmap (if explicitly provided) -> use as-is
        3. cmap_mode (if not "auto") -> force sequential or diverging
        4. auto-detection -> sequential if all positive, diverging if bipolar
        5. contrast_safe -> trim colormap ends for visibility (if enabled)

    Parameters
    ----------
    data : array-like
        The 2D data array to be plotted.
    cmap : str, optional
        Colormap name. If None, will be determined based on cmap_mode.
    cmap_mode : str, optional, default: "auto"
        "auto", "sequential", or "diverging".
        - "auto": auto-detect based on data range
        - "sequential": force sequential colormap (viridis)
        - "diverging": force diverging colormap (RdBu_r)
    center : numeric or str, optional
        Center value for diverging colormaps.
        - None: use 0 for diverging mode
        - "auto": auto-detect center (0 if data crosses zero, else midpoint)
        - numeric: use this value as center
    norm : matplotlib.colors.Normalize, optional
        Explicit normalization. If provided, overrides all other normalization.
    contrast_safe : bool, optional, default: True
        If True, trim colormap ends to ensure minimum contrast with background.
    min_contrast : float, optional, default: 2.5
        Minimum WCAG contrast ratio (2.5 = AA large text, 3.0 = AA normal text).
    background_rgb : tuple, optional
        RGB tuple (r, g, b) of background color. If None, defaults to white (1,1,1).

    Returns
    -------
    tuple
        (cmap, norm) resolved values for plotting.

    Scientific Rationale
    -------------------
    - Sequential colormaps (e.g., viridis): appropriate for unipolar data
      where magnitude represents intensity (e.g., temperature, concentration).
    - Diverging colormaps (e.g., RdBu_r): appropriate for bipolar data
      where sign matters (e.g., deviation from reference, positive/negative signals).
    - Centered normalization: ensures meaningful zero reference for
      comparing positive and negative deviations.
    - Contrast safety: ensures colors are visible against the plot background.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, Normalize

    if background_rgb is None:
        background_rgb = (1.0, 1.0, 1.0)

    norm_explicitly_provided = norm is not None

    # Priority 1: if norm is explicitly provided, use it as-is
    if norm is not None:
        # norm is explicitly provided - use it, cmap still applies
        if cmap is None:
            cmap = plt.get_cmap("viridis")
        elif isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        # Apply contrast safety only if norm is not explicitly provided by user
        if contrast_safe and not norm_explicitly_provided:
            cmap = _ensure_min_contrast(cmap, background_rgb, min_contrast)

        return cmap, norm

    # Get data range for auto-detection
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    # Determine mode: sequential vs diverging
    use_diverging = False

    if cmap_mode == "diverging":
        use_diverging = True
    elif cmap_mode == "sequential":
        use_diverging = False
    else:  # cmap_mode == "auto"
        # Auto-detect: diverging if data crosses zero
        use_diverging = vmin < 0 < vmax

    # Resolve colormap
    if cmap is not None:
        # User explicitly provided cmap - use it
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
    elif use_diverging:
        # Diverging default: RdBu_r (red-blue, reversed)
        cmap = plt.get_cmap("RdBu_r")
    else:
        # Sequential default: viridis
        cmap = plt.get_cmap("viridis")

    # Resolve normalization
    if use_diverging:
        # Diverging mode: use TwoSlopeNorm for centered colormap
        if center is None:
            center_value = 0
        elif center == "auto":
            center_value = 0 if vmin < 0 < vmax else (vmin + vmax) / 2
        else:
            center_value = center

        # Ensure vmin < vcenter < vmax for TwoSlopeNorm
        # If data doesn't span across center, adjust
        if center_value <= vmin:
            center_value = vmin + (vmax - vmin) / 2
        if center_value >= vmax:
            center_value = vmin + (vmax - vmin) / 2

        norm = TwoSlopeNorm(vmin=vmin, vcenter=center_value, vmax=vmax)
    else:
        # Sequential mode: simple linear normalization
        norm = Normalize(vmin=vmin, vmax=vmax)

    # Apply contrast safety if enabled and norm was auto-generated
    if contrast_safe and not norm_explicitly_provided:
        cmap = _ensure_min_contrast(cmap, background_rgb, min_contrast)

    return cmap, norm


def plot_stack(dataset, **kwargs):
    """
    Plot a 2D dataset as a stack plot.

    Parameters
    ----------
    dataset : :class:~spectrochempy.ddataset.nddataset.NDDataset
        Source of data to plot.
    method : str, optional, default: preference.method_1D or preference.method_2D
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
    method : str, optional, default: preference.method_1D or preference.method_2D
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
    method : str, optional, default: preference.method_1D or preference.method_2D
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
    method : str, optional, default: preference.method_1D or preference.method_2D
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
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
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

    # style handled at figure creation (get_figure)
    rc_overrides = prefs.set_latex_font(prefs.font.family)
    if rc_overrides:
        with matplotlib.rc_context(rc_overrides):
            pass  # rc_overrides applied for subsequent plotting

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

    # Other properties that can be passed as arguments
    # ------------------------------------------------------------------------
    lw = kwargs.get("linewidth", kwargs.get("lw", prefs.lines_linewidth))
    ls = kwargs.get("linestyle", kwargs.get("ls", prefs.lines_linestyle))
    marker = kwargs.get("marker", kwargs.get("m"))
    markersize = kwargs.get("markersize", kwargs.get("ms", prefs.lines_markersize))

    alpha = kwargs.get("calpha", prefs.contour_alpha)

    number_x_labels = prefs.number_of_x_labels
    number_y_labels = prefs.number_of_y_labels
    number_z_labels = prefs.number_of_z_labels

    if method in ["waterfall"]:
        nxl = number_x_labels * 2
        nyl = number_z_labels * 2
    elif method in ["stack"]:
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

    if kwargs.get("x_reverse", kwargs.get("reverse", x.reversed if x else False)):
        xlim.reverse()

    ax.set_xlim(xlim)

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
        zdata = new.imag.masked_data  # TODO: quaternion case (3 imag.components)

    zlim = kwargs.get("zlim", (np.ma.min(zdata), np.ma.max(zdata)))

    if method in ["stack", "waterfall"]:
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
        y_reverse = kwargs.get("y_reverse", y.reversed if y else False)
        if y_reverse:
            ylim.reverse()

        # set the limits
        # ----------------
        ax.set_ylim(ylim)

    # ------------------------------------------------------------------------
    # plot the dataset
    # ------------------------------------------------------------------------
    ax.grid(prefs.axes_grid)

    # Resolve colormap and normalization using unified helper
    # Priority: norm > cmap > cmap_mode > auto-detection > contrast_safe
    cmap = kwargs.get("cmap")
    cmap_mode = kwargs.get("cmap_mode", "auto")
    center = kwargs.get("center")
    norm = kwargs.get("norm")
    contrast_safe = kwargs.get("contrast_safe", True)
    min_contrast = kwargs.get("min_contrast", 2.5)

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
    if method in ["map", "image", "surface"]:
        cmap, norm = _resolve_2d_colormap(
            zdata,
            cmap=cmap,
            cmap_mode=cmap_mode,
            center=center,
            norm=norm,
            contrast_safe=contrast_safe,
            min_contrast=min_contrast,
            background_rgb=background_rgb,
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

    if method in ["surface"]:
        # Ensure 3D axes
        if not hasattr(ax, "plot_surface"):
            fig = ax.figure
            fig.delaxes(ax)
            ax = fig.add_subplot(111, projection="3d")
            ndaxes["main"] = ax

        X, Y = np.meshgrid(xdata, ydata)
        Z = zdata.copy()

        # masker data not taken into account in surface plot
        Z[dataset.mask] = np.nan

        # Plot the surface.  #TODO : improve this (or remove it)

        antialiased = kwargs.get("antialiased", prefs.antialiased)
        rcount = kwargs.get("rcount", prefs.rcount)
        ccount = kwargs.get("ccount", prefs.ccount)
        ax.set_facecolor("w")
        ax.plot_surface(
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
        _plot_waterfall(ax, new, xdata, ydata, zdata, prefs, xlim, ylim, zlim, **kwargs)

    elif method in ["image"]:
        # Support both new cmap parameter and legacy image_cmap parameter
        # cmap is already resolved by _resolve_2d_colormap above
        # For image method, also check for legacy image_cmap parameter
        if cmap is None:
            cmap = kwargs.get("image_cmap", prefs.image_cmap)
        if discrete_data:
            method = "map"

        else:
            kwargs["nlevels"] = 500
            if not hasattr(new, "clevels") or new.clevels is None:
                new.clevels = _get_clevels(zdata, prefs, **kwargs)
            c = ax.contourf(xdata, ydata, zdata, new.clevels, alpha=alpha)
            c.set_cmap(cmap)
            c.set_norm(norm)

    elif method in ["map"]:
        if discrete_data:
            _colormap = plt.get_cmap(cmap)
            scalarMap = ScalarMappable(norm=norm, cmap=_colormap)

            # marker = kwargs.get('marker', kwargs.get('m', None))
            markersize = kwargs.get("markersize", kwargs.get("ms", 5.0))
            # markevery = kwargs.get('markevery', kwargs.get('me', 1))

            for i in ydata:
                for j in xdata:
                    (li,) = ax.plot(j, i, lw=lw, marker="o", markersize=markersize)
                    li.set_color(scalarMap.to_rgba(zdata[i - 1, j - 1]))

        else:
            # contour plot
            # -------------
            if not hasattr(new, "clevels") or new.clevels is None:
                new.clevels = _get_clevels(zdata, prefs, **kwargs)

            c = ax.contour(xdata, ydata, zdata, new.clevels, linewidths=lw, alpha=alpha)
            c.set_cmap(cmap)
            c.set_norm(norm)

    elif method in ["stack"]:
        # stack plot
        # ----------
        # now plot the collection of lines
        # map colors - always use y-coordinate range (not data intensity)
        vmin, vmax = ylim
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Get palette parameter for auto-detection
        palette = kwargs.pop("palette", None)

        # Check if user explicitly provided color or cmap (backward compatibility)
        explicit_color = kwargs.get("color")
        explicit_cmap = kwargs.get("colormap") or kwargs.get("cmap")

        if explicit_color is not None:
            # User explicitly passed color - use single color
            colors = [explicit_color]
            scalarMap = None
        elif explicit_cmap is not None:
            # User explicitly passed colormap - use continuous mapping
            _colormap = plt.get_cmap(
                explicit_cmap if explicit_cmap != "Undefined" else "viridis"
            )
            scalarMap = ScalarMappable(norm=norm, cmap=_colormap)
            colors = None
        else:
            # Use auto-detection helper
            colors, is_categorical = _resolve_stack_colors(
                new, palette=palette, n=ysize
            )
            if is_categorical:
                scalarMap = None
            else:
                # Continuous - create scalarMap
                _colormap = plt.get_cmap("viridis")
                scalarMap = ScalarMappable(norm=norm, cmap=_colormap)
                colors = None

        # we display the line in the reverse order, so that the last
        # are behind the first.

        clear = kwargs.get("clear", True)
        lines = []
        if not clear and not transposed:
            lines.extend(ax.lines)  # keep the old lines

        line0 = Line2D(
            xdata,
            zdata[0],
            lw=lw,
            ls=ls,
            marker=marker,
            markersize=markersize,
            picker=True,
        )

        for i in range(zdata.shape[0]):
            li = cpy(line0)
            li.set_ydata(zdata[i])
            lines.append(li)
            if scalarMap is not None:
                li.set_color(scalarMap.to_rgba(ydata[i]))
            else:
                li.set_color(colors[i % len(colors)])

            fmt = kwargs.get("label_fmt", "{:.5f}")
            li.set_label(fmt.format(ydata[i]))
            li.set_zorder(zdata.shape[0] + 1 - i)

        # store the full set of lines
        new._ax_lines = lines[:]

        # but display only a subset of them in order to accelerate the drawing
        maxlines = kwargs.get("maxlines", prefs.max_lines_in_stack)
        setpy = max(len(new._ax_lines) // maxlines, 1)

        for line in new._ax_lines[::setpy]:
            ax.add_line(line)

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
        if method in ["stack"]:
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
        if method in ["stack"]:
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

    _axcb = None
    if "colorbar" in ndaxes:  # noqa: SIM102
        if "surface" not in method:
            axec = ndaxes["colorbar"]
            axec.name += nameadd
            _axcb = mpl.colorbar.ColorbarBase(
                axec,
                cmap=plt.get_cmap(cmap),
                norm=norm,
            )
            _axcb.set_label(zlabel)
    #        else:
    #            new._fig.colorbar(surf, shrink=0.5, aspect=10)

    # do we display the zero line
    if kwargs.get("show_zero", False):
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)

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

    degazim = kwargs.get("azim", 10)
    degelev = kwargs.get("elev", 30)

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

    # minimum = -maximum

    nlevels = kwargs.get("nlevels", kwargs.get("nc", prefs.number_of_contours))
    start = kwargs.get("start", prefs.contour_start) * maximum
    negative = kwargs.get("negative", True)
    if negative < 0:
        negative = True

    c = np.arange(nlevels)
    cl = np.log(c + 1.0)
    clevel = cl * (maximum - start) / cl.max() + start
    clevelneg = -clevel
    clevelc = clevel
    if negative:
        clevelc = sorted(np.concatenate((clevel, clevelneg)))

    return clevelc
