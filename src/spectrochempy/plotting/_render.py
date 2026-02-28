## ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
L2 Rendering Primitives Layer.

This module contains pure rendering functions that draw on provided Axes objects.
L2 is stateless, deterministic, and contains NO policy decisions.
All style resolution, zorder computation, stacking logic, and orchestration
remain in the L3 layer (plot1d.py, plot2d.py).
"""

__all__ = [
    "render_lines",
    "render_image",
    "render_contour",
    "render_surface",
]

import numpy as np
from matplotlib.lines import Line2D


def _expand_to_list(value, n, name="value"):
    """Expand a scalar or list to a list of length n."""
    if value is None:
        return [None] * n
    if not isinstance(value, list):
        return [value] * n
    if len(value) < n:
        return value * n
    return value


def render_lines(
    ax,
    x,
    Y,
    *,
    colors=None,
    linestyles=None,
    linewidths=None,
    markers=None,
    markersizes=None,
    markerfacecolors=None,
    markeredgecolors=None,
    alpha=None,
    zorders=None,
    reverse=False,
    labels=None,
    label_fmt="{:.5f}",
    picker=True,
    **kwargs,
):
    """
    Render lines on the given Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    x : array-like
        X coordinates for all lines.
    Y : array-like
        Y data. Can be 1D (single line) or 2D (multiple lines, rows are lines).
    colors : list of colors, optional
        Colors for each line.
    linestyles : list of str, optional
        Line styles.
    linewidths : list of float, optional
        Line widths.
    markers : list of str, optional
        Marker styles.
    markersizes : list of float, optional
        Marker sizes.
    markerfacecolors : list of colors, optional
        Marker face colors.
    markeredgecolors : list of colors, optional
        Marker edge colors.
    alpha : float, optional
        Transparency for all lines.
    zorders : list of int, optional
        Z-order for each line. If None, no explicit zorder is set.
    reverse : bool, optional
        If True, iterate through lines in reverse order.
    labels : list of str, optional
        Labels for each line.
    label_fmt : str, optional
        Format string for labels.
    picker : bool, optional
        Enable picking on lines.
    **kwargs
        Additional parameters passed to Line2D.

    Returns
    -------
    list of matplotlib.lines.Line2D
        The drawn line objects.
    """
    Y = np.atleast_2d(Y)
    n_lines = Y.shape[0]

    labels = _expand_to_list(labels, n_lines, "labels")
    colors = _expand_to_list(colors, n_lines, "colors")
    linestyles = _expand_to_list(linestyles, n_lines, "linestyles")
    linewidths = _expand_to_list(linewidths, n_lines, "linewidths")
    markers = _expand_to_list(markers, n_lines, "markers")
    markersizes = _expand_to_list(markersizes, n_lines, "markersizes")
    markerfacecolors = _expand_to_list(markerfacecolors, n_lines, "markerfacecolors")
    markeredgecolors = _expand_to_list(markeredgecolors, n_lines, "markeredgecolors")
    zorders = _expand_to_list(zorders, n_lines, "zorders")

    lines = []
    indices = list(range(n_lines))
    if reverse:
        indices = list(reversed(indices))

    for idx in indices:
        line = Line2D(
            x,
            Y[idx],
            linestyle=linestyles[idx],
            marker=markers[idx],
            markersize=markersizes[idx],
            markerfacecolor=markerfacecolors[idx],
            markeredgecolor=markeredgecolors[idx],
            linewidth=linewidths[idx],
            color=colors[idx],
            alpha=alpha,
            label=labels[idx] if labels[idx] is not None else label_fmt.format(idx),
            picker=picker,
            **kwargs,
        )

        if zorders[idx] is not None:
            line.set_zorder(zorders[idx])

        lines.append(line)
        ax.add_line(line)

    return lines


def render_image(
    ax, x, y, Z, *, cmap=None, norm=None, alpha=None, levels=None, **kwargs
):
    """
    Render an image using contourf with high number of levels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    x : array-like
        X coordinates.
    y : array-like
        Y coordinates.
    Z : array-like
        The intensity data.
    cmap : matplotlib Colormap, optional
        The colormap.
    norm : matplotlib.colors.Normalize, optional
        The normalization.
    alpha : float, optional
        Transparency.
    levels : int, optional
        Number of contour levels (default 500 for smooth image).
    **kwargs
        Additional parameters passed to ax.contourf.

    Returns
    -------
    matplotlib.contour.QuadContourSet
        The contour set.
    """
    if levels is None:
        levels = 500

    c = ax.contourf(x, y, Z, levels, alpha=alpha, **kwargs)
    if cmap is not None:
        c.set_cmap(cmap)
    if norm is not None:
        c.set_norm(norm)

    return c


def render_contour(
    ax,
    x,
    y,
    Z,
    *,
    levels=None,
    cmap=None,
    norm=None,
    linewidths=None,
    alpha=None,
    filled=False,
    **kwargs,
):
    """
    Render contour lines or filled contours.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    x : array-like
        X coordinates.
    y : array-like
        Y coordinates.
    Z : array-like
        The intensity data.
    levels : array-like, optional
        Contour levels.
    cmap : matplotlib Colormap, optional
        The colormap.
    norm : matplotlib.colors.Normalize, optional
        The normalization.
    linewidths : float or list, optional
        Line widths.
    alpha : float, optional
        Transparency.
    filled : bool, optional
        If True, draw filled contours (contourf). If False, draw lines (contour).
    **kwargs
        Additional parameters passed to ax.contour/contourf.

    Returns
    -------
    matplotlib.contour.QuadContourSet
        The contour set.
    """
    if filled:
        if levels is None:
            c = ax.contourf(x, y, Z, alpha=alpha, **kwargs)
        else:
            c = ax.contourf(x, y, Z, levels, alpha=alpha, **kwargs)
    else:
        if levels is None:
            c = ax.contour(x, y, Z, linewidths=linewidths, alpha=alpha, **kwargs)
        else:
            c = ax.contour(
                x, y, Z, levels, linewidths=linewidths, alpha=alpha, **kwargs
            )

    if cmap is not None:
        c.set_cmap(cmap)
    if norm is not None:
        c.set_norm(norm)

    return c


def render_scatter(
    ax,
    x,
    y,
    *,
    colors=None,
    marker=None,
    markersizes=None,
    alpha=None,
    zorder=None,
    **kwargs,
):
    """
    Pure rendering function for scatter plots.

    This function MUST:
        - Call ax.scatter()
        - Accept already-resolved style values
        - NOT compute any style
        - NOT modify rcParams
        - NOT clear axes
        - NOT set labels
        - NOT call show

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    x : array-like
        X coordinates.
    y : array-like
        Y coordinates.
    colors : color or list of colors, optional
        Already-resolved colors.
    marker : str, optional
        Marker style.
    markersizes : float or list of floats, optional
        Marker sizes.
    alpha : float, optional
        Alpha (transparency).
    zorder : int, optional
        Z-order for layering.
    **kwargs
        Additional parameters passed to ax.scatter.

    Returns
    -------
    matplotlib.collections.PathCollection
        The scatter plot object.
    """
    return ax.scatter(
        x,
        y,
        c=colors,
        marker=marker,
        s=markersizes,
        alpha=alpha,
        zorder=zorder,
        **kwargs,
    )


def render_surface(
    ax,
    X,
    Y,
    Z,
    *,
    cmap=None,
    norm=None,
    linewidth=0,
    antialiased=True,
    rcount=None,
    ccount=None,
    edgecolor="k",
    **kwargs,
):
    """
    Render a 3D surface plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The 3D axes to draw on.
    X : array-like
        X coordinates (meshgrid).
    Y : array-like
        Y coordinates (meshgrid).
    Z : array-like
        The intensity data.
    cmap : matplotlib Colormap, optional
        The colormap.
    norm : matplotlib.colors.Normalize, optional
        The normalization.
    linewidth : float, optional
        Line width.
    antialiased : bool, optional
        Enable antialiasing.
    rcount : int, optional
        Maximum number of rows to sample.
    ccount : int, optional
        Maximum number of columns to sample.
    edgecolor : color, optional
        Edge color.
    **kwargs
        Additional parameters passed to ax.plot_surface.

    Returns
    -------
    matplotlib.collections.Poly3DCollection
        The surface object.
    """
    return ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cmap,
        linewidth=linewidth,
        antialiased=antialiased,
        rcount=rcount,
        ccount=ccount,
        edgecolor=edgecolor,
        norm=norm,
        **kwargs,
    )
