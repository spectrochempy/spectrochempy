## ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Composite plotmerit function.

This module provides a refactored plotmerit implementation that uses
L1 (style resolution) and L2 (rendering primitives) layers.
"""

__all__ = ["plotmerit"]

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from spectrochempy.plotting._render import render_lines
from spectrochempy.plotting._style import resolve_2d_colormap


def _resolve_exp_colors(data, exp_c, n, min_contrast=1.5):
    """Resolve experimental colors based on user input."""
    is_2d = data.ndim == 2

    if exp_c is None:
        if is_2d:
            cmap, _ = resolve_2d_colormap(
                data=data,
                cmap=None,
                cmap_mode="auto",
                center=None,
                norm=None,
                contrast_safe=True,
                min_contrast=min_contrast,
                background_rgb=(1.0, 1.0, 1.0),
                geometry="line",
                diverging_margin=0.05,
            )
            rgba_colors = cmap(np.linspace(0, 1, n))
            colors = [tuple(c[:3]) for c in rgba_colors]
            return colors, False
        return [tuple(mcolors.to_rgba("tab10")[:3])], False
    if isinstance(exp_c, str):
        try:
            cmap = plt.get_cmap(exp_c)
            rgba_colors = cmap(np.linspace(0, 1, n))
            colors = [tuple(c[:3]) for c in rgba_colors]
            return colors, False
        except ValueError:
            return [tuple(mcolors.to_rgba(exp_c)[:3])], False
    elif hasattr(exp_c, "colors") or hasattr(exp_c, "N"):
        cmap = exp_c if hasattr(exp_c, "N") else plt.get_cmap(exp_c)
        rgba_colors = cmap(np.linspace(0, 1, n))
        colors = [tuple(c[:3]) for c in rgba_colors]
        return colors, False
    elif isinstance(exp_c, (list, tuple)):
        colors = []
        for c in exp_c:
            if isinstance(c, str):
                try:
                    cmap = plt.get_cmap(c)
                    rgba_colors = cmap(np.linspace(0, 1, n))
                    colors.extend([tuple(rgba[:3]) for rgba in rgba_colors])
                except ValueError:
                    colors.append(c)
            else:
                rgba = mcolors.to_rgba(c)
                colors.append(tuple(rgba[:3]))
        while len(colors) < n:
            colors.extend(colors[:n])
        return colors[:n], True
    else:
        return [tuple(mcolors.to_rgba(exp_c)[:3])], False


def _resolve_other_colors(colorspec, n, default):
    """Resolve calc_c or resid_c colors."""
    if colorspec is None:
        rgba = mcolors.to_rgba(default)
        return [tuple(rgba[:3])]
    if isinstance(colorspec, str):
        try:
            cmap = plt.get_cmap(colorspec)
            rgba_colors = cmap(np.linspace(0, 1, n))
            return [tuple(c[:3]) for c in rgba_colors]
        except ValueError:
            rgba = mcolors.to_rgba(colorspec)
            return [tuple(rgba[:3])]
    if hasattr(colorspec, "colors") or hasattr(colorspec, "N"):
        cmap = colorspec if hasattr(colorspec, "N") else plt.get_cmap(colorspec)
        rgba_colors = cmap(np.linspace(0, 1, n))
        return [tuple(c[:3]) for c in rgba_colors]
    if isinstance(colorspec, (list, tuple)):
        colors = list(colorspec)
        while len(colors) < n:
            colors.extend(colorspec)
        return colors[:n]
    rgba = mcolors.to_rgba(colorspec)
    return [tuple(rgba[:3])]


def plotmerit(
    analysis_object,
    X=None,
    X_hat=None,
    *,
    ax=None,
    clear=True,
    exp_c=None,
    calc_c=None,
    resid_c=None,
    exp_linestyle="-",
    calc_linestyle="--",
    resid_linestyle="-",
    exp_linewidth=1.2,
    calc_linewidth=1.0,
    resid_linewidth=1.0,
    offset=0,
    min_contrast=1.5,
    title=None,
    show_yaxis=False,
    nb_traces="all",
    **kwargs,
):
    r"""
    Plot the input (X), reconstructed (X_hat) and residuals.

    Parameters
    ----------
    analysis_object : Analysis object
        The analysis object (e.g., PCA, PLSRegression) that has X and inverse_transform.
    X : NDDataset, optional
        Original dataset. If not provided (default), the `X` attribute is used
        and X_hat is computed using `inverse_transform`.
    X_hat : NDDataset, optional
        Inverse transformed dataset. If `X` is provided, `X_hat` must also be provided.
    ax : Axes, optional
        Matplotlib axes to plot on. If None, create new figure.
    clear : bool, optional
        Whether to clear the axes before plotting. Default: True.
    exp_c : color, colormap, or list of colors, optional
        Color(s) for experimental spectra.
    calc_c : color, colormap, or list of colors, optional
        Color(s) for calculated spectra.
    resid_c : color, colormap, or list of colors, optional
        Color(s) for residual spectra.
    exp_linestyle : str, optional
        Line style for experimental spectra. Default: "-".
    calc_linestyle : str, optional
        Line style for calculated spectra. Default: "--".
    resid_linestyle : str, optional
        Line style for residual spectra. Default: "-".
    exp_linewidth : float, optional
        Line width for experimental spectra. Default: 1.2.
    calc_linewidth : float, optional
        Line width for calculated spectra. Default: 1.0.
    resid_linewidth : float, optional
        Line width for residual spectra. Default: 1.0.
    offset : float, optional
        Separation (in percent) between X, X_hat and E. Default: 0.
    min_contrast : float, optional
        Minimum contrast ratio for sequential colormaps. Default: 1.5.
    title : str, optional
        Plot title.
    show_yaxis : bool, optional
        Whether to show y-axis. Default: False.
    nb_traces : int or 'all', optional
        Number of lines to display. Default: 'all'.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Axes
        Matplotlib axes.
    """
    # Figure/axes management
    if ax is None:
        _, ax = plt.subplots()

    if clear:
        ax.cla()

    # Get data
    if X is None:
        X = analysis_object.X
        if X_hat is None:
            X_hat = analysis_object.inverse_transform()
    elif X_hat is None:
        raise ValueError(
            "If X is provided, an externally computed X_hat dataset "
            "must also be provided."
        )

    if X._squeeze_ndim == 1:
        X = X.squeeze()
        X_hat = X_hat.squeeze()

    # Number of traces to keep
    if X.ndim == 2 and nb_traces != "all":
        inc = int(X.shape[0] / nb_traces)
        X = X[::inc]
        X_hat = X_hat[::inc]

    # Compute residual
    res = X - X_hat

    # Offset logic
    ma = max(X.max(), X_hat.max())
    mao = ma * offset / 100
    mad = ma * offset / 100 + ma / 10

    # Determine dimensions
    is_2d = X.ndim == 2
    n_traces = X.shape[0] if is_2d else 1

    # Resolve colors using L1-style helpers
    exp_colors, exp_is_categorical = _resolve_exp_colors(
        X, exp_c, n_traces, min_contrast
    )
    calc_colors = _resolve_other_colors(calc_c, n_traces, "#2a6fbb")
    resid_colors = _resolve_other_colors(resid_c, n_traces, "0.4")

    # Prepare data with offsets
    res_offset = res - mad
    calc_offset = X_hat - mao
    exp_offset = X

    # Get x coordinates
    if is_2d:
        dimx = X.dims[-1]
        x = getattr(X, dimx)
        if x is not None and x._implements("CoordSet"):
            x = x.default
        if x is not None and (not x.is_empty or x.is_labeled):
            xdata = x.data
            if xdata is None or not np.any(xdata):
                xdata = range(X.shape[-1])
        else:
            xdata = range(X.shape[-1])
    else:
        dimx = X.dims[-1]
        x = getattr(X, dimx)
        if x is not None and x._implements("CoordSet"):
            x = x.default
        if x is not None and (not x.is_empty or x.is_labeled):
            xdata = x.data
            if xdata is None or not np.any(xdata):
                xdata = range(X.shape[-1])
        else:
            xdata = range(X.shape[-1])

    # Convert to numpy arrays
    xdata = np.asarray(xdata)
    if is_2d:
        res_data = np.asarray(res_offset.masked_data)
        calc_data = np.asarray(calc_offset.masked_data)
        exp_data = np.asarray(exp_offset.masked_data)
    else:
        res_data = np.atleast_2d(np.asarray(res_offset.masked_data))
        calc_data = np.atleast_2d(np.asarray(calc_offset.masked_data))
        exp_data = np.atleast_2d(np.asarray(exp_offset.masked_data))

    # Prepare style parameters
    if is_2d:
        if exp_is_categorical:
            exp_color_list = exp_colors
            calc_color_list = calc_colors[0] if len(calc_colors) == 1 else calc_colors
            resid_color_list = (
                resid_colors[0] if len(resid_colors) == 1 else resid_colors
            )
        else:
            exp_color_list = exp_colors
            calc_color_list = calc_colors[0] if len(calc_colors) == 1 else calc_colors
            resid_color_list = (
                resid_colors[0] if len(resid_colors) == 1 else resid_colors
            )
    else:
        exp_color_list = [exp_colors[0]]
        calc_color_list = [calc_colors[0]]
        resid_color_list = [resid_colors[0]]

    # Compute zorders: residual (bottom), calculated (middle), experimental (top)
    if is_2d:
        n = res_data.shape[0]
        resid_zorders = [1] * n
        calc_zorders = [2] * n
        exp_zorders = [3] * n
    else:
        resid_zorders = [1]
        calc_zorders = [2]
        exp_zorders = [3]

    # Render residual (bottom layer)
    render_lines(
        ax,
        xdata,
        res_data,
        colors=resid_color_list,
        linestyles=[resid_linestyle] * len(resid_zorders),
        linewidths=[resid_linewidth] * len(resid_zorders),
        zorders=resid_zorders,
        reverse=False,
    )

    # Render calculated (middle layer)
    render_lines(
        ax,
        xdata,
        calc_data,
        colors=calc_color_list,
        linestyles=[calc_linestyle] * len(calc_zorders),
        linewidths=[calc_linewidth] * len(calc_zorders),
        zorders=calc_zorders,
        reverse=False,
    )

    # Render experimental (top layer)
    render_lines(
        ax,
        xdata,
        exp_data,
        colors=exp_color_list,
        linestyles=[exp_linestyle] * len(exp_zorders),
        linewidths=[exp_linewidth] * len(exp_zorders),
        zorders=exp_zorders,
        reverse=False,
    )

    # Let matplotlib compute y-limits from actual plotted lines
    ax.relim()
    ax.autoscale_view()

    # Set title and y-axis visibility
    if title is None:
        title = f"{analysis_object.name} plot of merit"
    ax.set_title(title)
    ax.yaxis.set_visible(show_yaxis)

    return ax
