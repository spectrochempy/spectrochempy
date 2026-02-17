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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

from spectrochempy.plotting._render import render_lines
from spectrochempy.plotting._style import resolve_stack_colors
from spectrochempy.utils.mplutils import show as mpl_show


def plotmerit(
    analysis_object,
    X=None,
    X_hat=None,
    *,
    ax=None,
    clear=True,
    orig_c=None,
    recon_c=None,
    resid_c=None,
    orig_linestyle="-",
    recon_linestyle="-",
    resid_linestyle="-",
    orig_linewidth=1.2,
    recon_linewidth=1.0,
    resid_linewidth=0.8,
    recon_alpha=0.65,
    offset=0,
    min_contrast=1.5,
    title=None,
    show_yaxis=False,
    nb_traces="all",
    show=True,
    **kwargs,
):
    r"""
    Plot the original, reconstructed, and residual spectra.

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
    orig_c : color, colormap, or list of colors, optional
        Color(s) for original spectra.
    recon_c : color, colormap, or list of colors, optional
        Color(s) for reconstructed spectra. If None, uses same colors as original
        with transparency applied via recon_alpha.
    resid_c : color, optional
        Color for residual spectra. Default: neutral gray ("0.4").
    orig_linestyle : str, optional
        Line style for original spectra. Default: "-".
    recon_linestyle : str, optional
        Line style for reconstructed spectra. Default: "-".
    resid_linestyle : str, optional
        Line style for residual spectra. Default: "-".
    orig_linewidth : float, optional
        Line width for original spectra. Default: 1.2.
    recon_linewidth : float, optional
        Line width for reconstructed spectra. Default: 1.0.
    resid_linewidth : float, optional
        Line width for residual spectra. Default: 0.8.
    recon_alpha : float, optional
        Transparency for reconstructed spectra (0-1). Default: 0.65.
    offset : float, optional
        Separation (in percent) between original, reconstructed, and residual. Default: 0.
    min_contrast : float, optional
        Minimum contrast ratio for sequential colormaps. Default: 1.5.
    title : str, optional
        Plot title.
    show_yaxis : bool, optional
        Whether to show y-axis. Default: False.
    nb_traces : int or 'all', optional
        Number of lines to display. Default: 'all'.
    show : bool, optional
        Whether to display the figure. Default: True.
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

    # Color resolution helpers
    def _normalize_colors(colors_raw, n):
        """Normalize colors to list of tuples (RGB)."""
        colors = []
        for c in colors_raw:
            if hasattr(c, "__iter__") and not isinstance(c, str):
                colors.append(tuple(float(x) for x in c[:3]))
            else:
                colors.append(c)
        return colors

    def _is_colormap_name(s):
        """Check if string looks like a colormap name (not a color name)."""
        if not isinstance(s, str):
            return False
        if s.endswith("_r"):
            return True
        if s in ("continuous", "categorical"):
            return False
        return s in colormaps

    # Resolve original colors using L1
    orig_colors_raw, orig_is_categorical, _ = resolve_stack_colors(
        dataset=X,
        palette=orig_c,
        n=n_traces,
        geometry="line",
        min_contrast=min_contrast,
    )
    orig_colors = _normalize_colors(orig_colors_raw, n_traces)

    # Resolve reconstructed colors
    # If recon_c is None, use original colors with transparency
    # If recon_c is explicit, resolve similarly to original
    if recon_c is None:
        recon_colors = orig_colors
    elif isinstance(recon_c, str) and not _is_colormap_name(recon_c):
        recon_colors = [recon_c]
    else:
        recon_colors_raw, _, _ = resolve_stack_colors(
            dataset=X,
            palette=recon_c,
            n=1 if n_traces == 1 else n_traces,
            geometry="line",
            min_contrast=min_contrast,
        )
        recon_colors = _normalize_colors(recon_colors_raw, n_traces)
        if n_traces > 1 and len(recon_colors) == 1:
            recon_colors = recon_colors * n_traces

    # Resolve residual colors (default: neutral gray)
    if resid_c is None:
        resid_colors = ["0.4"]
    elif isinstance(resid_c, str) and not _is_colormap_name(resid_c):
        resid_colors = [resid_c]
    else:
        resid_colors_raw, _, _ = resolve_stack_colors(
            dataset=X,
            palette=resid_c,
            n=1 if n_traces == 1 else n_traces,
            geometry="line",
            min_contrast=min_contrast,
        )
        resid_colors = _normalize_colors(resid_colors_raw, n_traces)
        if n_traces > 1 and len(resid_colors) == 1:
            resid_colors = resid_colors * n_traces

    # Prepare data with offsets
    res_offset = res - mad
    recon_offset = X_hat - mao
    orig_offset = X

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
        recon_data = np.asarray(recon_offset.masked_data)
        orig_data = np.asarray(orig_offset.masked_data)
    else:
        res_data = np.atleast_2d(np.asarray(res_offset.masked_data))
        recon_data = np.atleast_2d(np.asarray(recon_offset.masked_data))
        orig_data = np.atleast_2d(np.asarray(orig_offset.masked_data))

    # Prepare style parameters
    if is_2d:
        orig_color_list = orig_colors
        recon_color_list = recon_colors[0] if len(recon_colors) == 1 else recon_colors
        resid_color_list = resid_colors[0] if len(resid_colors) == 1 else resid_colors
    else:
        orig_color_list = [orig_colors[0]]
        recon_color_list = [recon_colors[0]]
        resid_color_list = [resid_colors[0]]

    # Zorder policy:
    # residual (1) - bottom, gray, thin
    # original (2) - middle, solid
    # reconstructed (3) - top, semi-transparent (alpha applied)
    if is_2d:
        n = res_data.shape[0]
        resid_zorders = [1] * n
        orig_zorders = [2] * n
        recon_zorders = [3] * n
    else:
        resid_zorders = [1]
        orig_zorders = [2]
        recon_zorders = [3]

    # Render residual (bottom layer - zorder 1)
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

    # Render original (middle layer - zorder 2)
    render_lines(
        ax,
        xdata,
        orig_data,
        colors=orig_color_list,
        linestyles=[orig_linestyle] * len(orig_zorders),
        linewidths=[orig_linewidth] * len(orig_zorders),
        zorders=orig_zorders,
        reverse=False,
    )

    # Render reconstructed (top layer - zorder 3, with alpha)
    render_lines(
        ax,
        xdata,
        recon_data,
        colors=recon_color_list,
        linestyles=[recon_linestyle] * len(recon_zorders),
        linewidths=[recon_linewidth] * len(recon_zorders),
        zorders=recon_zorders,
        reverse=False,
        alpha=recon_alpha,
    )

    # Let matplotlib compute y-limits from actual plotted lines
    ax.relim()
    ax.autoscale_view()

    # Set title and y-axis visibility
    if title is None:
        title = f"{analysis_object.name} plot of merit"
    ax.set_title(title)
    ax.yaxis.set_visible(show_yaxis)

    if show:
        mpl_show()

    return ax
