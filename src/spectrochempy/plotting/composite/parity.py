# ======================================================================================
# Copyright (c) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Parity plot function.

This module provides a standalone parityplot function for visualizing
predicted vs measured values.
"""

__all__ = ["parityplot"]


from spectrochempy.utils.mplutils import _maybe_show
from spectrochempy.utils.mplutils import _setup_axes


def parityplot(
    Y,
    Y_hat,
    *,
    ax=None,
    clear=True,
    show=True,
    s=None,
    c=None,
    marker=None,
    cmap=None,
    norm=None,
    vmin=None,
    vmax=None,
    alpha=0.5,
    linewidths=None,
    edgecolors=None,
    plotnonfinite=False,
    data=None,
    **kwargs,
):
    """
    Plot predicted vs measured values (parity plot).

    Creates a scatter plot of ``Y_hat`` (predicted) vs ``Y`` (measured)
    with a diagonal reference line (y = x).

    Parameters
    ----------
    Y : `NDDataset`
        Measured values.
    Y_hat : `NDDataset`
        Predicted values.
    ax : `~matplotlib.axes.Axes`, optional
        Axes to plot on. If None, a new figure is created.
    clear : `bool`, optional
        Whether to clear the axes before plotting. Default: True.
        Only used when ``ax`` is provided.
    show : `bool`, optional
        Whether to display the figure. Default: True.
    s : `float` or array-like, optional
        Marker size in points**2.
    c : array-like or list of colors, optional
        Marker colors.
    marker : `str`, optional
        Marker style.
    cmap : `str` or `Colormap`, optional
        Colormap for scalar data mapping.
    norm : `str` or `Normalize`, optional
        Normalization method for scalar data.
    vmin, vmax : `float`, optional
        Data range for colormap.
    alpha : `float`, optional
        Alpha blending value. Default: 0.5.
    linewidths : `float` or array-like, optional
        Linewidth of marker edges.
    edgecolors : color or sequence of colors, optional
        Edge color of markers.
    plotnonfinite : `bool`, optional
        Whether to plot nonfinite data. Default: False.
    data : optional
        Unused parameter for compatibility.
    **kwargs
        Additional keyword arguments passed to `~matplotlib.axes.Axes.scatter`.

    Returns
    -------
    `~matplotlib.axes.Axes`
        The matplotlib axes containing the plot.
    """
    ax = _setup_axes(ax=ax, clear=clear)

    squeeze_ndim = getattr(Y, "_squeeze_ndim", None)
    if squeeze_ndim == 1:
        Y = Y.squeeze()
        Y_hat = Y_hat.squeeze()

    scatter_kwargs = {
        "s": s,
        "c": c,
        "marker": marker,
        "cmap": cmap,
        "norm": norm,
        "vmin": vmin,
        "vmax": vmax,
        "alpha": alpha,
        "linewidths": linewidths,
        "edgecolors": edgecolors,
        "plotnonfinite": plotnonfinite,
        "data": data,
    }
    scatter_kwargs = {k: v for k, v in scatter_kwargs.items() if v is not None}
    scatter_kwargs.update(kwargs)

    if len(Y.shape) == 1:
        ax.scatter(Y.data, Y_hat.data, **scatter_kwargs)
    else:
        for col in range(Y.shape[1]):
            ax.scatter(Y.data[:, col], Y_hat.data[:, col], **scatter_kwargs)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xymin = min(xmin, ymin)
    xymax = max(xmax, ymax)
    ax.set_xlim(xymin, xymax)
    ax.set_ylim(xymin, xymax)
    ax.plot([xymin, xymax], [xymin, xymax])
    ax.legend()
    ax.set_xlabel("measured values")
    ax.set_ylabel("predicted values")
    ax.figure.tight_layout()

    _maybe_show(show)
    return ax
