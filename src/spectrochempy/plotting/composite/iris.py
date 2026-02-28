# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
IRIS plotting composite functions.

This module provides plotting functions for IRIS analysis results
following the L1/L2/L3 plotting architecture.
"""

__all__ = [
    "plot_iris_lcurve",
    "plot_iris_distribution",
    "plot_iris_merit",
]


from spectrochempy.plotting._render import render_contour
from spectrochempy.plotting._render import render_scatter
from spectrochempy.plotting._style import resolve_2d_colormap
from spectrochempy.plotting._style import resolve_line_style
from spectrochempy.utils.exceptions import NotFittedError
from spectrochempy.utils.mplutils import get_figure
from spectrochempy.utils.mplutils import show as mpl_show


def plot_iris_lcurve(
    analysis_object,
    *,
    ax=None,
    clear=True,
    show=True,
    scale="ll",
    title="L curve",
    marker="o",
    color=None,
    markersize=None,
    **kwargs,
):
    """
    Plot the L-curve for IRIS analysis.

    Parameters
    ----------
    analysis_object : IRIS
        The IRIS analysis object.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, create new figure.
    clear : bool, optional
        Whether to clear the axes before plotting. Default: True.
    show : bool, optional
        Whether to display the figure. Default: True.
    scale : str, optional
        Two-character string indicating log scale for x and y axes.
        First character: y-axis (l=log, n=normal).
        Second character: x-axis (l=log, n=normal).
        Default: "ll".
    title : str, optional
        Plot title. Default: "L curve".
    marker : str, optional
        Marker style. Default: "o".
    color : color, optional
        Color for the scatter points.
    markersize : float, optional
        Marker size.
    **kwargs
        Additional keyword arguments passed to style resolution.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes.
    """
    if not analysis_object._fitted:
        raise NotFittedError("The fit method must be used before using this method")

    rss = analysis_object.RSS
    sm = analysis_object.SM

    if ax is None:
        fig = get_figure()
        ax = fig.add_subplot(111)
    elif clear:
        ax.clear()

    style_kwargs = resolve_line_style(
        dataset=None,
        geometry="scatter",
        kwargs=dict(
            marker=marker,
            color=color,
            markersize=markersize,
            **kwargs,
        ),
    )

    render_scatter(
        ax,
        rss,
        sm,
        colors=style_kwargs.get("color"),
        marker=style_kwargs.get("marker"),
        markersizes=style_kwargs.get("markersize"),
        alpha=style_kwargs.get("alpha"),
    )

    ax.set_title(title)
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Curvature")

    if len(scale) >= 1 and scale[0] == "l":
        ax.set_yscale("log")
    if len(scale) >= 2 and scale[1] == "l":
        ax.set_xscale("log")

    if show:
        mpl_show()

    return ax


def plot_iris_distribution(
    analysis_object,
    *,
    index=None,
    ax=None,
    clear=True,
    show=True,
    title=None,
    cmap=None,
    cmap_mode="auto",
    center=None,
    **kwargs,
):
    """
    Plot the distribution function for IRIS analysis.

    Parameters
    ----------
    analysis_object : IRIS
        The IRIS analysis object.
    index : int or list of int, optional
        Index(es) of the inversions (lambda values) to plot.
        If None, plot all.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If provided, used for single index only.
    clear : bool, optional
        Whether to clear the axes before plotting. Default: True.
    show : bool, optional
        Whether to display the figure. Default: True.
    title : str, optional
        Plot title. If None, default lambda-aware title is used.
    cmap : str or Colormap, optional
        Colormap name or object.
    cmap_mode : str, optional
        Colormap mode ("auto", "sequential", "diverging").
        Default: "auto".
    center : float, optional
        Center value for diverging colormaps.
    **kwargs
        Additional keyword arguments passed to colormap resolution.

    Returns
    -------
    matplotlib.axes.Axes or list of matplotlib.axes.Axes
        The matplotlib axes. Returns a list for multiple indices,
        single Axes for single index.
    """
    if not analysis_object._fitted:
        raise NotFittedError("The fit method must be used before using this method")

    if index is None:
        index = list(range(len(analysis_object._lambdas)))
    elif isinstance(index, int):
        index = [index]

    axeslist = []

    for i in index:
        f_i = analysis_object.f[i].squeeze()

        if ax is None or len(index) > 1:
            fig = get_figure()
            ax = fig.add_subplot(111)
        elif clear:
            ax.clear()

        cmap_resolved, norm = resolve_2d_colormap(
            data=f_i.data,
            cmap=cmap,
            cmap_mode=cmap_mode,
            center=center,
            geometry="contour",
        )

        render_contour(
            ax,
            x=f_i.x,
            y=f_i.y,
            Z=f_i.data,
            cmap=cmap_resolved,
            norm=norm,
            filled=False,
        )

        if title is None:
            title = rf"2D IRIS distribution, $\lambda$ = {analysis_object._lambdas.data[i]:.2e}"
        ax.set_title(title)

        if show:
            mpl_show()

        axeslist.append(ax)

        if len(index) > 1:
            ax = None

    if len(index) == 1:
        return axeslist[0]
    return axeslist


def plot_iris_merit(
    analysis_object,
    *,
    index=None,
    ax=None,
    clear=True,
    show=True,
    title=None,
    **kwargs,
):
    """
    Plot the merit function (original, reconstructed, residuals) for IRIS analysis.

    Parameters
    ----------
    analysis_object : IRIS
        The IRIS analysis object.
    index : int or list of int, optional
        Index(es) of the inversions (lambda values) to plot.
        If None, plot all.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If provided, used for single index only.
    clear : bool, optional
        Whether to clear the axes before plotting. Default: True.
    show : bool, optional
        Whether to display the figure. Default: True.
    title : str, optional
        Plot title. If None, default lambda-aware title is used.
    **kwargs
        Additional keyword arguments passed to plotmerit.

    Returns
    -------
    matplotlib.axes.Axes or list of matplotlib.axes.Axes
        The matplotlib axes. Returns a list for multiple indices,
        single Axes for single index.
    """
    if not analysis_object._fitted:
        raise NotFittedError("The fit method must be used before using this method")

    if index is None:
        index = list(range(len(analysis_object._lambdas)))
    elif isinstance(index, int):
        index = [index]

    X = analysis_object.X
    X_hat = analysis_object.inverse_transform()

    from spectrochempy.plotting.composite import plotmerit

    axeslist = []

    for i in index:
        X_hat_i = X_hat[i].squeeze() if X_hat.ndim == 3 else X_hat

        if ax is None or len(index) > 1:
            fig = get_figure()
            current_ax = fig.add_subplot(111)
        else:
            current_ax = ax
            if clear:
                current_ax.clear()

        plotmerit(
            analysis_object=analysis_object,
            X=X,
            X_hat=X_hat_i,
            ax=current_ax,
            clear=False,
            show=False,
            **kwargs,
        )

        if title is None:
            title = rf"2D IRIS merit plot, $\lambda$ = {analysis_object._lambdas.data[i]:.2e}"
        current_ax.set_title(title)

        if show:
            mpl_show()

        axeslist.append(current_ax)

        if len(index) > 1:
            ax = None

    if len(index) == 1:
        return axeslist[0]
    return axeslist
