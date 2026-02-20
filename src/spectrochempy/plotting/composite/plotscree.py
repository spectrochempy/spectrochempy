# ======================================================================================
# Copyright (c) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Scree plot composite function.

This module provides a standalone plot_scree function for visualizing
explained variance in principal component analysis.
"""

__all__ = ["plot_scree"]

import numpy as np

from spectrochempy.utils.mplutils import get_figure
from spectrochempy.utils.mplutils import show as mpl_show


def plot_scree(
    explained,
    cumulative=None,
    *,
    ax=None,
    clear=True,
    title="Scree plot",
    bar_color="tab:blue",
    line_color="tab:orange",
    show=True,
):
    """
    Plot a scree plot with explained variance bars and cumulative curve.

    Creates a scree plot showing individual explained variance as bars
    on the left y-axis and cumulative explained variance as a line on
    the right y-axis (twinx).

    Parameters
    ----------
    explained : array-like
        Explained variance values (as percentages) for each component.
    cumulative : array-like, optional
        Cumulative explained variance values (as percentages).
        If None, computed from ``np.cumsum(explained)``.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    clear : bool, optional
        Whether to clear the axes before plotting. Default: True.
        Only used when ``ax`` is provided.
    title : str or None, optional
        Plot title. Default: "Scree plot".
        If None, no title is set.
    bar_color : color, optional
        Color for the bar plot. Default: "tab:blue".
    line_color : color, optional
        Color for the cumulative line. Default: "tab:orange".
    show : bool, optional
        Whether to display the figure. Default: True.

    Returns
    -------
    matplotlib.axes.Axes
        The primary axes (left y-axis with bars).

    Examples
    --------
    >>> import numpy as np
    >>> from spectrochempy import plot_scree
    >>> explained = np.array([40.0, 25.0, 15.0, 10.0, 5.0, 3.0, 2.0])
    >>> ax = plot_scree(explained, show=False)
    """
    explained = np.asarray(explained)
    if cumulative is None:
        cumulative = np.cumsum(explained)
    else:
        cumulative = np.asarray(cumulative)

    n = len(explained)
    x = np.arange(1, n + 1)

    if ax is None:
        fig = get_figure()
        ax = fig.add_subplot(111)
    elif clear:
        ax.clear()

    ax.bar(x, explained, color=bar_color, align="center")
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0, max(explained) * 1.05)
    ax.set_xlabel("components")
    ax.set_ylabel("explained variance / %")

    ax2 = ax.twinx()
    ax2.plot(x, cumulative, color=line_color, marker="o", markersize=4)

    first_explained = explained[0]
    raw_min = first_explained - 2.0
    raw_min = max(0, raw_min)

    if raw_min >= 10:
        nice_min = 5 * int(raw_min / 5)
    else:
        nice_min = float(int(raw_min))

    if nice_min >= first_explained:
        if first_explained >= 10:
            nice_min -= 5
        else:
            nice_min -= 1

    if first_explained <= 5:
        nice_min = 0.0

    ax2.set_ylim(nice_min, 100)
    ax2.set_ylabel("cumulative explained variance / %")

    if title is not None:
        ax.set_title(title)

    if show:
        mpl_show()

    return ax
