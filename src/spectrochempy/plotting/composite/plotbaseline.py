# ======================================================================================
# Copyright (c) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Composite plot_baseline function for Baseline correction visualization.

Produces a two-axes plot:
  - Top: original spectra + baseline curves (baseline drawn IN FRONT via zorder)
  - Bottom: baseline-corrected spectra

L3 composite: figure creation, orchestration, style resolution.
"""

__all__ = ["plot_baseline"]

import matplotlib.pyplot as plt
import numpy as np

from spectrochempy.plotting._render import render_lines
from spectrochempy.plotting._style import resolve_stack_colors
from spectrochempy.utils.mplutils import make_label
from spectrochempy.utils.mplutils import show as mpl_show


def plot_baseline(
    original,
    baseline,
    corrected,
    *,
    regions=None,
    show_regions=False,
    region_color=None,
    region_alpha=None,
    ax=None,
    clear=True,
    show=True,
    linewidth=1.0,
    linestyle="-",
    baseline_linestyle="-",
    baseline_color="tab:orange",
    **kwargs,
):
    """
    Plot original, baseline, and corrected spectra in two stacked axes.

    Parameters
    ----------
    original : NDDataset
        Original spectra (1D or 2D).
    baseline : NDDataset
        Computed baseline (same shape as original).
    corrected : NDDataset
        Baseline-corrected spectra (same shape as original).
    regions : iterable of (x0, x1) pairs, optional
        Baseline fitting regions to highlight on the plot.
        Each region is a pair of x-coordinates in physical units.
    show_regions : bool, optional
        If True and regions provided, display region spans on the top axis.
        Default is False.
    region_color : str, optional
        Face color for region spans. If None, uses preferences.baseline_region_color.
    region_alpha : float, optional
        Transparency for region spans (0-1). If None, uses preferences.baseline_region_alpha.
    ax : None
        Must be None. This function creates its own two-axes figure.
    clear : bool, optional
        If True, clear axes before plotting. Default is True.
    show : bool, optional
        If True, display the figure. Default is True.
    linewidth : float, optional
        Line width for all lines. Default is 1.0.
    linestyle : str, optional
        Line style for original and corrected spectra. Default is "-".
    baseline_linestyle : str, optional
        Line style for baseline curves. Default is "-".
    baseline_color : str, optional
        Color for baseline curves. Default is "tab:orange".
    **kwargs
        Additional keyword arguments (reserved for future use).

    Returns
    -------
    tuple
        (ax_top, ax_bottom) matplotlib axes.

    Raises
    ------
    ValueError
        If ax is provided (this function creates its own axes).
        If input shapes do not match.
        If input ndim is not 1 or 2.
    """
    if ax is not None:
        raise ValueError(
            "plot_baseline creates its own two-axes figure. "
            "The 'ax' parameter must be None."
        )

    from spectrochempy.application.preferences import preferences as prefs

    if region_color is None:
        region_color = prefs.baseline_region_color
    if region_alpha is None:
        region_alpha = prefs.baseline_region_alpha

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    if clear:
        ax1.cla()
        ax2.cla()

    if original.ndim not in (1, 2):
        raise ValueError(
            f"original must be 1D or 2D, got ndim={original.ndim}"
        )
    if baseline.ndim not in (1, 2):
        raise ValueError(
            f"baseline must be 1D or 2D, got ndim={baseline.ndim}"
        )
    if corrected.ndim not in (1, 2):
        raise ValueError(
            f"corrected must be 1D or 2D, got ndim={corrected.ndim}"
        )

    if original.ndim == 1:
        orig_data = np.asarray(original.masked_data).reshape(1, -1)
    else:
        orig_data = np.asarray(original.masked_data)

    if baseline.ndim == 1:
        base_data = np.asarray(baseline.masked_data).reshape(1, -1)
    else:
        base_data = np.asarray(baseline.masked_data)

    if corrected.ndim == 1:
        corr_data = np.asarray(corrected.masked_data).reshape(1, -1)
    else:
        corr_data = np.asarray(corrected.masked_data)

    if orig_data.shape != base_data.shape:
        raise ValueError(
            f"Shape mismatch: original {orig_data.shape} vs baseline {base_data.shape}"
        )
    if orig_data.shape != corr_data.shape:
        raise ValueError(
            f"Shape mismatch: original {orig_data.shape} vs corrected {corr_data.shape}"
        )

    n_traces, n_points = orig_data.shape

    dimx = original.dims[-1]
    x = getattr(original, dimx, None)

    if x is not None and x._implements("CoordSet"):
        x = x.default

    if x is not None and (not x.is_empty or x.is_labeled):
        xdata = x.data
        if xdata is None or not np.any(xdata):
            xdata = np.arange(n_points)
    else:
        xdata = np.arange(n_points)

    xdata = np.asarray(xdata)

    orig_colors_raw, _, _ = resolve_stack_colors(
        dataset=original,
        palette=None,
        n=n_traces,
        geometry="line",
    )

    if not isinstance(orig_colors_raw, list):
        if hasattr(orig_colors_raw, "__iter__"):
            orig_colors = list(orig_colors_raw)
            if len(orig_colors) == 1 and n_traces > 1:
                orig_colors = orig_colors * n_traces
            elif len(orig_colors) < n_traces:
                orig_colors = list(orig_colors_raw) * (
                    (n_traces // len(orig_colors_raw)) + 1
                )
                orig_colors = orig_colors[:n_traces]
        else:
            orig_colors = [orig_colors_raw] * n_traces
    else:
        orig_colors = orig_colors_raw
        if len(orig_colors) < n_traces:
            orig_colors = orig_colors * ((n_traces // len(orig_colors)) + 1)
            orig_colors = orig_colors[:n_traces]

    corrected_colors = orig_colors

    base_colors = [baseline_color] * n_traces

    orig_zorders = [1] * n_traces
    base_zorders = [2] * n_traces
    corr_zorders = [1] * n_traces

    orig_linestyles = [linestyle] * n_traces
    base_linestyles = [baseline_linestyle] * n_traces
    corr_linestyles = [linestyle] * n_traces

    orig_linewidths = [linewidth] * n_traces
    base_linewidths = [linewidth] * n_traces
    corr_linewidths = [linewidth] * n_traces

    if show_regions and regions:
        for r in regions:
            if r is None or len(r) < 2:
                continue
            x0, x1 = r[0], r[1]
            xmin, xmax = sorted([x0, x1])
            ax1.axvspan(
                xmin,
                xmax,
                facecolor=region_color,
                alpha=region_alpha,
                zorder=0,
            )

    render_lines(
        ax1,
        xdata,
        orig_data,
        colors=orig_colors,
        linestyles=orig_linestyles,
        linewidths=orig_linewidths,
        zorders=orig_zorders,
        reverse=False,
    )

    render_lines(
        ax1,
        xdata,
        base_data,
        colors=base_colors,
        linestyles=base_linestyles,
        linewidths=base_linewidths,
        zorders=base_zorders,
        reverse=False,
    )

    render_lines(
        ax2,
        xdata,
        corr_data,
        colors=corrected_colors,
        linestyles=corr_linestyles,
        linewidths=corr_linewidths,
        zorders=corr_zorders,
        reverse=False,
    )

    ax1.set_xlim(xdata.min(), xdata.max())

    top_min = np.nanmin([np.nanmin(orig_data), np.nanmin(base_data)])
    top_max = np.nanmax([np.nanmax(orig_data), np.nanmax(base_data)])

    top_range = top_max - top_min

    if top_range > 0:
        pad = 0.02 * top_range
    else:
        pad = 0.02 * max(abs(top_max), 1.0)

    ax1.set_ylim(top_min - pad, top_max + pad)

    bot_min = np.nanmin(corr_data)
    bot_max = np.nanmax(corr_data)

    bot_range = bot_max - bot_min

    if bot_range > 0:
        pad = 0.02 * bot_range
    else:
        pad = 0.02 * max(abs(bot_max), 1.0)

    ax2.set_ylim(bot_min - pad, bot_max + pad)

    xlabel = make_label(x, dimx)
    ax2.set_xlabel(xlabel)

    ylabel = make_label(original, "values")
    ax1.set_ylabel(ylabel)
    ax2.set_ylabel(ylabel)

    if x is not None:
        reversed_flag = getattr(x, "reversed", False)
        if not reversed_flag and len(xdata) >= 2 and xdata[0] > xdata[-1]:
            reversed_flag = True
        if reversed_flag:
            ax1.invert_xaxis()

    if show:
        mpl_show()

    return (ax1, ax2)
