# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# ======================================================================================

"""
Composite plotmerit and plot_compare functions.

plot_compare:
    Generic comparison between two datasets (experimental vs reference).

plotmerit:
    Thin wrapper around plot_compare for AnalysisBase objects,
    including multi-parameter reconstructions.
"""

__all__ = ["plot_merit", "plot_compare"]

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from spectrochempy.plotting._render import render_lines
from spectrochempy.utils.mplutils import make_label
from spectrochempy.utils.mplutils import show as mpl_show

# ======================================================================================
# Generic comparison function
# ======================================================================================


def plot_compare(
    X,
    X_ref,
    *,
    ax=None,
    clear=True,
    residual=True,
    title=None,
    show_yaxis=True,
    show=True,
):
    """
    Compare two datasets (X vs X_ref) with optional residual.

    Parameters
    ----------
    X : NDDataset
        reference dataset.
    X_ref : NDDataset
        Dataset to compare against (e.g., reconstructed).
    residual : bool, optional
        Whether to plot residual X - X_ref.
    """

    # ----------------------------
    # Figure management (L3 only)
    # ----------------------------
    if ax is None:
        _, ax = plt.subplots()

    if clear:
        ax.cla()

    # ----------------------------
    # Shape validation
    # ----------------------------
    if X.ndim == 1:
        X = X.atleast_2d()
        X_ref = X_ref.atleast_2d()

    if X.ndim != 2 or X_ref.ndim != 2:
        raise ValueError("X and X_ref must be 1D or 2D datasets.")

    if X.shape != X_ref.shape:
        raise ValueError(f"Shape mismatch: X {X.shape} vs X_ref {X_ref.shape}")

    n_traces, n_points = X.shape

    # ----------------------------
    # Compute data arrays
    # ----------------------------
    orig_data = np.asarray(X.masked_data)
    ref_data = np.asarray(X_ref.masked_data)

    if residual:
        res_data = np.asarray((X - X_ref).masked_data)
    else:
        res_data = None

    # ----------------------------
    # X coordinate extraction
    # ----------------------------
    dimx = X.dims[-1]
    x = getattr(X, dimx)

    if x is not None and x._implements("CoordSet"):
        x = x.default

    if x is not None and (not x.is_empty or x.is_labeled):
        xdata = x.data
        if xdata is None or not np.any(xdata):
            xdata = np.arange(n_points)
    else:
        xdata = np.arange(n_points)

    xdata = np.asarray(xdata)

    # ----------------------------
    # Semantic colors (fixed)
    # ----------------------------
    orig_color = ["tab:blue"]
    ref_color = ["tab:orange"]
    resid_color = ["0.6"]

    # ----------------------------
    # Z-order policy
    # residual < reconstructed < experimental
    # ----------------------------
    resid_z = [0] * n_traces
    ref_z = [1] * n_traces
    orig_z = [2] * n_traces

    linewidth = 1.0

    # ----------------------------
    # Render residual first
    # ----------------------------
    if residual:
        render_lines(
            ax,
            xdata,
            res_data,
            colors=resid_color,
            linestyles=["-"] * n_traces,
            linewidths=[linewidth] * n_traces,
            zorders=resid_z,
            reverse=False,
        )

    # ----------------------------
    # Render reconstructed
    # ----------------------------
    render_lines(
        ax,
        xdata,
        ref_data,
        colors=ref_color,
        linestyles=["-"] * n_traces,
        linewidths=[linewidth] * n_traces,
        zorders=ref_z,
        reverse=False,
    )

    # ----------------------------
    # Render experimental
    # ----------------------------
    render_lines(
        ax,
        xdata,
        orig_data,
        colors=orig_color,
        linestyles=["-"] * n_traces,
        linewidths=[linewidth] * n_traces,
        zorders=orig_z,
        reverse=False,
    )

    # ----------------------------
    # Deterministic limits
    # ----------------------------
    ax.set_xlim(xdata.min(), xdata.max())

    if residual:
        y_min = np.nanmin(res_data)
    else:
        y_min = min(np.nanmin(orig_data), np.nanmin(ref_data))

    y_max = max(np.nanmax(orig_data), np.nanmax(ref_data))

    data_range = y_max - y_min
    pad = 0.02 * data_range if data_range > 0 else 0.02

    ax.set_ylim(y_min - pad, y_max + pad)

    # ----------------------------
    # Axis labeling
    # ----------------------------
    xlabel = make_label(x, dimx)
    ax.set_xlabel(xlabel)

    ylabel = make_label(X, "values")
    ax.set_ylabel(ylabel)

    # Axis reversal (e.g., IR wavenumber)
    if x is not None:
        reversed_flag = getattr(x, "reversed", False)
        if reversed_flag:
            ax.invert_xaxis()

    # ----------------------------
    # Legend
    # ----------------------------
    handles = [
        Line2D([0], [0], color="tab:blue", label=X.name),
        Line2D([0], [0], color="tab:orange", label=X_ref.name),
    ]

    if residual:
        handles.append(Line2D([0], [0], color="0.6", label="difference"))

    ax.legend(handles=handles, loc="best")

    # ----------------------------
    # Title
    # ----------------------------
    if title is not None:
        ax.set_title(title)

    ax.yaxis.set_visible(show_yaxis)

    if show:
        mpl_show()

    return ax


# ======================================================================================
# plotmerit (Analysis wrapper)
# ======================================================================================


def plot_merit(
    analysis_object,
    X=None,
    X_hat=None,
    index=None,
    *,
    ax=None,
    clear=True,
    title=None,
    show_yaxis=True,
    show=True,
    **kwargs,
):
    """
    Plot merit for an analysis object.

    Delegates rendering to plot_compare().
    """

    # Backward compatibility (old IRIS API)
    if X is not None and isinstance(X, (int, np.integer)):
        index = int(X)
        X = None
        X_hat = None

    # Retrieve data
    if X is None:
        X = analysis_object.X
        if X_hat is None:
            X_hat = analysis_object.inverse_transform()
    elif X_hat is None:
        raise ValueError("If X is provided, X_hat must also be provided.")

    # Dimensionality
    X_ndim = X.ndim
    X_hat_ndim = X_hat.ndim

    # ----------------------------
    # Multi-parameter reconstruction
    # ----------------------------
    if X_hat_ndim == X_ndim + 1:
        n_params = X_hat.shape[0]

        # Plot all regularizations on same axes
        if index is None:
            if ax is None:
                _, ax = plt.subplots()

            if clear:
                ax.cla()

            for i in range(n_params):
                X_hat_i = X_hat[i].squeeze()

                plot_compare(
                    X,
                    X_hat_i,
                    ax=ax,
                    clear=False,
                    residual=True,
                    show=False,
                )

            if title is not None:
                ax.set_title(title)

            if show:
                mpl_show()

            return ax

        # Single index
        if isinstance(index, int):
            X_hat_i = X_hat[index].squeeze()
            return plot_compare(
                X,
                X_hat_i,
                ax=ax,
                clear=clear,
                residual=True,
                title=title,
                show_yaxis=show_yaxis,
                show=show,
            )

        # Iterable of indices
        indices = list(index)
        axes_list = []

        for i in indices:
            X_hat_i = X_hat[i].squeeze()
            ax_i = plot_compare(
                X,
                X_hat_i,
                ax=None,
                clear=True,
                residual=True,
                title=title,
                show_yaxis=show_yaxis,
                show=show,
            )
            axes_list.append(ax_i)

        return axes_list

    # ----------------------------
    # Single reconstruction
    # ----------------------------
    elif X_hat_ndim == X_ndim:
        return plot_compare(
            X,
            X_hat,
            ax=ax,
            clear=clear,
            residual=True,
            title=title,
            show_yaxis=show_yaxis,
            show=show,
        )

    else:
        raise ValueError(
            f"Unexpected dimensionality: X_hat has {X_hat_ndim} dimensions, "
            f"expected {X_ndim} or {X_ndim + 1}"
        )
