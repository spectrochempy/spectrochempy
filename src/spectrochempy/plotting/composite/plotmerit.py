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

import numpy as np
from matplotlib.lines import Line2D

from spectrochempy.plotting._render import render_lines
from spectrochempy.utils.mplutils import _maybe_show
from spectrochempy.utils.mplutils import _setup_axes
from spectrochempy.utils.mplutils import make_label

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
    exp_c=None,
    calc_c=None,
    resid_c=None,
    exp_linestyle="-",
    calc_linestyle="--",
    resid_linestyle="-",
    exp_linewidth=1.0,
    calc_linewidth=1.6,
    resid_linewidth=1.0,
    exp_label=None,
    calc_label=None,
    resid_label="difference",
    legend_loc="best",
    kind=None,
    method=None,
    offset=None,
    nb_traces="all",
    **kwargs,
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
    ax = _setup_axes(ax, clear=clear)

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

    if nb_traces != "all":
        nb_traces = int(nb_traces)
        if nb_traces <= 0:
            raise ValueError("nb_traces must be a positive integer or 'all'.")
        if nb_traces < n_traces:
            indices = np.linspace(0, n_traces - 1, nb_traces, dtype=int)
            X = X[indices]
            X_ref = X_ref[indices]
            n_traces = nb_traces

    # ----------------------------
    # Kwargs normalization
    # Pop kwargs that would conflict with render_lines explicit params.
    # Priority: per-category params > kwargs defaults > hardcoded defaults.
    # ----------------------------
    color_kw = kwargs.pop("color", None)
    c_kw = kwargs.pop("c", None)
    if color_kw is not None:
        if exp_c is None:
            exp_c = color_kw
        if calc_c is None:
            calc_c = color_kw
        if resid_c is None:
            resid_c = color_kw
    elif c_kw is not None:
        if exp_c is None:
            exp_c = c_kw
        if calc_c is None:
            calc_c = c_kw
        if resid_c is None:
            resid_c = c_kw

    linestyle_kw = kwargs.pop("linestyle", None)
    ls_kw = kwargs.pop("ls", None)
    if linestyle_kw is not None or ls_kw is not None:
        default_ls = linestyle_kw if linestyle_kw is not None else ls_kw
        exp_linestyle = default_ls
        calc_linestyle = default_ls
        resid_linestyle = default_ls

    marker_kw = kwargs.pop("marker", None)
    markersize_kw = kwargs.pop("markersize", None)
    kwargs.pop("ms", None)

    lw_kw = kwargs.pop("lw", None)
    linewidth_kw = kwargs.pop("linewidth", None)
    if lw_kw is not None:
        exp_linewidth = calc_linewidth = resid_linewidth = lw_kw
    if linewidth_kw is not None:
        exp_linewidth = calc_linewidth = resid_linewidth = linewidth_kw
    # ----------------------------
    # Compute data arrays
    # ----------------------------
    orig_data = np.asarray(X.masked_data)
    ref_data = np.asarray(X_ref.masked_data)

    res_data = np.asarray((X - X_ref).masked_data) if residual else None

    plot_orig_data = orig_data.copy()
    plot_ref_data = ref_data.copy()
    plot_res_data = res_data.copy() if residual else None

    if residual and offset not in (None, 0):
        signal_min = min(np.nanmin(orig_data), np.nanmin(ref_data))
        signal_max = max(np.nanmax(orig_data), np.nanmax(ref_data))
        signal_range = signal_max - signal_min
        offset_scale = signal_range if signal_range > 0 else 1.0
        residual_offset = (float(offset) / 100.0) * offset_scale
        plot_res_data = plot_res_data - residual_offset

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
    # Semantic styles (fixed)
    # ----------------------------
    orig_color = ["tab:blue" if exp_c is None else exp_c]
    ref_color = ["tab:orange" if calc_c is None else calc_c]
    resid_color = ["0.6" if resid_c is None else resid_c]
    orig_linestyle = [exp_linestyle] * n_traces
    ref_linestyle = [calc_linestyle] * n_traces
    resid_linestyle = [resid_linestyle] * n_traces

    plot_kind = (kind or method or "line").lower()
    marker_kwargs = {}
    if plot_kind == "scatter":
        orig_linestyle = ["None"] * n_traces
        ref_linestyle = ["None"] * n_traces
        resid_linestyle = ["None"] * n_traces
        marker_kwargs = {
            "markers": [marker_kw if marker_kw is not None else "o"] * n_traces,
            "markersizes": [markersize_kw if markersize_kw is not None else 3]
            * n_traces,
        }
    elif plot_kind != "line":
        raise ValueError("kind/method must be 'line' or 'scatter'.")

    # ----------------------------
    # Z-order policy
    # residual < experimental < reconstructed
    #
    # Keeping the reconstructed profile on top makes near-perfect fits visible
    # instead of hiding the orange line entirely under the experimental trace.
    # ----------------------------
    resid_z = [0] * n_traces
    orig_z = [1] * n_traces
    ref_z = [2] * n_traces
    # ----------------------------
    # Render residual first
    # ----------------------------
    if residual:
        render_lines(
            ax,
            xdata,
            plot_res_data,
            colors=resid_color,
            linestyles=resid_linestyle,
            linewidths=[resid_linewidth] * n_traces,
            zorders=resid_z,
            reverse=False,
            **marker_kwargs,
            **kwargs,
        )

    # ----------------------------
    # Render experimental
    # ----------------------------
    render_lines(
        ax,
        xdata,
        plot_orig_data,
        colors=orig_color,
        linestyles=orig_linestyle,
        linewidths=[exp_linewidth] * n_traces,
        alpha=0.85,
        zorders=orig_z,
        reverse=False,
        **marker_kwargs,
        **kwargs,
    )

    # ----------------------------
    # Render reconstructed
    # ----------------------------
    render_lines(
        ax,
        xdata,
        plot_ref_data,
        colors=ref_color,
        linestyles=ref_linestyle,
        linewidths=[calc_linewidth] * n_traces,
        zorders=ref_z,
        reverse=False,
        **marker_kwargs,
        **kwargs,
    )

    # ----------------------------
    # Deterministic limits
    # ----------------------------
    ax.set_xlim(xdata.min(), xdata.max())

    plotted_arrays = [plot_orig_data, plot_ref_data]
    if residual:
        plotted_arrays.append(plot_res_data)

    y_min = min(np.nanmin(arr) for arr in plotted_arrays)
    y_max = max(np.nanmax(arr) for arr in plotted_arrays)

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
        Line2D(
            [0],
            [0],
            color=orig_color[0],
            linestyle=orig_linestyle[0],
            marker=marker_kwargs.get("markers", [None])[0],
            label=exp_label or X.name,
        ),
        Line2D(
            [0],
            [0],
            color=ref_color[0],
            linestyle=ref_linestyle[0],
            marker=marker_kwargs.get("markers", [None])[0],
            label=calc_label or X_ref.name,
        ),
    ]

    if residual:
        handles.append(
            Line2D(
                [0],
                [0],
                color=resid_color[0],
                linestyle=resid_linestyle[0],
                marker=marker_kwargs.get("markers", [None])[0],
                label=resid_label,
            ),
        )

    ax.legend(handles=handles, loc=legend_loc)

    # ----------------------------
    # Title
    # ----------------------------
    if title is not None:
        ax.set_title(title)

    ax.yaxis.set_visible(show_yaxis)

    _maybe_show(show)

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
            ax = _setup_axes(ax, clear=clear)

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

            _maybe_show(show)

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
                **kwargs,
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
                **kwargs,
            )
            axes_list.append(ax_i)

        return axes_list

    # ----------------------------
    # Single reconstruction
    # ----------------------------
    if X_hat_ndim == X_ndim:
        return plot_compare(
            X,
            X_hat,
            ax=ax,
            clear=clear,
            residual=True,
            title=title,
            show_yaxis=show_yaxis,
            show=show,
            **kwargs,
        )

    raise ValueError(
        f"Unexpected dimensionality: X_hat has {X_hat_ndim} dimensions, "
        f"expected {X_ndim} or {X_ndim + 1}"
    )
