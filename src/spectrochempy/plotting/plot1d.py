## ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Module containing 1D plotting functions.

This module consumes normalized plotting methods/kwargs and focuses on artist
creation plus 1D axis policy. Backend dispatch and final display ownership live
above this layer.
"""

__all__ = [
    "plot_1D",
    "plot_pen",
    "plot_scatter",
    "plot_bar",
    "plot_multiple",
    "plot_scatter_pen",
]

__dataset_methods__ = __all__

import numpy as np

from spectrochempy.application.preferences import preferences
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.plotting._kwargs import normalize_plot_kwargs
from spectrochempy.plotting._kwargs import normalize_style_argument
from spectrochempy.plotting._methods import validate_method_for_target_dimension
from spectrochempy.plotting._style import resolve_line_style
from spectrochempy.utils.mplutils import make_label
from spectrochempy.utils.typeutils import is_sequence

# --------------------------------------------------------------------------------------
# plot_1D
# --------------------------------------------------------------------------------------


def plot_1D(dataset, method=None, **kwargs):
    """
    Plot of one-dimensional data.

    Parameters
    ----------
    dataset : :class:`~spectrochempy.core.dataset.nddataset.NDDataset`
        Source of data to plot.
    method : str, optional, default: auto-detected from data dimensionality
        Name of plotting method to use. If None, method is chosen based on data
        dimensionality.

        Available 1D plotting methods include:

        - ``pen``: solid line plot
        - ``bar``: bar graph
        - ``scatter``: scatter plot
        - ``scatter+pen``: scatter plot with solid line


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
    reverse : bool or None [optional, default=None/False
        In principle, coordinates run from left to right,
        except for wavenumbers
        (e.g., FTIR spectra) or ppm (e.g., NMR), that spectrochempy
        will try to guess. But if reverse is set, then this is the
        setting which will be taken into account.
    show_complex : bool, optional, default: False
        Show both real and imaginary component for complex data.
        By default only the real component is displayed.
    show : bool, optional, default: True
        Whether SpectroChemPy should perform its explicit display step after
        plotting. In notebook environments, figures may still render inline
        without this explicit call.
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
    twinx : :class:`~matplotlib.axes.Axes`, optional, default: None
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

    Returns
    -------
    Matplolib Axes or None
        The matplotlib axes containing the plot if successful, None otherwise.

    See Also
    --------
    plot
    plot_1D
    plot_pen
    plot_scatter
    plot_bar
    plot_scatter_pen
    plot_multiple
    multiplot

    """
    from spectrochempy.plotting.plot_setup import lazy_ensure_mpl_config

    lazy_ensure_mpl_config()

    import matplotlib.backend_bases  # noqa: F401
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib.ticker import ScalarFormatter

    # Get preferences
    # ----------------------------------------------------------------------------------
    prefs = preferences
    kwargs = normalize_plot_kwargs(kwargs)

    # Resolve plotting style(s) locally (no global rcParams / no prefs.style mutation)
    style = normalize_style_argument(
        kwargs.pop("style", None),
        default=getattr(prefs, "style", None) or ["scpy"],
    )

    # Build font rc overrides from prefs (L1 helper - no mutation)
    from spectrochempy.plotting._style import build_font_rc_overrides

    font_overrides = build_font_rc_overrides(prefs)

    # Build LaTeX font overrides (for mathtext rendering)
    latex_overrides = prefs.set_latex_font(prefs.font.family)

    # Merge overrides: latex_overrides first, then font_overrides (font wins)
    merged_overrides = {}
    merged_overrides.update(latex_overrides)
    merged_overrides.update(font_overrides)

    # Apply styles and font overrides within context for entire plotting operation
    with plt.style.context(style), matplotlib.rc_context(merged_overrides):
        # Redirections ?
        # ------------------------------------------------------------------------
        # should we redirect the plotting to another method
        if dataset._squeeze_ndim > 1:
            if method is None:
                return dataset.plot_2D(**kwargs)
            method = validate_method_for_target_dimension(
                method,
                target="2d+",
                source="plot_1D() with non-1D data",
            )
            return dataset.plot_2D(method=method, **kwargs)

        # if plotly execute plotly routine not this one
        if kwargs.get("use_plotly", prefs.use_plotly):
            raise NotImplementedError(
                "Plotly plotting is not currently available. Use the default Matplotlib backend."
            )

        # often we do need to plot only data
        # when plotting on top of a previous plot
        # data_only = kwargs.get("data_only", False)

        # Get the data to plot
        # ---------------------------------------------------------------
        new = dataset  # .copy()
        if new.size > 1:
            # don't apply to array of size one to preserve the x coordinate!!!!
            new = new.squeeze()

        # is that a plot with twin axis
        is_twinx = kwargs.get("twinx") is not None

        # if dataset is complex it is possible to overlap
        # with the imaginary component
        show_complex = kwargs.pop("show_complex", False)

        if kwargs.pop("scatter", False) and "scatter" not in (method or ""):
            method = "scatter"

        # Resolve line/marker styles using centralized L1 function
        style_kwargs = resolve_line_style(
            dataset=new,
            geometry="line",
            kwargs=kwargs,
            prefs=prefs,
            method=method,
        )

        color = style_kwargs["color"]
        lw = style_kwargs["linewidth"]
        ls = style_kwargs["linestyle"]
        marker = style_kwargs["marker"]
        markersize = style_kwargs["markersize"]
        markerfacecolor = style_kwargs["markerfacecolor"]
        markeredgecolor = style_kwargs["markeredgecolor"]
        alpha = style_kwargs["alpha"]
        markeredgewidth = kwargs.get("markeredgewidth", 1.0)

        markevery = kwargs.get("markevery", kwargs.get("me", 1))

        # Figure setup
        # ------------------------------------------------------------------------
        method, fig, ndaxes = new._figure_setup(
            ndim=1,
            method=method,
            style=style,
            **kwargs,
        )

        pen = "pen" in (method or "") or kwargs.pop("pen", False)
        scatter = "scatter" in method or marker != "auto"
        bar = "bar" in method

        # Use the axes mapping from _figure_setup(). The fallback only covers
        # defensive recovery if a future change returns an incomplete mapping.
        if "main" in ndaxes:
            ax = ndaxes["main"]
        else:
            if fig.get_axes():
                ax = fig.get_axes()[0]
                ax.name = "main"
            else:
                ax = fig.add_subplot(1, 1, 1)
                ax.name = "main"

        # Other ax properties that can be passed as arguments
        # ------------------------------------------------------------------------
        number_x_labels = kwargs.get("n_x_labels", prefs.number_of_x_labels)
        number_y_labels = kwargs.get("n_y_labels", prefs.number_of_y_labels)
        ax.xaxis.set_major_locator(MaxNLocator(number_x_labels))
        ax.yaxis.set_major_locator(MaxNLocator(number_y_labels))
        ax.xaxis.set_ticks_position("bottom")
        if not is_twinx:
            # do not move these label for twin axes!
            ax.yaxis.set_ticks_position("left")

        # the next lines are to avoid multipliers in axis scale
        formatter = ScalarFormatter(useOffset=False)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        xscale = kwargs.get("xscale", "linear")
        yscale = kwargs.get("yscale", "linear")

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        grid = kwargs.get("grid", prefs.axes_grid)
        ax.grid(grid)

        # ----------------------------------------------------------------------------------
        # plot the dataset
        # ----------------------------------------------------------------------------------
        # abscissa axis
        # the actual dimension name is the first in the new.dims list
        dimx = new.dims[-1]
        x = getattr(new, dimx)
        if x is not None and x._implements("CoordSet"):
            # if several coords, take the default ones:
            x = x.default
        xsize = new.size
        show_x_points = False
        if x is not None and hasattr(x, "show_datapoints"):
            show_x_points = x.show_datapoints
        if show_x_points:
            # remove data and units for display
            x = Coord.arange(xsize)

        if x is not None and (not x.is_empty or x.is_labeled):
            xdata = x.data
            # discrete_data = False
            if not np.any(xdata) and x.is_labeled:
                # discrete_data = True
                # take into account the fact that sometimes axis
                # have just labels
                xdata = range(1, len(x.labels) + 1)
        else:
            xdata = range(xsize)

        # take into account the fact that sometimes axis have just labels
        if xdata is None:
            xdata = range(xsize)

        # ordinates (by default we plot real component of the data)
        if not kwargs.pop("imag", False) or kwargs.get("show_complex", False):
            z = new.real
            zdata = z.masked_data
        else:
            z = new.imag
            zdata = z.masked_data

        # plot_lines
        # ------------------------------------------------------------------------
        label = kwargs.get("label")

        # Determine effective linestyle
        if scatter and not pen:
            effective_linestyle = "None"
        elif pen:
            effective_linestyle = ls if ls.upper() != "AUTO" else "-"
        else:
            effective_linestyle = "None"

        # Determine effective marker
        effective_marker = marker if marker.upper() != "AUTO" else None

        if bar:
            # bar only
            line = ax.bar(
                xdata,
                zdata.squeeze(),
                edgecolor="k",
                color=None
                if isinstance(color, str) and color.upper() == "AUTO"
                else color,
                align="center",
                label=label,
                width=kwargs.get("width", 0.1),
                alpha=alpha,
            )
        else:
            # Unified Line2D path for pen, scatter, and scatter_pen
            (line,) = ax.plot(
                xdata,
                zdata.T,
                linestyle=effective_linestyle,
                marker=effective_marker,
                markersize=markersize,
                markeredgewidth=markeredgewidth,
                markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor,
                markevery=markevery,
                label=label,
                alpha=alpha,
            )

            # Set color and linewidth if not auto
            if not (isinstance(color, str) and color.upper() == "AUTO"):
                line.set_color(color)
            if not (isinstance(lw, str) and lw.upper() == "AUTO"):
                line.set_linewidth(lw)

        if show_complex and pen:
            # add the imaginary component for pen only plot
            zimagdata = new.imag.masked_data
            ax.plot(xdata, zimagdata.T, ls="--")

        if kwargs.get("plot_model", False):
            import warnings

            warnings.warn(
                "NDDataset.modeldata has been removed. "
                "Fit/model results should be represented as explicit NDDataset objects.",
                FutureWarning,
                stacklevel=4,
            )

        # ----------------------------------------------------------------------------------
        # axis
        # ----------------------------------------------------------------------------------
        # axis
        # ----------------------------------------------------------------------------------
        data_only = kwargs.get("data_only", False)

        if len(xdata) > 1:
            # abscissa limits?
            xl = [xdata[0], xdata[-1]]
            xl.sort()

            if bar or len(xdata) < number_x_labels + 1:
                # extend the axis so that the labels are not too close to limits
                inc = (xdata[1] - xdata[0]) * 0.5
                xl = [xl[0] - inc, xl[1] + inc]

            # ordinates limits?
            amp = np.ma.ptp(z.masked_data) / 50.0
            zl = [
                np.ma.min(z.masked_data) - amp,
                np.ma.max(z.masked_data) + amp,
            ]

            # check if some data are not already present on the graph
            # and take care of their limits
            multiplelines = 2 if kwargs.get("show_zero", False) else 1
            if len(ax.lines) > multiplelines and not show_complex:
                # get the previous xlim and zlim
                xlim = list(ax.get_xlim())
                xl[-1] = max(xlim[-1], xl[-1])
                xl[0] = min(xlim[0], xl[0])

                zlim = list(ax.get_ylim())
                zl[-1] = max(zlim[-1], zl[-1])
                zl[0] = min(zlim[0], zl[0])

        if data_only or len(xdata) == 1:
            xl = ax.get_xlim()

        xlim = list(kwargs.get("xlim", xl))  # we read the argument xlim
        # that should have the priority
        xlim.sort()

        # reversed axis?
        if kwargs.get("x_reverse", kwargs.get("reverse", x.reversed if x else False)):
            xlim.reverse()

        if data_only or len(xdata) == 1:
            zl = ax.get_ylim()

        zlim = list(kwargs.get("zlim", kwargs.get("ylim", zl)))
        # we read the argument zlim or ylim
        # which have the priority
        zlim.sort()

        # set the limits
        if not is_twinx:
            # when twin axes, we keep the setting of the first ax plotted
            ax.set_xlim(xlim)
        else:
            ax.tick_params("y", colors=color)

        ax.set_ylim(zlim)

        if data_only:
            # if data only (we will not set axes and labels
            # it was probably done already in a previous plot
            return ax

        # ------------------------------------------------------------------------
        # labels
        # ------------------------------------------------------------------------
        # x label

        xlabel = kwargs.get("xlabel")
        if show_x_points:
            xlabel = "data points"
        if not xlabel:
            xlabel = make_label(x, new.dims[-1])
        ax.set_xlabel(xlabel)

        # x tick labels

        uselabel = kwargs.get("uselabel", False)
        if x and x.is_labeled and (uselabel or not np.any(x.data)):
            if x.data is not None:
                xt = ax.get_xticks()
                ticklabels = x.labels[x._loc2index(xt), 0]
                ax.set_xticks(ax.get_xticks(), labels=ticklabels, rotation=90.0)
            else:
                ax.set_xticks(xdata)
                ax.set_xticklabels(x.labels)

        # z label

        zlabel = kwargs.get("zlabel", kwargs.get("ylabel"))
        if not zlabel:
            zlabel = make_label(new, "z")

        # ax.set_ylabel(zlabel)

        # do we display the ordinate axis?
        if kwargs.get("show_z", True) and not is_twinx:
            ax.set_ylabel(zlabel)
        elif kwargs.get("show_z", True) and is_twinx:
            ax.set_ylabel(zlabel, color=color)
        else:
            ax.set_yticks([])

        # do we display the zero line
        if kwargs.get("show_zero", False):
            ax.haxlines(label="zero_line")

        # display a title
        # ------------------------------------------------------------------------
        title = kwargs.get("title")
        if title:
            ax.set_title(title)
        elif kwargs.get("plottitle", False):
            ax.set_title(new.name)

        # masks
        if kwargs.get("show_mask", False):
            ax.fill_between(
                xdata,
                zdata.min() - 1.0,
                zdata.max() + 1,
                where=new.mask,
                facecolor="#FFEEEE",
                alpha=0.3,
            )

        return ax


def plot_scatter(dataset, **kwargs):
    """
    Plot a 1D dataset with marker-based scatter rendering.

    This is equivalent to ``dataset.plot(method="scatter", ...)``. Use it when
    the visual intent should be explicit in the code. The compatibility flag
    ``scatter=True`` is still accepted by ``plot()`` for existing user code.

    Parameters
    ----------
    dataset : :class:~spectrochempy.ddataset.nddataset.NDDataset
        Source of data to plot.

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
    twinx : :class:`~matplotlib.axes.Axes`, optional, default: None
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

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes containing the scatter plot.

    See Also
    --------
    plot
    plot_1D
    plot_pen
    plot_bar
    plot_scatter_pen
    plot_multiple
    multiplot
    """
    return plot_1D(dataset, method="scatter", **kwargs)


def plot_pen(dataset, **kwargs):
    """
    Plot a 1D dataset with line rendering.

    This is equivalent to ``dataset.plot(method="pen", ...)`` and is the
    default geometry for ordinary 1D dataset plotting.

    Parameters
    ----------
    dataset : :class:~spectrochempy.ddataset.nddataset.NDDataset
        Source of data to plot.

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
    twinx : :class:`~matplotlib.axes.Axes`, optional, default: None
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

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes containing the line plot.

    See Also
    --------
    plot
    plot_1D
    plot_scatter
    plot_bar
    plot_scatter_pen
    plot_multiple
    multiplot
    """
    return plot_1D(dataset, method="pen", **kwargs)


def plot_scatter_pen(dataset, **kwargs):
    """
    Plot a 1D dataset with solid pen by default.

    Alias of plot (with `method` argument set to `scatter_pen` .

    Parameters
    ----------
    dataset : :class:~spectrochempy.ddataset.nddataset.NDDataset
        Source of data to plot.

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
    twinx : :class:`~matplotlib.axes.Axes`, optional, default: None
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

    Returns
    -------
    Matplolib Axes or None
        The matplotlib axes containing the plot if successful, None otherwise.

    See Also
    --------
    plot
    plot_1D
    plot_pen
    plot_scatter
    plot_bar
    plot_multiple
    multiplot
    """
    return plot_1D(dataset, method="scatter_pen", **kwargs)


def plot_bar(dataset, **kwargs):
    """
    Plot a 1D dataset as a bar chart.

    This is equivalent to ``dataset.plot(method="bar", ...)``.

    Parameters
    ----------
    dataset : :class:~spectrochempy.ddataset.nddataset.NDDataset
        Source of data to plot.

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
    twinx : :class:`~matplotlib.axes.Axes`, optional, default: None
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

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes containing the bar plot.

    See Also
    --------
    plot
    plot_1D
    plot_scatter
    plot_pen
    plot_scatter_pen
    plot_multiple
    multiplot
    """
    return plot_1D(dataset, method="bar", **kwargs)


def plot_multiple(
    datasets,
    method="scatter",
    pen=True,
    labels=None,
    marker="AUTO",
    color="AUTO",
    ls="AUTO",
    lw=1,
    shift=0,
    **kwargs,
):
    """
    Overlay several 1D datasets on a single Matplotlib axes.

    Parameters
    ----------
    datasets : `list` of 1D `NDDataset`
        Datasets to plot. If a single dataset is passed, ``plot_multiple()``
        falls back to the regular ``dataset.plot()`` path and preserves the
        requested method.
    method : `str` among [scatter, pen]
        Geometry to use for each overlaid dataset. ``"scatter"`` produces
        marker-based rendering; ``"pen"`` produces line rendering.
    pen : bool, optional, default: True
        If ``method="scatter"``, also draw connecting lines between markers.
    labels : a `list` of `str`, optional
        Labels used for the legend. The length of the list must match the number
        of datasets to plot.
    marker : `str`, list` os `str` or `AUTO`, optional, default: 'AUTO'
        Marker type for scatter plot. If marker is not provided then the scatter type
        of plot is chosen automatically.
    color : `str`, list` os `str` or `AUTO`, optional, default: 'AUTO'
        Color of the lines. If color is not provided then the color of the lines is
        chosen automatically.
    ls: `str`, `list` os `str` or `AUTO`, optional, default: 'AUTO'
        Line style definition. If ls is not provided then the line style is chosen
        automatically.
    lw: `float`, `list`of  `floats`, optional, default: 1.0
        Line width. If lw is not provided then the line width is chosen automatically.
    shift: `float`, `list`of  `floats`, optional, default: 0.0
        Vertical shift of the lines.
    **kwargs
        Other parameters passed to the underlying 1D plotting calls. Common
        aliases such as ``lw``, ``ls``, ``ms``, ``mew``, and ``c`` are
        normalized internally before rendering.

    Returns
    -------
    matplotlib.axes.Axes
        The shared axes containing the overlaid datasets.

    See Also
    --------
    plot_pen
    plot_scatter
    multiplot

    """
    kwargs = normalize_plot_kwargs(kwargs)

    if not is_sequence(datasets):
        # we need a sequence. Else it is a single plot.
        return datasets.plot(
            method=method,
            pen=pen,
            marker=marker,
            color=color,
            ls=ls,
            lw=lw,
            **kwargs,
        )

    def _valid(x, desc):
        if is_sequence(x) and len(x) != len(datasets):
            raise ValueError(
                f"list of {desc} must be of same length as the datasets list",
            )
        if not is_sequence(x) and x != "AUTO":
            return [x] * len(datasets)

        return x

    labels = _valid(labels, "labels")

    for dataset in datasets:
        if dataset._squeeze_ndim > 1:
            raise NotImplementedError(
                "plot multiple is designed to work on "
                "1D dataset only. you may achieved "
                "several plots with "
                "the `clear=False` parameter as a work "
                "around "
                "solution",
            )

    # do not save during this plots, nor apply any commands
    # we will make this when all plots will be done

    kwargs["output"] = None
    kwargs["commands"] = []
    legend = kwargs.pop(
        "legend",
        None,
    )  # remove 'legend' from kwargs before calling plot
    # else it will generate a conflict

    marker = _valid(marker, "marker")
    color = _valid(color, "color")
    ls = _valid(ls, "ls")
    lw = _valid(lw, "lw")
    shift = _valid(shift, "shift")

    # Explicitly create figure and axes once
    # This ensures deterministic behavior without relying on clear=False
    from spectrochempy.plotting.plot_setup import lazy_ensure_mpl_config

    lazy_ensure_mpl_config()

    from spectrochempy.utils.mplutils import get_figure
    from spectrochempy.utils.mplutils import show as mpl_show

    fig = get_figure(
        preferences=kwargs.get("preferences"),
        style=kwargs.get("style"),
        figsize=kwargs.get("figsize"),
        dpi=kwargs.get("dpi"),
    )

    # Create a single axes for all datasets
    ax = fig.add_subplot(1, 1, 1)
    ax.name = "main"

    # Save user's show preference, but suppress during loop
    user_show = kwargs.get("show", True)
    kwargs["show"] = False  # Suppress display during loop

    # Now plot all datasets on the explicit axes
    sh = 0
    for i, s in enumerate(datasets):
        # Apply shift and plot on explicit axes
        # Note: ax is explicitly provided, so clear parameter is ignored
        ax = (s + shift[i] + sh).plot(
            method=method,
            pen=pen,
            marker=(marker[i] if marker != "AUTO" else marker),
            color=color[i] if color != "AUTO" else color,
            ls=ls[i] if ls != "AUTO" else ls,
            lw=lw[i] if lw != "AUTO" else lw,
            ax=ax,  # Explicit axes reuse - clear is ignored when ax is provided
            **kwargs,
        )
        sh += shift[i]

    # Restore the caller's preference for any later code using kwargs.
    kwargs["show"] = user_show

    # Build a combined legend on the shared axes when requested.
    if legend is not None:
        _ = ax.legend(
            ax.lines,
            labels,
            shadow=True,
            loc=legend,
            frameon=True,
            fontsize="small",
        )

    if user_show:
        mpl_show()

    return ax
