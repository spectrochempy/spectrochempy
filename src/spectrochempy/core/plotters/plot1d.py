# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Module containing 1D plotting function(s)."""

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
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter

from spectrochempy.application.preferences import preferences
from spectrochempy.core.dataset.arraymixins.ndplot import (
    NDPlot,  # noqa: F401 # for the docstring to be determined it necessary to import NDPlot
)
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.utils.docutils import docprocess
from spectrochempy.utils.mplutils import make_label
from spectrochempy.utils.typeutils import is_sequence

# --------------------------------------------------------------------------------------
# plot_1D
# --------------------------------------------------------------------------------------


@docprocess.dedent
def plot_1D(dataset, method=None, **kwargs):
    """
    Plot of one-dimensional data.

    Parameters
    ----------
    %(plot.parameters)s

    Other Parameters
    ----------------
    %(plot.other_parameters.no_colorbar|projections|transposed|y_reverse)s

    Returns
    -------
    %(plot.returns)s

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
    # Get preferences
    # ----------------------------------------------------------------------------------
    prefs = preferences

    # before going further, check if the style is passed in the parameters
    style = kwargs.pop("style", None)
    if style is not None:
        prefs.style = style
    # else we assume this has been set before calling plot()

    prefs.set_latex_font(prefs.font.family)  # reset latex settings

    # Redirections ?
    # ------------------------------------------------------------------------
    # should we redirect the plotting to another method
    if dataset._squeeze_ndim > 1:
        return dataset.plot_2D(**kwargs)

    # if plotly execute plotly routine not this one
    if kwargs.get("use_plotly", prefs.use_plotly):
        return dataset.plotly(**kwargs)

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

    # some pen or scatter property
    color = kwargs.get("color", kwargs.get("c", "auto"))
    lw = kwargs.get("linewidth", kwargs.get("lw", "auto"))
    ls = kwargs.get("linestyle", kwargs.get("ls", "auto"))

    marker = kwargs.get("marker", kwargs.get("m", "auto"))
    markersize = kwargs.get("markersize", kwargs.get("ms", prefs.lines_markersize))
    markevery = kwargs.get("markevery", kwargs.get("me", 1))
    markerfacecolor = kwargs.get("markerfacecolor", kwargs.get("mfc", "auto"))
    markeredgecolor = kwargs.get("markeredgecolor", kwargs.get("mec", "k"))

    # Figure setup
    # ------------------------------------------------------------------------
    method = new._figure_setup(ndim=1, method=method, **kwargs)

    pen = "pen" in method or kwargs.pop("pen", False)
    scatter = "scatter" in method or marker != "auto"
    bar = "bar" in method

    ax = new.ndaxes["main"]

    # Other ax properties that can be passed as arguments
    # ------------------------------------------------------------------------
    number_x_labels = prefs.number_of_x_labels
    number_y_labels = prefs.number_of_y_labels
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

    ax.grid(prefs.axes_grid)

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
    if scatter and pen:
        (line,) = ax.plot(
            xdata,
            zdata.T,  # marker = marker,
            markersize=markersize,
            markevery=markevery,
            markeredgewidth=1.0,
            # markerfacecolor = markerfacecolor,
            markeredgecolor=markeredgecolor,
            label=label,
        )
    elif scatter:
        (line,) = ax.plot(
            xdata,
            zdata.T,
            ls="",  # marker = marker,
            markersize=markersize,
            markeredgewidth=1.0,
            markevery=markevery,
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
            label=label,
        )
    elif pen:
        (line,) = ax.plot(xdata, zdata.T, marker="", label=label)

    elif bar:
        # bar only
        line = ax.bar(
            xdata,
            zdata.squeeze(),
            # color=color,
            edgecolor="k",
            align="center",
            label=label,
            width=kwargs.get("width", 0.1),
        )  # barwidth = line[0].get_width()
    else:
        raise ValueError("label not valid")

    if show_complex and pen:
        # add the imaginary component for pen only plot
        zimagdata = new.imag.masked_data
        ax.plot(xdata, zimagdata.T, ls="--")

    if kwargs.get("plot_model", False):
        modeldata = new.modeldata  # TODO: what's about mask?
        ax.plot(
            xdata,
            modeldata.T,
            ls=":",
            lw="2",
            label=label,
        )  # TODO: improve this!!!

    # line attributes
    if pen and not (isinstance(color, str) and color.upper() == "AUTO"):
        # set the color if defined in the preferences or options
        line.set_color(color)

    if pen and not (isinstance(lw, str) and lw.upper() == "AUTO"):
        # set the line width if defined in the preferences or options
        line.set_linewidth(lw)

    if pen and ls.upper() != "AUTO":
        # set the line style if defined in the preferences or options
        line.set_linestyle(ls)

    if scatter and marker.upper() != "AUTO":
        # set the line style if defined in the preferences or options
        line.set_marker(marker)

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
        zl = [np.ma.min(z.masked_data) - amp, np.ma.max(z.masked_data) + amp]

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
        new._plot_resume(dataset, **kwargs)
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

    new._plot_resume(dataset, **kwargs)

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


@docprocess.dedent
def plot_scatter(dataset, **kwargs):
    """
    Plot a 1D dataset as a scatter plot (points can be added on lines).

    Alias of plot (with `method` argument set to `scatter` .

    Parameters
    ----------
    %(plot.parameters.no_method)s

    Other Parameters
    ----------------
    %(plot.other_parameters.no_colorbar|projections|transposed|y_reverse)s

    Returns
    -------
    %(plot.returns)s

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


@docprocess.dedent
def plot_pen(dataset, **kwargs):
    """
    Plot a 1D dataset with solid pen by default.

    Alias of plot (with `method` argument set to `pen`.

    Parameters
    ----------
    %(plot.parameters.no_method)s

    Other Parameters
    ----------------
    %(plot.other_parameters.no_colorbar|projections|transposed|y_reverse)s

    Returns
    -------
    %(plot.returns)s

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


@docprocess.dedent
def plot_scatter_pen(dataset, **kwargs):
    """
    Plot a 1D dataset with solid pen by default.

    Alias of plot (with `method` argument set to `scatter_pen` .

    Parameters
    ----------
    %(plot.parameters.no_method)s

    Other Parameters
    ----------------
    %(plot.other_parameters.no_colorbar|projections|transposed|y_reverse)s

    Returns
    -------
    %(plot.returns)s

    See Also
    --------
    plot
    plot_1D
    plot_scatter
    plot_bar
    plot_pen
    plot_multiple
    multiplot
    """
    return plot_1D(dataset, method="scatter_pen", **kwargs)


@docprocess.dedent
def plot_bar(dataset, **kwargs):
    """
    Plot a 1D dataset with bars.

    Alias of plot (with `method` argument set to `bar`.

    Parameters
    ----------
    %(plot.parameters.no_method)s

    Other Parameters
    ----------------
    %(plot.other_parameters.no_colorbar|projections|transposed|y_reverse)s

    Returns
    -------
    %(plot.returns)s

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
    Plot a series of 1D datasets as a scatter plot with optional lines between markers.

    Parameters
    ----------
    datasets : `list` of 1D `NDDataset`
        NDdatasets to plot.
    method : `str` among [scatter, pen]
        Method to use for plotting.
    pen : bool, optional, default: True
        If method is scatter, this flag tells to draw also the lines
        between the marks.
    labels : a `list` of `str`, optional
        Labels used for the legend. The length of the list must be equal to the number
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
        Other parameters that will be passed to the plot1D function.

    """
    if not is_sequence(datasets):
        # we need a sequence. Else it is a single plot.
        return datasets.plot(**kwargs)

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

    output = kwargs.get("output")
    kwargs["output"] = None
    commands = kwargs.get("commands", [])
    kwargs["commands"] = []
    clear = kwargs.pop("clear", True)
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

    # now we can plot
    sh = 0
    for i, s in enumerate(datasets):  # , colors, markers):
        ax = (s + shift[i] + sh).plot(
            method=method,
            pen=pen,
            marker=(marker[i] if marker != "AUTO" else marker),
            color=color[i] if color != "AUTO" else color,
            ls=ls[i] if ls != "AUTO" else ls,
            lw=lw[i] if lw != "AUTO" else lw,
            clear=clear,
            **kwargs,
        )
        sh += shift[i]
        clear = False  # clear=False is necessary for the next plot to say
        # that we will plot on the same figure

    # scale all plots
    if legend is not None:
        _ = ax.legend(
            ax.lines,
            labels,
            shadow=True,
            loc=legend,
            frameon=True,
            fontsize="small",
        )

    # now we can output the final figure
    kw = {"output": output, "commands": commands}
    datasets[0]._plot_resume(datasets[-1], **kw)

    return ax
