# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
Module containing 1D plotting function(s).
"""

__all__ = [
    "plot_1D",
    "plot_pen",
    "plot_scatter",
    "plot_bar",
    "plot_multiple",
    "plot_scatter_pen",
]

__dataset_methods__ = [
    "plot_1D",
    "plot_lines",
    "plot_pen",
    "plot_scatter",
    "plot_bar",
    "plot_scatter_pen",
]

import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from spectrochempy.utils import (
    make_label,
    is_sequence,
    add_docstring,
    plot_method,
)  # , deprecated
from spectrochempy.core.dataset.coord import Coord


_PLOT1D_DOC = """\
ax : Axe, optional
    Axe where to plot. If not specified, create a new one.
style : str, optional, default: `dataset.preferences.style` (scpy)
    Matplotlib stylesheet (use `available_style` to get a list of available
    styles for plotting.
use_plotly : bool, optional, default: `preferences.use_plotly` (False)
    Should we use plotly instead of matplotlib for plotting.
twinx : :class:`~matplotlib.Axes` instance, optional, default: None
    If this is not None, then a twin axes will be created with a
    common x dimension.
clear : bool, optional, default: True
    If false, hold the current figure and ax until a new plot is performed.
reverse : bool or None [optional, default=None/False
    In principle, coordinates run from left to right,
    except for wavenumbers
    (*e.g.*, FTIR spectra) or ppm (*e.g.*, NMR), that spectrochempy
    will try to guess. But if reverse is set, then this is the
    setting which will be taken into account.
data_only : bool, optional, default: False
    Only the plot is done. No addition of axes or label specifications.
imag : bool, optional, default: False
    Show imaginary component for complex data. By default the real component is
    displayed.
show_complex : bool, optional, default: False
    Show both real and imaginary component for complex data.
    By default only the real component is displayed.
figsize : tuple, optional, default is (3.4, 1.7)
    figure size.
dpi : int, optional
    the number of pixel per inches.
xlim : tuple, optional
    limit on the horizontal axis.
zlim or ylim : tuple, optional
    limit on the vertical axis.
color or c : color, optional, default: auto
    color of the line.
linewidth or lw : float, optional, default: auto
    line width.
linestyle or ls : str, optional, default: auto
    line style definition.
marker, m: str, optional, default: auto
    marker type for scatter plot. If marker != "" then the scatter type of plot is chosen automatically.
markeredgecolor or mec: color, optional
markeredgewidth or mew: float, optional
markerfacecolor or mfc: color, optional
markersize or ms: float, optional
markevery: None or int
title : str
    Title of the plot (or subplot) axe.
plottitle: bool, optional, default: False
    Use the name of the dataset as title. Works only if title is not defined
xlabel : str, optional
    label on the horizontal axis.
zlabel or ylabel : str, optional
    label on the vertical axis.
uselabel_x: bool, optional
    use x coordinate label as x tick labels
show_z : bool, optional, default: True
    should we show the vertical axis.
show_zero : bool, optional
    show the zero basis.
show_mask: bool, optional
    Should we display the mask using colored area.
plot_model : Bool,
    plot model data if available.
modellinestyle or modls : str
    line style of the model.
offset : float
    offset of the model individual lines.
commands : str,
    matplotlib commands to be executed.
output : str,
    name of the file to save the figure.
vshift : float, optional
    vertically shift the line from its baseline.
"""


@plot_method("1D", _PLOT1D_DOC)
def plot_scatter(dataset, **kwargs):
    """
    Plot a 1D dataset as a scatter plot (points can be added on lines).

    Alias of plot (with `method` argument set to ``scatter``.
    """


@plot_method("1D", _PLOT1D_DOC)
def plot_pen(dataset, **kwargs):
    """
    Plot a 1D dataset with solid pen by default.

    Alias of plot (with `method` argument set to ``pen``.
    """


@plot_method("1D", _PLOT1D_DOC)
def plot_scatter_pen(dataset, **kwargs):
    """
    Plot a 1D dataset with solid pen by default.

    Alias of plot (with `method` argument set to ``scatter_pen``.
    """


@plot_method("1D", _PLOT1D_DOC)
def plot_bar(dataset, **kwargs):
    """
    Plot a 1D dataset with bars.

    Alias of plot (with `method` argument set to ``bar``.
    """


def plot_multiple(datasets, method="scatter", pen=True, labels=None, **kwargs):
    """
    Plot a series of 1D datasets as a scatter plot with optional lines between markers.

    Parameters
    ----------
    datasets : a list of ndatasets
    method : str among [scatter, pen]
    pen : bool, optional, default: True
        If method is scatter, this flag tells to draw also the lines
        between the marks.
    labels : a list of str, optional
        Labels used for the legend.
    **kwargs : dic
        Other parameters that will be passed to the plot1D function.

    Other Parameters
    ----------------
    {0}

    See Also
    --------
    plot_1D
    plot_pen
    plot_scatter
    plot_bar
    plot_scatter_pen
    """
    if not is_sequence(datasets):
        # we need a sequence. Else it is a single plot.
        return datasets.plot(**kwargs)

    if not is_sequence(labels) or len(labels) != len(datasets):
        # we need a sequence of labels of same lentgh as datasets
        raise ValueError(
            "the list of labels must be of same length " "as the datasets list"
        )

    for dataset in datasets:
        if dataset._squeeze_ndim > 1:
            raise NotImplementedError(
                "plot multiple is designed to work on "
                "1D dataset only. you may achieved "
                "several plots with "
                "the `clear=False` parameter as a work "
                "around "
                "solution"
            )

    # do not save during this plots, nor apply any commands
    # we will make this when all plots will be done

    output = kwargs.get("output", None)
    kwargs["output"] = None
    commands = kwargs.get("commands", [])
    kwargs["commands"] = []
    clear = kwargs.pop("clear", True)
    legend = kwargs.pop(
        "legend", None
    )  # remove 'legend' from kwargs before calling plot
    # else it will generate a conflict

    for s in datasets:  # , colors, markers):

        ax = s.plot(
            method=method,
            pen=pen,
            marker="AUTO",
            color="AUTO",
            ls="AUTO",
            clear=clear,
            **kwargs
        )
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
            facecolor="lightyellow",
        )

    # now we can output the final figure
    kw = {"output": output, "commands": commands}
    datasets[0]._plot_resume(datasets[-1], **kw)

    return ax


# ------------------------------------------------------------------
# plot_1D
# ------------------------------------------------------------------


@add_docstring(_PLOT1D_DOC)
def plot_1D(dataset, method=None, **kwargs):
    """
    Plot of one-dimensional data.

    Parameters
    ----------
    dataset : :class:`~spectrochempy.ddataset.nddataset.NDDataset`
        Source of data to plot.
    method : str, optional, default: dataset.preference.method_1D
        The method can be one among ``pen``, ``bar``, ``scatter`` or ``scatter+pen``.
        Default values is ``pen``, i.e., solid lines are drawn. This default can be changed
        using ``dataset.preference.method_1D``.
        To draw a Bar graph, use method ``bar``.
        For a Scatter plot, use method ``scatter``.
        For pen and scatter simultaneously, use method ``scatter+pen``.
    **kwargs : dict
        See other parameters.

    Other Parameters
    ----------------
    {0}

    See Also
    --------
    plot_pen
    plot_scatter
    plot_bar
    plot_scatter_pen
    plot_multiple
    """

    # Get preferences
    # ------------------------------------------------------------------------

    prefs = dataset.preferences

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
        # dont' apply to array of size one to preserve the x coordinate!!!!
        new = new.squeeze()

    # is that a plot with twin axis
    is_twinx = kwargs.pop("twinx", None) is not None

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

    pen = "pen" in method
    scatter = "scatter" in method or marker
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

    # ------------------------------------------------------------------------
    # plot the dataset
    # ------------------------------------------------------------------------

    # abscissa axis
    # the actual dimension name is the first in the new.dims list
    dimx = new.dims[-1]
    x = getattr(new, dimx)
    if x is not None and x.implements("CoordSet"):
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
        if not np.any(xdata):
            if x.is_labeled:
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
    label = kwargs.get("label", None)
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
            color=color,
            edgecolor="k",
            align="center",
            label=label,
        )  # barwidth = line[0].get_width()
    else:
        raise ValueError("label not valid")

    if show_complex and pen:
        # add the imaginary component for pen only plot
        if new.is_quaternion:
            zimagdata = new.RI.masked_data
        else:
            zimagdata = new.imag.masked_data
        ax.plot(xdata, zimagdata.T, ls="--")

    if kwargs.get("plot_model", False):
        modeldata = new.modeldata  # TODO: what's about mask?
        ax.plot(
            xdata, modeldata.T, ls=":", lw="2", label=label
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

    # ------------------------------------------------------------------------
    # axis
    # ------------------------------------------------------------------------

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
        # if data only (we will  ot set axes and labels
        # it was probably done already in a previous plot
        new._plot_resume(dataset, **kwargs)
        return ax

    # ------------------------------------------------------------------------
    # labels
    # ------------------------------------------------------------------------

    # x label

    xlabel = kwargs.get("xlabel", None)
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

    zlabel = kwargs.get("zlabel", kwargs.get("ylabel", None))
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
    title = kwargs.get("title", None)
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


if __name__ == "__main__":
    pass
