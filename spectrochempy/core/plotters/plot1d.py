# -*- coding: utf-8 -*-

#
# =============================================================================
# Copyright (©) 2015-2022 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================
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

from spectrochempy.utils import make_label, is_sequence, add_docstring  # , deprecated
from spectrochempy.core.dataset.coord import Coord


_PLOT1D_DOC = """\
widget : Matplotlib or PyQtGraph widget (for GUI only)
    The widget where to plot in the GUI application. This is not used if
    plots are made in jupyter notebook.
twinx : :class:`~matplotlib.Axes` instance, optional, default: None
    If this is not None, then a twin axes will be created with a
    common x dimension.
title : str
    Title of the plot (or subplot) axe.
style : str, optional, default='notebook'
    Matplotlib stylesheet (use `available_style` to get a list of available
    styles for plotting.
reverse : bool or None [optional, default=None/False
    In principle, coordinates run from left to right,
    except for wavenumbers
    (*e.g.*, FTIR spectra) or ppm (*e.g.*, NMR), that spectrochempy
    will try to guess. But if reverse is set, then this is the
    setting which will be taken into account.
clear : bool, optional, default: True
    If false, hold the current figure and ax until a new plot is performed.
data_only : bool, optional, default: False
    Only the plot is done. No addition of axes or label specifications.
imag : bool, optional, default: False
    Show imaginary component. By default only the real component is
    displayed.
show_complex : bool, optional, default: False
    Show both real and imaginary component.
    By default only the real component is displayed.
dpi : int, optional
    the number of pixel per inches.
figsize : tuple, optional, default is (3.4, 1.7)
    figure size.
fontsize : int, optional
    font size in pixels, default is 10.
imag : bool, optional, default False
    By default real component is shown.
    Set to True to display the imaginary component
xlim : tuple, optional
    limit on the horizontal axis.
zlim or ylim : tuple, optional
    limit on the vertical axis.
color or c : matplotlib valid color, optional
    color of the line.  # TODO : a list if several line
linewidth or lw : float, optional
    line width.
linestyle or ls : str, optional
    line style definition.
xlabel : str, optional
    label on the horizontal axis.
zlabel or ylabel : str, optional
    label on the vertical axis.
showz : bool, optional, default=True
    should we show the vertical axis.
plot_model : Bool,
    plot model data if available.
modellinestyle or modls : str
    line style of the model.
offset : float
    offset of the model individual lines.
commands : str,
    matplotlib commands to be executed.
show_zero : boolean, optional
    show the zero basis.
output : str,
    name of the file to save the figure.
vshift : float, optional
    vertically shift the line from its baseline.
"""


# plot scatter ---------------------------------------------------------------
@add_docstring(_PLOT1D_DOC)
def plot_scatter(dataset, **kwargs):
    """
    Plot a 1D dataset as a scatter plot (points can be added on lines).

    Alias of plot (with `method` argument set to ``scatter``.

    Parameters
    ----------
    dataset : :class:`~spectrochempy.ddataset.nddataset.NDDataset`
        Source of data to plot.
    **kwargs : dict
        See other parameters.

    Other Parameters
    ----------------
    {0}

    See Also
    --------
    plot_1D
    plot_pen
    plot_bar
    plot_scatter_pen
    plot_multiple
    """
    if kwargs.get("use_plotly", False):
        return dataset.plotly(**kwargs)  # pragma: no cover
    else:
        return plot_1D(dataset, method="scatter", **kwargs)


# plot pen (default) ---------------------------------------------------------


@add_docstring(_PLOT1D_DOC)
def plot_pen(dataset, **kwargs):
    """
    Plot a 1D dataset with solid pen by default.

    Alias of plot (with `method` argument set to ``pen``.

    Parameters
    ----------
    dataset : :class:`~spectrochempy.ddataset.nddataset.NDDataset`
        Source of data to plot.
    **kwargs : dict
        See other parameters.

    Other Parameters
    ----------------
    {0}

    See Also
    --------
    plot_1D
    plot_scatter
    plot_bar
    plot_scatter_pen
    plot_multiple
    """
    if kwargs.get("use_plotly", False):
        return dataset.plotly(**kwargs)  # pragma: no cover
    else:
        return plot_1D(dataset, method="pen", **kwargs)


# plot pen (default) ---------------------------------------------------------


@add_docstring(_PLOT1D_DOC)
def plot_scatter_pen(dataset, **kwargs):
    """
    Plot a 1D dataset with solid pen by default.

    Alias of plot (with `method` argument set to ``scatter_pen``.

    Parameters
    ----------
    dataset : :class:`~spectrochempy.ddataset.nddataset.NDDataset`
        Source of data to plot.
    **kwargs : dict
        See other parameters.

    Other Parameters
    ----------------
    {0}

    See Also
    --------
    plot_1D
    plot_pen
    plot_scatter
    plot_bar
    plot_multiple
    """
    if kwargs.get("use_plotly", False):
        return dataset.plotly(**kwargs)  # pragma: no cover
    else:
        return plot_1D(dataset, method="scatter+pen", **kwargs)


# plot bars ------------------------------------------------------------------


@add_docstring(_PLOT1D_DOC)
def plot_bar(dataset, **kwargs):
    """
    Plot a 1D dataset with bars.

    Alias of plot (with `method` argument set to ``bar``.

    Parameters
    ----------
    dataset : :class:`~spectrochempy.ddataset.nddataset.NDDataset`
        Source of data to plot.
    **kwargs : dict
        See other parameters.

    Other Parameters
    ----------------
    {0}

    See Also
    --------
    plot_1D
    plot_pen
    plot_scatter
    plot_scatter_pen
    plot_multiple
    """
    if kwargs.get("use_plotly", False):
        return dataset.plotly(**kwargs)  # pragma: no cover
    else:
        return plot_1D(dataset, method="bar", **kwargs)


# plot multiple --------------------------------------------------------------


@add_docstring(_PLOT1D_DOC)
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
def plot_1D(dataset, method="pen", **kwargs):
    """
    Plot of one-dimensional data.

    Parameters
    ----------
    dataset : :class:`~spectrochempy.ddataset.nddataset.NDDataset`
        Source of data to plot.
    method : str, optional, default: pen
        The method can be one among ``pen``, ``bar``,  or ``scatter``
        Default values is ``pen``, i.e., solid lines are drawn.
        To draw a Bar graph, use method : ``bar``.
        For a Scatter plot, use method : ``scatter``.
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

    # Method of plot
    # ------------------------------------------------------------------------

    # some addtional options may exists in kwargs
    pen = kwargs.pop("pen", False)  # lines and pen synonyms
    scatter = kwargs.pop("scatter", False)

    # final choice of method
    pen = (method == "pen") or pen
    scatter = (method == "scatter" and not pen) or scatter
    scatterpen = ((method == "scatter" or scatter) and pen) or (method == "scatter+pen")
    bar = method == "bar"

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

    # Figure setup  #
    # ------------------------------------------------------------------------
    new._figure_setup(ndim=1, scatter=scatter, scatterpen=scatterpen, **kwargs)

    ax = new.ndaxes["main"]

    # If no method parameters was provided when this function was called,
    # we first look in the meta parameters of the dataset for the defaults

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
    if scatterpen:
        # pen + scatter
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
        # scatter only
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
        # pen only
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
        # add the imaginaly component for pen only plot
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
    if (pen or scatterpen) and not (isinstance(color, str) and color.upper() == "AUTO"):
        # set the color if defined in the preferences or options
        line.set_color(color)

    if (pen or scatterpen) and not (isinstance(lw, str) and lw.upper() == "AUTO"):
        # set the line width if defined in the preferences or options
        line.set_linewidth(lw)

    if (pen or scatterpen) and ls.upper() != "AUTO":
        # set the line style if defined in the preferences or options
        line.set_linestyle(ls)

    if (scatter or scatterpen) and marker.upper() != "AUTO":
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

    uselabelx = kwargs.get("uselabel_x", False)
    if (
        x
        and x.is_labeled
        and (uselabelx or not np.any(x.data))
        and len(x.labels) < number_x_labels + 1
    ):
        # TODO refine this to use different orders of labels
        ax.set_xticks(xdata)
        ax.set_xticklabels(x.labels)

    # z label

    zlabel = kwargs.get("zlabel", None)
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
