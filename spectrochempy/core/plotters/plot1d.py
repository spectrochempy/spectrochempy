# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================



"""
Module containing 1D plotting function(s)

"""

__all__ = ['plot_1D', 'plot_lines', 'plot_pen', 'plot_scatter', 'plot_bar',
           'plot_multiple']

__dataset_methods__ = ['plot_1D', 'plot_lines', 'plot_pen', 'plot_scatter',
                       'plot_bar']


# ----------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------

import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from spectrochempy.application import log, plotter_preferences, preferences
from spectrochempy.core.plotters.utils import make_label
from spectrochempy.utils import is_sequence, deprecated

# ----------------------------------------------------------------------------
# localimports
# ----------------------------------------------------------------------------

# plot scatter ----------------------------------------------------------------

def plot_scatter(dataset, **kwargs):
    """
    Plot a 1D dataset as a scatter plot (points can be added on lines).

    Alias of plot (with `method` argument set to ``scatter``.

    """
    kwargs['method'] = 'scatter'
    ax = plot_1D(dataset, **kwargs)
    return ax


# plot lines ----------------------------------------------------------------

@deprecated('Use method=pen or plot_pen() instead.')
def plot_lines(dataset, **kwargs):
    """
    Plot a 1D dataset with solid lines by default.

    Alias of plot (with `method` argument set to ``lines``.

    """
    kwargs['method'] = 'lines'
    ax = plot_1D(dataset, **kwargs)
    return ax

# plot pen (default) --------------------------------------------------------

def plot_pen(dataset, **kwargs):
    """
    Plot a 1D dataset with solid pen by default.

    Alias of plot (with `method` argument set to ``pen``.

    """
    kwargs['method'] = 'pen'
    ax = plot_1D(dataset, **kwargs)
    return ax

# plot bars -----------------------------------------------------------------

def plot_bar(dataset, **kwargs):
    """
    Plot a 1D dataset with bars.

    Alias of plot (with `method` argument set to ``bar``.

    """
    kwargs['method'] = 'bar'
    ax = plot_1D(dataset, **kwargs)
    return ax


# plot multiple ----------------------------------------------------------------

def plot_multiple(datasets, method='scatter', pen=True,
                  labels = None, **kwargs):
    """
    Plot a series of 1D datasets as a scatter plot
    with optional lines between markers.

    Parameters
    ----------
    datasets : a list of ndatasets
    method : str among [scatter, pen]
    pen : bool, optional, default:True
        if method is scatter, this flag tells to draw also the lines
        between the marks.
    labels : a list of str, optional
        labels used for the legend.
    **kwargs : other parameters that will be passed to the plot1D function

    """
    if not is_sequence(datasets):
        # we need a sequence. Else it is a single plot.
        return datasets.plot(**kwargs)

    if not is_sequence(labels) or len(labels)!=len(datasets):
        # we need a sequence of labels of same lentgh as datasets
        raise ValueError('the list of labels must be of same length '
                         'as the datasets list')

    for dataset in datasets:
        if dataset.ndim > 1:
            raise NotImplementedError('plot multiple is designed to work on '
                                      '1D dataset only. you may achieved '
                                      'several plots with '
                                      'the `hold` parameter as a work around '
                                      'solution')

    hold = False

    # do not save during this plots, nor apply any commands
    output = kwargs.get('output', None)
    kwargs['output']=None
    commands = kwargs.get('commands', [])
    kwargs['commands'] = []

    for s in datasets : #, colors, markers):

        ax = s.plot(method= method,
                    pen=pen,
                    hold=hold, **kwargs)
        hold = True
        # hold is necessary for the next plot to say
        # that we will plot on the same figure

    # scale all plots
    legend = kwargs.get('legend', None)
    if legend is not None:
        leg = ax.legend(ax.lines, labels, shadow=True, loc=legend,
                        frameon=True, facecolor='lightyellow')
    kw = {'output': output, 'commands': commands}
    datasets[0]._plot_resume(datasets[-1], **kw)

    return ax


# ------------------------------------------------------------------------------
# plot_1D
# ------------------------------------------------------------------------------
def plot_1D(dataset, **kwargs):
    """
    Plot of one-dimensional data

    Parameters
    ----------
    dataset : :class:`~spectrochempy.ddataset.nddataset.NDDataset`
        Source of data to plot.
    method : str, optional, default:pen
        The method can be one among ``pen``, ``bar``,  or ``scatter``
        Default values is ``pen``, i.e., solid lines are drawn.
        To draw a Bar graph, use method: ``bar``.
        For a Scatter plot, use method: ``scatter``.
    twinx : :class:`~matplotlib.Axes` instance, optional, default:None
        If this is not None, then a twin axes will be created with a
        common x dimension.
    title: str
        Title of the plot (or subplot) axe.
    style : str, optional, default = 'notebook'
        Matplotlib stylesheet (use `available_style` to get a list of available
        styles for plotting
    reverse: bool or None [optional, default= None/False
        In principle, coordinates run from left to right, except for wavenumbers
        (*e.g.*, FTIR spectra) or ppm (*e.g.*, NMR), that spectrochempy
        will try to guess. But if reverse is set, then this is the
        setting which will be taken into account.
    hold: bool, optional, default:False
        If true hold the current figure and ax until a new plot is performed.
    data_only: bool, optional, default:False
        Only the plot is done. No addition of axes or label specifications
        (current if any or automatic settings are kept.
    imag: bool, optional, default:False
        Show imaginary part. By default only the real part is displayed.
    show_complex: bool, optional, default:False
        Show both real and imaginary part.
        By default only the real part is displayed.
    dpi: int, optional
        the number of pixel per inches
    figsize: tuple, optional, default is (3.4, 1.7)
        figure size
    fontsize: int, optional
        font size in pixels, default is 10
    imag: bool, optional, default False
        By default real part is shown. Set to True to display the imaginary part
    xlim: tuple, optional
        limit on the horizontal axis
    zlim or ylim: tuple, optional
        limit on the vertical axis
    color or c: matplotlib valid color, optional
        color of the line #TODO: a list if several line
    linewidth or lw: float, optional
        line width
    linestyle or ls: str, optional
        line style definition
    xlabel: str, optional
        label on the horizontal axis
    zlabel or ylabel: str, optional
        label on the vertical axis
    showz: bool, optional, default=True
        should we show the vertical axis
    plot_model:Bool,
        plot model data if available
    modellinestyle or modls: str,
        line style of the model
    offset: float,
        offset of the model individual lines
    commands: str,
        matplotlib commands to be executed
    show_zero: boolean, optional
        show the zero basis
    output: str,
        name of the file to save the figure
    vshift: float, optional
        vertically shift the line from its baseline
    kwargs : additional keywords

    """
    # where to plot?
    # ---------------

    new = dataset.copy()

    # figure setup
    # ------------

    new._figure_setup(**kwargs)

    ax = new.ndaxes['main']

    # Other properties
    # ------------------

    method = kwargs.pop('method','pen')
    is_twinx = kwargs.pop('twinx', None) is not None

    # lines is deprecated
    pen = kwargs.pop('pen',kwargs.pop('lines',False)) # in case it is
                            # scatter we can also show the lines
    pen = method=='pen' or method=='lines' or pen
    scatter = method=='scatter' and not pen
    scatterpen = method=='scatter' and pen
    bar = method=='bar'

    show_complex = kwargs.pop('show_complex', False)

    color = kwargs.get('color', kwargs.get('c', None))    # default to rc
    lw = kwargs.get('linewidth', kwargs.get('lw', None))  # default to rc
    ls = kwargs.get('linestyle', kwargs.get('ls', None))  # default to rc

    marker = kwargs.get('marker', kwargs.get('m', None))  # default to rc
    markersize = kwargs.get('markersize', kwargs.get('ms', 5.))
    markevery = kwargs.get('markevery', kwargs.get('me', 1))
    markerfacecolor = kwargs.get('markerfacecolor', kwargs.get('mfc', None))
    markeredgecolor = kwargs.get('markeredgecolor', kwargs.get('mec', None))

    xscale = kwargs.get('xscale', 'linear')
    yscale = kwargs.get('yscale', 'linear')


    number_x_labels = plotter_preferences.number_of_x_labels  # get from config
    number_y_labels = plotter_preferences.number_of_y_labels
    ax.xaxis.set_major_locator(MaxNLocator(number_x_labels))
    ax.yaxis.set_major_locator(MaxNLocator(number_y_labels))
    ax.xaxis.set_ticks_position('bottom')
    if not is_twinx:
        # do not move these label for twin axes!
        ax.yaxis.set_ticks_position('left')

    # the next lines are to avoid multipliers in axis scale
    formatter = ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xscale(xscale, nonposx='mask')
    ax.set_yscale(yscale, nonposy='mask')

    # ------------------------------------------------------------------------
    # plot the dataset
    # ------------------------------------------------------------------------

    # abscissa axis
    x = new.x

    # take into account the fact that sometimes axis have just labels
    xdata = x.data
    if not np.any(xdata):
        xdata = range(1,len(x.labels)+1)

    # ordinates (by default we plot real part of the data)
    if not kwargs.pop('imag', False) or kwargs.get('show_complex', False):
        z = new.real
        zdata = z.masked_data
    else:
        z = new.imag
        zdata = z.masked_data

    # offset
    offset = kwargs.pop('offset', 0.0)
    zdata = zdata - offset

    # plot_lines
    # -----------------------------

    if scatterpen:
        line, = ax.plot(xdata, zdata,
                        marker = marker,
                        markersize = markersize,
                        markevery = markevery,
                        markeredgewidth = 1.,
                        markerfacecolor = markerfacecolor,
                        markeredgecolor = markeredgecolor)
    elif scatter:
        line, = ax.plot(xdata, zdata,
                        ls = "",
                        marker = marker,
                        markersize = markersize,
                        markeredgewidth = 1.,
                        markevery = markevery,
                        markerfacecolor = markerfacecolor,
                        markeredgecolor = markeredgecolor)
    elif pen:
        line, = ax.plot(xdata, zdata, marker="")

    elif bar:
        line = ax.bar(xdata, zdata, color=color,
                      edgecolor='k', align='center')
        barwidth = line[0].get_width()

    if show_complex and pen:
        zimagdata = new.imag.masked_data
        ax.plot(xdata, zimagdata, ls='--')

    if kwargs.get('plot_model', False):
        modeldata = new.modeldata                 #TODO: what's about mask?
        ax.plot(xdata, modeldata.T, ls=':', lw='2')   #TODO: improve this!!!

    # line attributes
    if pen and color:
        line.set_color(color)

    if pen and lw:
        line.set_linewidth(lw)

    if pen and ls:
        line.set_linestyle(ls)

    # -------------------------------------------------------------------------
    # axis
    # -------------------------------------------------------------------------

    # abscissa limits?
    xl = [xdata[0], xdata[-1]]
    xl.sort()

    if bar or len(x.labels) < number_x_labels+1:
        # extend the axis so that the labels are not too close to the limits
        inc = (xdata[1]-xdata[0])*.5
        xl = [xl[0]-inc, xl[1]+inc]

    # ordinates limits?
    amp = np.ma.ptp(z.masked_data) / 50.
    zl = [np.ma.min(z.masked_data) - amp, np.ma.max(z.masked_data) + amp]

    # check if some data ar not already present on the graph
    # and take care of their limits
    multiplelines = 2 if kwargs.get('show_zero', False) else 1
    if len(ax.lines) > multiplelines:
        # get the previous xlim and zlim
        xlim = list(ax.get_xlim())
        xl[-1] = max(xlim[-1], xl[-1])
        xl[0] = min(xlim[0], xl[0])

        zlim = list(ax.get_ylim())
        zl[-1] = max(zlim[-1], zl[-1])
        zl[0] = min(zlim[0], zl[0])

    xlim = list(kwargs.get('xlim', xl))  # we read the argument xlim
                                         # that should have the priority
    xlim.sort()

    # reversed axis?
    reverse = new.x.is_reversed
    if kwargs.get('reverse', reverse):
        xlim.reverse()

    zlim = list(kwargs.get('zlim', kwargs.get('ylim', zl)))
                                # we read the argument zlim or ylim
                                # which have the priority
    zlim.sort()

    # set the limits
    if not is_twinx:
        # when twin axes, we keep the setting of the first ax plotted
        ax.set_xlim(xlim)
    else:
        ax.tick_params('y', colors=color)

    ax.set_ylim(zlim)


    # -------------------------------------------------------------------------
    # labels
    # -------------------------------------------------------------------------
    if kwargs.get('data_only', False):
        # if data only (we will not set labels
        # it was probably done already in a previous plot
        new._plot_resume(dataset, **kwargs)
        return True

    # x label

    xlabel = kwargs.get("xlabel", None)
    if not xlabel:
        xlabel = make_label(new.x, 'x')
    ax.set_xlabel(xlabel)

    # x tick labels

    uselabel = kwargs.get('uselabel', False)
    if uselabel or not np.any(x.data):
        #TODO refine this to use different orders of labels
        ax.set_xticks(xdata)
        ax.set_xticklabels(x.labels)

    # z label

    zlabel = kwargs.get("zlabel", None)
    if not zlabel:
        zlabel = make_label(new, 'z')

    #ax.set_ylabel(zlabel)

    # do we display the ordinate axis?
    if kwargs.get('show_z', True) and not is_twinx:
        ax.set_ylabel(zlabel)
    elif kwargs.get('show_z', True) and is_twinx:
        ax.set_ylabel(zlabel, color=color)
    else:
        ax.set_yticks([])

    # do we display the zero line
    if kwargs.get('show_zero', False):
        ax.haxlines(label='zero_line')

    new._plot_resume(dataset, **kwargs)

    return ax

if __name__ == '__main__':

    pass