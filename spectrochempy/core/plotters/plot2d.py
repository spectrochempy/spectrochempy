# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""

"""
__all__ = ['plot_2D', 'plot_map', 'plot_stack', 'plot_image']

__dataset_methods__ = ['plot_2D', 'plot_map', 'plot_stack', 'plot_image']

# ----------------------------------------------------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------------------------------------------------
from copy import copy

# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------

from matplotlib.ticker import MaxNLocator, ScalarFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# localimports
# ----------------------------------------------------------------------------------------------------------------------

from ...core.plotters.utils import make_label
from ...core import project_preferences, general_preferences, log


# ======================================================================================================================
# nddataset plot2D functions
# ======================================================================================================================

# contour map (default) -------------------------------------------------------

def plot_map(dataset, **kwargs):
    """
    Plot a 2D dataset as a contoured map.

    Alias of plot_2D (with `method` argument set to ``map``.

    """
    kwargs['method'] = 'map'
    ax = plot_2D(dataset, **kwargs)
    return ax


# stack plot  -----------------------------------------------------------------

def plot_stack(dataset, **kwargs):
    """
    Plot a 2D dataset as a stacked plot.

    Alias of plot_2D (with `method` argument set to ``stack``).

    """
    kwargs['method'] = 'stack'
    ax = plot_2D(dataset, **kwargs)
    return ax


# image plot --------------------------------------------------------

def plot_image(dataset, **kwargs):
    """
    Plot a 2D dataset as an image plot.

    Alias of plot_2D (with `method` argument set to ``image``).

    """
    kwargs['method'] = 'image'
    ax = plot_2D(dataset, **kwargs)
    return ax


# generic plot (default stack plot) -------------------------------------------

def plot_2D(dataset, **kwargs):
    """
    PLot of 2D array.

    Parameters
    ----------
    dataset: :class:`~spectrochempy.ddataset.nddataset.NDDataset` to plot

    data_only: `bool` [optional, default=`False`]

        Only the plot is done. No addition of axes or label specifications
        (current if any or automatic settings are kept.

    projections: `bool` [optional, default=False]

    method: str [optional among ``map``, ``stack`` or ``image`` , default=``stack``]

    style : str, optional, default = 'notebook'
        Matplotlib stylesheet (use `available_style` to get a list of available
        styles for plotting

    reverse: `bool` or None [optional, default = None
        In principle, coordinates run from left to right, except for wavenumbers
        (e.g., FTIR spectra) or ppm (e.g., NMR), that spectrochempy
        will try to guess. But if reverse is set, then this is the
        setting which will be taken into account.

    x_reverse: `bool` or None [optional, default= None

    kwargs : additional keywords

    {}

    """.format(dataset._general_parameters_doc_)

    # get all plot preferences
    # ------------------------------------------------------------------------------------------------------------------

    prefs = dataset.plotmeta
    if not prefs.style:
        # not yet set, initialize with default project preferences
        prefs.update(project_preferences.to_dict())

    # If we are in the GUI, we will plot on a widget: but which one?
    # ------------------------------------------------------------------------------------------------------------------

    widget = kwargs.get('widget', None)

    if widget is not None:
        if hasattr(widget, 'implements') and widget.implements('PyQtGraphWidget'):
            # let's go to a particular treament for the pyqtgraph plots
            kwargs['usempl'] = usempl = False
            # we try to have a commmon interface for both plot library
            kwargs['ax'] = ax = widget  # return qt_plot_1D(dataset, **kwargs)
        else:
            # this must be a matplotlibwidget
            kwargs['usempl'] = usempl = True
            fig = widget.fig
            kwargs['ax'] = ax = fig.gca()

    # method of plot
    # ------------------------------------------------------------------------------------------------------------------

    method = kwargs.get('method', prefs.method_2D)

    data_only = kwargs.get('data_only', False)

    data_transposed = kwargs.get('data_transposed', False)

    if data_transposed:
        new = dataset.T  # transpose dataset
        nameadd = 'T'
    else:
        new = dataset.copy()  # TODO: why loose time to make a copy?
        nameadd = ''

    # figure setup
    # ------------------------------------------------------------------------------------------------------------------

    new._figure_setup(ndim=2, **kwargs)
    ax = new.ndaxes['main']
    ax.name = ax.name + nameadd

    # Other properties
    # ------------------------------------------------------------------------------------------------------------------

    colorbar = kwargs.get('colorbar', project_preferences.colorbar)

    cmap = mpl.rcParams['image.cmap']

    # viridis is the default setting,
    # so we assume that it must be overwritten here
    # except if style is grayscale which is a particular case.
    styles = kwargs.get('style', project_preferences.style)

    if styles and not "grayscale" in styles and cmap == "viridis":

        if method in ['map', 'image']:
            cmap = colormap = kwargs.get('colormap',
                                         kwargs.get('cmap', project_preferences.colormap))
        elif data_transposed:
            cmap = colormap = kwargs.get('colormap',
                                         kwargs.get('cmap', project_preferences.colormap_transposed))
        else:
            cmap = colormap = kwargs.get('colormap',
                                         kwargs.get('cmap', project_preferences.colormap_stack))

    lw = kwargs.get('linewidth', kwargs.get('lw',
                                            project_preferences.pen_linewidth))

    alpha = kwargs.get('calpha', project_preferences.contour_alpha)

    number_x_labels = project_preferences.number_of_x_labels
    number_y_labels = project_preferences.number_of_y_labels
    ax.xaxis.set_major_locator(MaxNLocator(number_x_labels))
    ax.yaxis.set_major_locator(MaxNLocator(number_y_labels))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # the next lines are to avoid multipliers in axis scale
    formatter = ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # ------------------------------------------------------------------------------------------------------------------
    # Set axis
    # ------------------------------------------------------------------------------------------------------------------

    # set the abscissa axis
    # ------------------------------------------------------------------------------------------------------------------
    # the actual dimension name is the last in the new.dims list
    dimx = new.dims[-1]
    x = getattr(new, dimx)
    xsize = new.shape[-1]
    if x is not None and (not x.is_empty or x.is_labeled):
        xdata = x.data
        discrete_data = False
        if not np.any(xdata):
            if x.labels:
                discrete_data = True
                # take into account the fact that sometimes axis have just labels
                xdata = range(1, len(x.labels) + 1)
    else:
        xdata = range(xsize)

    xl = [xdata[0], xdata[-1]]
    xl.sort()

    if xsize < number_x_labels + 1:
        # extend the axis so that the labels are not too close to the limits
        inc = abs(xdata[1] - xdata[0]) * .5
        xl = [xl[0] - inc, xl[1] + inc]

    if data_only:
        xl = ax.get_xlim()

    xlim = list(kwargs.get('xlim', xl))
    xlim.sort()
    xlim[-1] = min(xlim[-1], xl[-1])
    xlim[0] = max(xlim[0], xl[0])

    if kwargs.get('x_reverse', kwargs.get('reverse', x.reversed if x else False)):
        xlim.reverse()

    ax.set_xlim(xlim)

    # set the ordinates axis
    # ------------------------------------------------------------------------------------------------------------------
    # the actual dimension name is the second in the new.dims list
    dimy = new.dims[-2]
    y = getattr(new, dimy)
    ysize = new.shape[-2]
    if y is not None and (not y.is_empty or y.is_labeled):
        ydata = y.data
        if not np.any(ydata):
            if x.labels:
                ydata = range(1, len(y.labels) + 1)
    else:
        ydata = range(ysize)

    yl = [ydata[0], ydata[-1]]
    yl.sort()

    if ysize < number_y_labels + 1:
        # extend the axis so that the labels are not too close to the limits
        inc = abs(ydata[1] - ydata[0]) * .5
        yl = [yl[0] - inc, yl[1] + inc]

    if data_only:
        yl = ax.get_ylim()

    ylim = list(kwargs.get("ylim", yl))
    ylim.sort()
    ylim[-1] = min(ylim[-1], yl[-1])
    xlim[0] = max(ylim[0], yl[0])

    # z intensity (by default we plot real part of the data)
    # ------------------------------------------------------------------------------------------------------------------

    if not kwargs.get('imag', False):
        zdata = new.RR.masked_data
    else:
        zdata = new.RI.masked_data

    zlim = kwargs.get('zlim', (zdata.min(), zdata.max()))

    if method in ['stack']:

        # the z axis info
        # ---------------
        # zl = (np.min(np.ma.min(ys)), np.max(np.ma.max(ys)))
        amp = np.ma.ptp(zdata) / 50.
        zl = (np.min(np.ma.min(zdata) - amp), np.max(np.ma.max(zdata)) + amp)
        zlim = list(kwargs.get('zlim', zl))
        zlim.sort()
        z_reverse = kwargs.get('z_reverse', False)
        if z_reverse:
            zlim.reverse()

        # set the limits
        # ---------------
        ax.set_ylim(zlim)

    else:

        # the y axis info
        # ----------------
        if data_only:
            ylim = ax.get_ylim()

        ylim = list(kwargs.get('ylim', ylim))
        ylim.sort()
        y_reverse = kwargs.get('y_reverse', x.reversed if y else False)
        if y_reverse:
            ylim.reverse()

        # set the limits
        # ----------------
        ax.set_ylim(ylim)

    # ------------------------------------------------------------------------------------------------------------------
    # plot the dataset
    # by default contours are plotted
    # ------------------------------------------------------------------------------------------------------------------
    normalize = kwargs.get('normalize', None)

    if method in ['map', 'image']:
        zmin, zmax = zlim
        zmin = min(zmin, -zmax)
        zmax = max(-zmin, zmax)
        norm = mpl.colors.Normalize(vmin=zmin, vmax=zmax)

    if method in ['image']:
        if discrete_data:
            method = 'map'
        else:
            # image plot
            # ----------
            kwargs['nlevels'] = 500
            if new.clevels is None:
                new.clevels = clevels(zdata, **kwargs)
            c = ax.contourf(xdata, ydata, zdata,
                            new.clevels, linewidths=lw, alpha=alpha)
            c.set_cmap(cmap)
            c.set_norm(norm)

    if method in ['map']:

        if discrete_data:

            _colormap = cm = plt.get_cmap(cmap)
            scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=_colormap)

            marker = kwargs.get('marker', kwargs.get('m', None))
            markersize = kwargs.get('markersize', kwargs.get('ms', 5.))
            markevery = kwargs.get('markevery', kwargs.get('me', 1))

            for i in ydata:
                for j in xdata:
                    l, = ax.plot(j, i, lw=lw, marker='o',
                                 markersize=markersize)
                    l.set_color(scalarMap.to_rgba(zdata[i - 1, j - 1]))


        else:
            # contour plot
            # -------------
            if new.clevels is None:
                new.clevels = clevels(zdata, **kwargs)

            c = ax.contour(xdata, ydata, zdata,
                           new.clevels, linewidths=lw, alpha=alpha)
            c.set_cmap(cmap)
            c.set_norm(norm)


    elif method in ['stack']:

        # stack plot
        # ----------

        # now plot the collection of lines
        # --------------------------------
        # map colors using the colormap

        vmin, vmax = ylim
        norm = mpl.colors.Normalize(vmin=vmin,
                                    vmax=vmax)  # we normalize to the max time
        if normalize is not None:
            norm.vmax = normalize

        _colormap = cm = plt.get_cmap(cmap)
        scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=_colormap)

        # we display the line in the reverse order, so that the last
        # are behind the first.

        clear = kwargs.get('clear', True)
        lines = []
        if not clear and not data_transposed:
            lines.extend(ax.lines)  # keep the old lines

        line0, = ax.plot(xdata, zdata[0], lw=lw, picker=True)

        for i in range(zdata.shape[0]):
            l = copy(line0)
            l.set_ydata(zdata[i])
            lines.append(l)
            l.set_color(scalarMap.to_rgba(ydata[i]))
            l.set_label("{:.5f}".format(ydata[i]))
            l.set_zorder(zdata.shape[0] + 1 - i)

        # store the full set of lines
        new._ax_lines = lines[:]

        # but display only a subset of them in order to accelerate the drawing
        maxlines = kwargs.get('maxlines',
                              general_preferences.max_lines_in_stack)
        log.debug('max number of lines %d' % maxlines)
        setpy = max(len(new._ax_lines) // maxlines, 1)
        ax.lines = new._ax_lines[::setpy]  # displayed ax lines

    if data_only:
        # if data only (we will  ot set axes and labels
        # it was probably done already in a previous plot
        new._plot_resume(dataset, **kwargs)
        return ax

    # ------------------------------------------------------------------------------------------------------------------
    # labels
    # ------------------------------------------------------------------------------------------------------------------

    # x label
    # ------------------------------------------------------------------------------------------------------------------
    xlabel = kwargs.get("xlabel", None)
    if not xlabel:
        xlabel = make_label(x, 'x') if x else ''
    ax.set_xlabel(xlabel)

    # x tick labels

    uselabelx = kwargs.get('uselabel_x', False)
    if x and x.is_labeled and (uselabelx or not np.any(x.data)) and len(x.labels) < number_x_labels + 1:
        # TODO refine this to use different orders of labels
        ax.set_xticks(xdata)
        ax.set_xticklabels(x.labels)

    # y label
    # ------------------------------------------------------------------------------------------------------------------
    ylabel = kwargs.get("ylabel", None)
    if not ylabel:
        if method in ['stack']:
            ylabel = make_label(new, 'z')

        else:
            ylabel = make_label(y, 'y') if y else ''
            # y tick labels
            uselabely = kwargs.get('uselabel_y', False)
            if y and y.is_labeled and (uselabely or not np.any(y.data)) and len(y.labels) < number_y_labels:
                # TODO refine this to use different orders of labels
                ax.set_yticks(ydata)
                ax.set_yticklabels(y.labels)

    # z label
    # ------------------------------------------------------------------------------------------------------------------
    zlabel = kwargs.get("zlabel", None)
    if not zlabel:
        if method in ['stack']:
            zlabel = make_label(y, 'y') if y else ''
        else:
            zlabel = make_label(new, 'z')

    # do we display the ordinate axis?
    if kwargs.get('show_y', True):
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticks([])

    if colorbar:

        if not new._axcb:
            axec = new.ndaxes['colorbar']
            axec.name = axec.name + nameadd
            new._axcb = mpl.colorbar.ColorbarBase(axec, cmap=cmap, norm=norm)
            new._axcb.set_label(zlabel)
            # new._axcb.ax.yaxis.set_major_formatter(y_formatter)
            # #this doesn't work
        pass

    # do we display the zero line
    if kwargs.get('show_zero', False):
        ax.haxlines()

    # display a title
    # ------------------------------------------------------------------------------------------------------------------
    title = kwargs.get('title', None)
    if title:
        ax.set_title(title)

    new._plot_resume(dataset, **kwargs)

    return ax


# ======================================================================================================================
# clevels
# ======================================================================================================================

def clevels(data, **kwargs):
    """Utility function to determine contours levels
    """

    # contours
    maximum = data.max()
    minimum = -maximum

    nlevels = kwargs.get('nlevels', kwargs.get('nc',
                                               project_preferences.number_of_contours))
    start = kwargs.get('start', project_preferences.contour_start) * maximum
    negative = kwargs.get('negative', True)
    if negative < 0:
        negative = True

    c = np.arange(nlevels)
    cl = np.log(c + 1.)
    clevel = cl * (maximum - start) / cl.max() + start
    clevelneg = - clevel
    if negative:
        clevelc = sorted(list(np.concatenate((clevel, clevelneg))))

    return clevelc


if __name__ == '__main__':
    pass
