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

"""
from copy import copy

from matplotlib.ticker import MaxNLocator, ScalarFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from spectrochempy.application import app  # must come before plt import
from spectrochempy.plotters.utils import make_label

__all__ = ['plot_map', 'plot_stack', 'plot_image']
_methods = __all__[:]

plotter_preferences = app.plotter_preferences
log = app.log

# =============================================================================
# nddataset plot2D functions
# =============================================================================

# contour map (default) -------------------------------------------------------

def plot_map(source, **kwargs):
    """
    Plot a 2D dataset as a contoured map.

    Alias of plot_2D (with `method` argument set to ``map``.

    """
    kwargs['method'] = 'map'
    ax = plot_2D(source, **kwargs)
    return ax


# stack plot  -----------------------------------------------------------------

def plot_stack(source, **kwargs):
    """
    Plot a 2D dataset as a stacked plot.

    Alias of plot_2D (with `method` argument set to ``stack``).

    """
    kwargs['method'] = 'stack'
    ax = plot_2D(source, **kwargs)
    return ax


# image plot --------------------------------------------------------

def plot_image(source, **kwargs):
    """
    Plot a 2D dataset as an image plot.

    Alias of plot_2D (with `method` argument set to ``image``).

    """
    kwargs['method'] = 'image'
    ax = plot_2D(source, **kwargs)
    return  ax


# generic plot (default stack plot) -------------------------------------------

def plot_2D(source, **kwargs):
    """
    PLot of 2D array.

    Parameters
    ----------
    source: :class:`~spectrochempy.ddataset.nddataset.NDDataset` to plot

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

    """.format(source._general_parameters_doc_)

    # where to plot?
    # --------------

    #mpl.interactive(False)

    # method of plot
    # ------------

    data_only = kwargs.get('data_only', False)

    data_transposed = kwargs.get('data_transposed', False)

    if data_transposed:
        new = source.T  # transpose source
        nameadd='T'
    else:
        new = source.copy()
        nameadd =''

    # figure setup
    # ------------

    new._figure_setup(ndim=2, **kwargs)
    ax = new.ndaxes['main']
    ax.name = ax.name+nameadd

    # Other properties
    # ------------------

    method = kwargs.get('method', plotter_preferences.method_2D)

    colorbar = kwargs.get('colorbar', True)

    cmap = colormap = mpl.rcParams['image.cmap']

    # viridis is the default setting, so we assume that it must be overload here
    # except if style is grayscale which is a particular case.
    styles = kwargs.get('style',[] )
    if styles and not "grayscale" in styles and cmap == 'viridis':

        if method in ['map','image']:
            cmap = colormap = kwargs.get('colormap',
                            kwargs.get('cmap', plotter_preferences.colormap))
        elif data_transposed:
            cmap = colormap = kwargs.get('colormap',
                            kwargs.get('cmap', plotter_preferences.colormap_transposed))
        else:
            cmap = colormap = kwargs.get('colormap',
                            kwargs.get('cmap', plotter_preferences.colormap_stack))


    lw = kwargs.get('linewidth', kwargs.get('lw', plotter_preferences.linewidth))

    alpha = kwargs.get('calpha', plotter_preferences.contour_alpha)

    # -------------------------------------------------------------------------
    # plot the source
    # by default contours are plotted
    # -------------------------------------------------------------------------

    # abscissa axis
    x = new.x.data

    # ordinates axis
    y = new.y.data

    # z intensity (by default we plot real part of the data)
    if not kwargs.get('imag', False):
        z = new.RR.masked_data
    else:
        z = new.RI.masked_data
    zlim = kwargs.get('zlim', (z.min(), z.max()))

    if method in ['map', 'image']:

        zmin, zmax = zlim
        #if not kwargs.get('negative', True):
        zmin = min(zmin, -zmax)
        zmax = max(-zmin, zmax)
        norm = mpl.colors.Normalize(vmin=zmin, vmax=zmax)

    if method in ['map']:

        # contour plot
        # -------------
        if new.clevels is None:
            new.clevels = clevels(z, **kwargs)

        c = ax.contour(x, y, z,
                              new.clevels, linewidths=lw, alpha=alpha)
        c.set_cmap(cmap)
        c.set_norm(norm)

    elif method in ['image']:

        # image plot
        # ----------
        kwargs['nlevels'] = 500
        if new.clevels is None:
            new.clevels = clevels(z, **kwargs)
        c = ax.contourf(x, y, z,
                               new.clevels, linewidths=lw, alpha=alpha)
        c.set_cmap(cmap)
        c.set_norm(norm)

    elif method in ['stack']:

        # stack plot
        # ----------
        normalize = kwargs.get('normalize', None)

        # now plot the collection of lines
        # ---------------------------------
        # map colors using the colormap
        ylim = kwargs.get("ylim", None)

        if ylim is not None:
             vmin, vmax = ylim
        else:
             vmin, vmax = sorted([y[0], y[-1]])
        norm = mpl.colors.Normalize(vmin=vmin,
                                     vmax=vmax)  # we normalize to the max time
        if normalize is not None:
             norm.vmax = normalize

        _colormap = cm = plt.get_cmap(cmap)
        scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=_colormap)

        # we display the line in the reverse order, so that the last
        # are behind the first.

        hold = kwargs.get('hold', False)
        lines = []
        if hold and not data_transposed:
            lines.extend(new.ax.lines)  # keep the old lines

        line0, = ax.plot(x, z[0], lw=lw, picker=True)

        for i in range(z.shape[0]):
            l = copy(line0)
            l.set_ydata(z[i])
            lines.append(l)
            l.set_color(scalarMap.to_rgba(y[i]))
            l.set_label("{:.5f}".format(y[i]))
            l.set_zorder(z.shape[0]+1-i)

        # store the full set of lines
        new._ax_lines = lines[:]

        # but display only a subset of them in order to accelerate the drawing
        maxlines = kwargs.get('maxlines', plotter_preferences.max_lines_in_stack)
        log.debug('max number of lines %d'% maxlines)
        setpy = max(len(new._ax_lines) // maxlines, 1)
        new.ax.lines = new._ax_lines[::setpy]  # displayed ax lines

    if data_only:
        # if data only (we will  ot set axes and labels
        # it was probably done already in a previous plot
        new._plot_resume(source, **kwargs)
        return ax

    # -------------------------------------------------------------------------
    # axis limits and labels
    # -------------------------------------------------------------------------
    # abscissa limits?
    xl = [x.data[0], x.data[-1]]
    xl.sort()
    xlim = list(kwargs.get('xlim', xl))
    xlim.sort()
    xlim[-1] = min(xlim[-1], xl[-1])
    xlim[0] = max(xlim[0], xl[0])

    # reversed x axis?
    # -----------------
    if kwargs.get('x_reverse',
                  kwargs.get('reverse', new.x.is_reversed)):
        xlim.reverse()

    # set the limits
    # ---------------
    ax.set_xlim(xlim)

    # ordinates limits?
    # ------------------
    if method in ['stack']:
        # the z axis info
        # ----------------

        #zl = (np.min(np.ma.min(ys)), np.max(np.ma.max(ys)))
        amp = np.ma.ptp(z)/50.
        zl = (np.min(np.ma.min(z)-amp), np.max(np.ma.max(z))+amp)
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
        ylim = list(kwargs.get('ylim', new.ax.get_ylim()))
        ylim.sort()
        y_reverse = kwargs.get('y_reverse', new.y.is_reversed)
        if y_reverse:
            ylim.reverse()

        # set the limits
        # ----------------
        ax.set_ylim(ylim)

    number_x_labels = plotter_preferences.number_of_x_labels
    number_y_labels = plotter_preferences.number_of_y_labels
    new.ax.xaxis.set_major_locator(MaxNLocator(number_x_labels))
    new.ax.yaxis.set_major_locator(MaxNLocator(number_y_labels))
    # the next two line are to avoid multipliers in axis scale
    y_formatter = ScalarFormatter(useOffset=False)
    new.ax.yaxis.set_major_formatter(y_formatter)
    new.ax.xaxis.set_ticks_position('bottom')
    new.ax.yaxis.set_ticks_position('left')


    # -------------------------------------------------------------------------
    # labels
    # -------------------------------------------------------------------------

    # x label
    # -------
    xlabel = kwargs.get("xlabel", None)
    if not xlabel:
        xlabel = make_label(new.x, 'x')
    ax.set_xlabel(xlabel)

    # y label
    # --------
    ylabel = kwargs.get("ylabel", None)
    if not ylabel:
        if method in ['stack']:
            ylabel = make_label(new, 'z')
        else:
            ylabel = make_label(new.y, 'y')

    # z label
    # --------
    zlabel = kwargs.get("zlabel", None)
    if not zlabel:
        if method in ['stack']:
            zlabel = make_label(new.y, 'y')
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
            axec.name = axec.name+nameadd
            new._axcb = mpl.colorbar.ColorbarBase(axec, cmap=cmap, norm=norm)
            new._axcb.set_label(zlabel)
            # new._axcb.ax.yaxis.set_major_formatter(y_formatter) #this doesn't work
        pass

    # do we display the zero line
    if kwargs.get('show_zero', False):
        ax.haxlines()

    new._plot_resume(source, **kwargs)

    return ax


# =============================================================================
# clevels
# =============================================================================

def clevels(data, **kwargs):
    """Utility function to determine contours levels
    """

    # contours
    maximum = data.max()
    minimum = -maximum

    nlevels = kwargs.get('nlevels', kwargs.get('nc',
                                               plotter_preferences.number_of_contours))
    start = kwargs.get('start', plotter_preferences.contour_start) * maximum
    negative = kwargs.get('negative', True)
    if negative < 0:
        negative = True

    c = np.arange(nlevels)
    cl = np.log(c + 1.)
    clevel = cl * (maximum-start)/cl.max() + start
    clevelneg = - clevel
    if negative:
        clevelc = sorted(list(np.concatenate((clevel,clevelneg))))

    return clevelc

if __name__ == '__main__':

    from spectrochempy.api import NDDataset, scpdata, show

    A = NDDataset.read_omnic('irdata/NH4Y-activation.SPG', directory=scpdata)
    A.y -= A.y[0]
    A.y.to('hour', inplace=True)
    A.y.title = u'Aquisition time'
    ax = A.copy().plot_stack()
    axT = A.copy().plot_stack(data_transposed=True)
    ax2 = A.copy().plot_image(style=['sans', 'paper'], fontsize=9)

    mystyle = {'image.cmap': 'magma',
               'font.size': 10,
               'font.weight': 'bold',
               'axes.grid': True}
    # TODO: store these styles for further use
    A.plot(style=mystyle)
    A.plot(style=['sans', 'paper', 'grayscale'], colorbar=False)
    show()
    pass