# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================

#

"""

"""
import sys
from copy import copy

from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from spectrochempy.application import plotoptions, log
from spectrochempy.core.plotters.utils import make_label
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import SpectroChemPyWarning

__all__ = ['plot_2D', 'plot_map', 'plot_stack', 'plot_image']
_methods = __all__[:]


# =============================================================================
# nddataset plot2D functions
# =============================================================================

# contour map (default) -------------------------------------------------------

def plot_map(source, **kwargs):
    """
    Plot a 2D dataset as a contoured map.

    Alias of plot_2D (with `kind` argument set to ``map``.

    """
    kwargs['kind'] = 'map'
    ax = plot_2D(source, **kwargs)
    return ax


# stack plot  -----------------------------------------------------------------

def plot_stack(source, **kwargs):
    """
    Plot a 2D dataset as a stacked plot.

    Alias of plot_2D (with `kind` argument set to ``stack``).

    """
    kwargs['kind'] = 'stack'
    ax = plot_2D(source, **kwargs)
    return ax


# image plot --------------------------------------------------------

def plot_image(source, **kwargs):
    """
    Plot a 2D dataset as an image plot.

    Alias of plot_2D (with `kind` argument set to ``image``).

    """
    kwargs['kind'] = 'image'
    ax = plot_2D(source, **kwargs)
    return  ax


# generic plot (default stack plot) -------------------------------------------

def plot_2D(source, **kwargs):
    """
    PLot of 2D array.

    Parameters
    ----------
    source: :class:`~spectrochempy.core.ddataset.nddataset.NDDataset` to plot

    data_only: `bool` [optional, default=`False`]

        Only the plot is done. No addition of axes or label specifications
        (current if any or automatic settings are kept.

    projections: `bool` [optional, default=False]

    kind: `str` [optional among ``map``, ``stack`` or ``image`` , default=``stack``]

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

    mpl.interactive(False)

    # kind of plot
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

    kind = kwargs.get('kind', plotoptions.kind_2D)

    colorbar = kwargs.get('colorbar', True)

    if kind in ['map','image']:
        cmap = colormap = kwargs.get('colormap',
                        kwargs.get('cmap', plotoptions.colormap))
    elif data_transposed:
        cmap = colormap = kwargs.get('colormap',
                        kwargs.get('cmap', plotoptions.colormap_transposed))
    else:
        cmap = colormap = kwargs.get('colormap',
                        kwargs.get('cmap', plotoptions.colormap_stack))

    lw = kwargs.get('linewidth', kwargs.get('lw', plotoptions.linewidth))

    alpha = kwargs.get('calpha', plotoptions.contour_alpha)

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

    if kind in ['map', 'image']:
        zmin, zmax = zlim
        #if not kwargs.get('negative', True):
        zmin = min(zmin, -zmax)
        zmax = max(-zmin, zmax)
        norm = mpl.colors.Normalize(vmin=zmin, vmax=zmax)

    if kind in ['map']:

        # contour plot
        # -------------
        if new.clevels is None:
            new.clevels = clevels(z, **kwargs)

        c = ax.contour(x, y, z,
                              new.clevels, linewidths=lw, alpha=alpha)
        c.set_cmap(cmap)
        c.set_norm(norm)

    elif kind in ['image']:

        # image plot
        # ----------
        kwargs['nlevels'] = 500
        if new.clevels is None:
            new.clevels = clevels(z, **kwargs)
        c = ax.contourf(x, y, z,
                               new.clevels, linewidths=lw, alpha=alpha)
        c.set_cmap(cmap)
        c.set_norm(norm)

    elif kind in ['stack']:

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

        _colormap = cm = plt.get_cmap(colormap)
        scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=_colormap)

        # we display the line in the reverse order, so that the last
        # are behind the first.
        line0, = ax.plot(x, z[0], lw=lw, picker=True)
        lines = []

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
        maxlines = kwargs.get('maxlines', plotoptions.max_lines_in_stack)
        setpy = max(len(new._ax_lines) // maxlines, 1)
        new.ax.lines = new._ax_lines[::setpy]  # displayed ax lines

    if data_only:
        # if data only (we will  ot set axes and labels
        # it was probably done already in a previous plot
        new._plot_resume(source, **kwargs)
        return True

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
    if kind in ['stack']:
        # the z axis info
        # ----------------

        #zl = (np.min(np.ma.min(ys)), np.max(np.ma.max(ys)))
        amp = np.ma.ptp(z)/100.
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

    number_x_labels = plotoptions.number_of_x_labels
    number_y_labels = plotoptions.number_of_y_labels
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
        if kind in ['stack']:
            ylabel = make_label(new, 'z')
        else:
            ylabel = make_label(new.y, 'y')

    # z label
    # --------
    zlabel = kwargs.get("zlabel", None)
    if not zlabel:
        if kind in ['stack']:
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
    # avoid circular call to this module
    # from spectrochempy.application import plotoptions

    # contours
    maximum = data.max()
    minimum = -maximum

    nlevels = kwargs.get('nlevels', kwargs.get('nc',
                                               plotoptions.number_of_contours))
    start = kwargs.get('start', plotoptions.contour_start) * maximum
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
    ax = A.plot_stack()
    show()
    axT = A.plot_stack(data_transposed=True)
    show()
    pass