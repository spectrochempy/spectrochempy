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

from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from spectrochempy.application import plotoptions, log
from spectrochempy.core.plotters.utils import make_label

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
    return plot_2D(source, **kwargs)


# stack plot  -----------------------------------------------------------------

def plot_stack(source, **kwargs):
    """
    Plot a 2D dataset as a stacked plot.

    Alias of plot_2D (with `kind` argument set to ``stack``).

    """
    kwargs['kind'] = 'stack'
    return plot_2D(source, **kwargs)


# image plot --------------------------------------------------------

def plot_image(source, **kwargs):
    """
    Plot a 2D dataset as an image plot.

    Alias of plot_2D (with `kind` argument set to ``image``).

    """
    kwargs['kind'] = 'image'
    return plot_2D(source, **kwargs)


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

    kind: `str` [optional among ``map``, ``stack`` or ``3d`` , default=``stack``]

    style : str, optional, default = 'notebook'
        Matplotlib stylesheet (use `available_style` to get a list of available
        styles for plotting

    kwargs : additional keywords


    """
    # where to plot?
    # --------------

    source._figure_setup(ndim=2, **kwargs)

    # kind of plot
    # ------------

    data_only = kwargs.get('data_only', False)

    # Other properties
    # ------------------


    kind = kwargs.get('kind', plotoptions.kind_2D)

    colorbar = kwargs.get('colorbar', True)

    cmap = colormap = kwargs.pop('colormap',
                                 kwargs.pop('cmap',
                                            mpl.rcParams['image.cmap']))

    lw = kwargs.get('linewidth', kwargs.get('lw', plotoptions.linewidth))

    alpha = kwargs.get('calpha', plotoptions.calpha)

    # -------------------------------------------------------------------------
    # plot the source
    # by default contours are plotted
    # -------------------------------------------------------------------------

    # ordinates (by default we plot real part of the data)
    if not kwargs.get('imag', False):
        z = source.RR()
    else:
        z = source.IR()

    # abscissa axis
    x = source.x

    # ordinates axis
    y = source.y

    # limits to tdeff
    tdeff = z.meta.tdeff
    xeff = x.coords[:tdeff[1]]
    yeff = y.coords[:tdeff[0]]
    zeff = z.data[:tdeff[0],:tdeff[1]]

    if kind in ['map', 'image']:
        vmax = zeff.max()
        vmin = zeff.min()
        if not kwargs.get('negative', True):
            vmin=0
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    if kind in ['map']:

        # contour plot
        # -------------
        if z.clevels is None:
            z.clevels = clevels(zeff, **kwargs)
            c = source.ax.contour(xeff, yeff, zeff,
                              z.clevels, linewidths=lw, alpha=alpha)
        c.set_cmap(cmap)
        c.set_norm(norm)

    elif kind in ['image']:

        # image plot
        # ----------
        kwargs['nlevels'] = 500
        if z.clevels is None:
            z.clevels = clevels(zeff, **kwargs)
        c = source.ax.contourf(xeff, yeff, zeff,
                               z.clevels, linewidths=lw, alpha=alpha)
        c.set_cmap(cmap)
        c.set_norm(norm)

    elif kind in ['stack']:

        # stack plot
        # ----------
        step = kwargs.get("step", "all")
        normalize = kwargs.get('normalize', None)
        color = kwargs.get('color', 'colormap')

        if not isinstance(step, str):
            showed = np.arange(yeff[0], yeff[-1], float(step))
            ishowed = np.searchsorted(yeff, showed, 'left')
        elif step == 'all':
            ishowed = slice(None)
        else:
            raise ValueError(
                    'step parameter was not recognized. Should be: an int, "all"')

        zeffs = zeff[ishowed]

        # now plot the collection of lines
        # ---------------------------------
        if color is None:
            # very basic plot (likely the faster)
            # use the matplotlib color cycler
            source.ax.plot(xeff, zeffs, lw=lw)

        elif color != 'colormap':
            # just add a color to the line (the same for all)
            source.ax.plot(xeff, zeffs, c=color, lw=lw)

        elif color == 'colormap':
            # here we map the color of each line to the colormap
            # according to the y axis values
            # if not source._updateplot:
            ylim = kwargs.get("ylim", None)

            if ylim is not None:
                vmin, vmax = ylim
            else:
                vmin, vmax = yeff[0], yeff[-1]
            norm = mpl.colors.Normalize(vmin=vmin,
                                        vmax=vmax)  # we normalize to the max time
            if normalize is not None:
                norm.vmax = normalize

            sp = z.sort(inplace=False)

            ys = [sp.data[i] for i in range(len(sp.y.coords))]
            sc = sp.y.coords

            line_segments = LineCollection(
                    [list(zip(sp.x.coords, y)) for y in ys[::-1]],
                    # Make a sequence of x,s[i] pairs
                    # linewidths    = (0.5,1,1.5,2),
                    linewidths=(lw,),
                    linestyles='solid',
                    # alpha=.5,
            )
            line_segments.set_array(sc[::-1])
            line_segments.set_cmap(colormap)
            line_segments.set_norm(norm)

            source.ax.add_collection(line_segments)

    if data_only:
        # if data only (we will  ot set axes and labels
        # it was probably done already in a previous plot
        source._plot_resume(**kwargs)
        return True

    # -------------------------------------------------------------------------
    # axis limits and labels
    # -------------------------------------------------------------------------
    # abscissa limits?
    xl = [x.coords[0], x.coords[-1]]
    xl.sort()
    xlim = list(kwargs.get('xlim', xl))
    xlim.sort()
    xlim[-1] = min(xlim[-1], xl[-1])
    xlim[0] = max(xlim[0], xl[0])

    # reversed x axis?
    # -----------------
    if kwargs.get('x_reverse', x.is_reversed):
        xlim.reverse()

    # set the limits
    # ---------------
    source.ax.set_xlim(xlim)

    # ordinates limits?
    # ------------------
    if kind in ['stack']:
        # the z axis info
        # ----------------
        zl = (np.amin(np.amin(ys)), np.amax(np.amax(ys)))
        zlim = list(kwargs.get('zlim', zl))
        zlim.sort()
        z_reverse = kwargs.get('z_reverse', False)
        if z_reverse:
            zlim.reverse()

        # set the limits
        # ---------------
        source.ax.set_ylim(zlim)

    else:
        # the y axis info
        # ----------------
        ylim = list(kwargs.get('ylim', source.ax.get_ylim()))
        ylim.sort()
        y_reverse = kwargs.get('y_reverse', y.is_reversed)
        if y_reverse:
            ylim.reverse()

        # set the limits
        # ----------------
        source.ax.set_ylim(ylim)

    number_x_labels = plotoptions.number_of_x_labels
    number_y_labels = plotoptions.number_of_y_labels
    source.ax.xaxis.set_major_locator(MaxNLocator(number_x_labels))
    source.ax.yaxis.set_major_locator(MaxNLocator(number_y_labels))

    # -------------------------------------------------------------------------
    # labels
    # -------------------------------------------------------------------------

    # x label
    # -------
    xlabel = kwargs.get("xlabel", None)
    if not xlabel:
        xlabel = make_label(x, 'x')
    source.ax.set_xlabel(xlabel)

    # y label
    # --------
    ylabel = kwargs.get("ylabel", None)
    if not ylabel:
        ylabel = make_label(y, 'y')

    # z label
    # --------
    zlabel = kwargs.get("zlabel", None)
    if not zlabel:
        zlabel = make_label(z, 'z')

    # do we display the ordinate axis?
    if kwargs.get('show_y', True):
        if kind not in ['stack']:
            source.ax.set_ylabel(ylabel)
        else:
            source.ax.set_ylabel(zlabel)
    else:
        source.ax.set_yticks([])

    if colorbar:

        if kind in ['stack']:
            source._axcb = source.fig.colorbar(line_segments, ax=source.ax)
            source._axcb.set_ticks(np.linspace(int(vmin), int(vmax), 5))
            source._axcb.set_label(ylabel)
        else:
            if not source._axcb:
                source._axcb = mpl.colorbar.ColorbarBase(source.axec,
                                                                cmap=cmap,
                                                                norm=norm)
                source._axcb.set_label(zlabel)
            pass

    # do we display the zero line
    if kwargs.get('show_zero', False):
        source.ax.haxlines()

    source._plot_resume(**kwargs)

    return source.ax


# ===========================================================================
# clevels
# ===========================================================================
def clevels_old(data, **kwargs):
    """Utility function to determine contours levels
    """
    # avoid circular call to this module
    from spectrochempy.application import plotoptions

    # contours
    maximum = data.max()
    minimum = -maximum

    nlevels = kwargs.get('nlevels', kwargs.get('nc',
                                               plotoptions.number_of_contours))
    exponent = kwargs.get('exponent', plotoptions.cexponent)
    if 'start' in kwargs:
        log.info("'start' is deprecated, use 'negative' instead.")
    negative = kwargs.get('start', kwargs.get('negative', True))
    if negative < 0:
        negative = True


    maximum = data.max()
    ms = maximum / nlevels
    for xi in range(100):
        if ms * exponent ** xi > maximum:
            xl = xi
            break
    clevelc = [ms * exponent ** xi for xi in range(xl)]
    if negative:
        clevelc = [- ms * exponent ** xi for xi in range(xl)[::-1]] + clevelc

    return sorted(clevelc)

# ===========================================================================
# clevels
# ===========================================================================
def clevels(data, **kwargs):
    """Utility function to determine contours levels
    """
    # avoid circular call to this module
    from spectrochempy.application import plotoptions

    # contours
    maximum = data.max()
    minimum = -maximum

    nlevels = kwargs.get('nlevels', kwargs.get('nc',
                                               plotoptions.number_of_contours))
    start = kwargs.get('start', maximum / 20.)
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