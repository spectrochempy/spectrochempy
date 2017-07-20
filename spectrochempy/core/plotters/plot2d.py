# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2016 LCS
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

from spectrochempy.application import plotoptions as options
from spectrochempy.core.plotters.utils import make_label

__all__ = ['plot_2D', 'plot_map', 'plot_stack']
_methods = __all__[:]

# =============================================================================
# nddataset plot2D functions
# =============================================================================

# contour map -----------------------------------------------------------------

def plot_map(source, **kwargs):
    """
    Plot a 2D dataset as a contoured map.

    Alias of plot_2D (with `kind` argument set to ``map``.

    """
    kwargs['kind'] = 'map'
    return plot_2D(source, **kwargs)


# stack plot (default) --------------------------------------------------------

def plot_stack(source, **kwargs):
    """
    Plot a 2D dataset as a stacked plot.

    Alias of plot_2D (with `kind` argument set to ``stack``).

    """
    kwargs['kind'] = 'stack'
    return plot_2D(source, **kwargs)


# generic plot (default stack plot) -------------------------------------------

def plot_2D(source, **kwargs):
    """
    PLot of 2D array.

    Parameters
    ----------
    source : :class:`~spectrochempy.core.ddataset.nddataset.NDDataset` to plot

    projections : `bool` [optional, default=False]

    kind : `str` [optional among ``map``, ``stack`` or ``3d`` , default=``stack``]

    kwargs : additional keywords


    """
    # where to plot?
    # ----------------
    if not kwargs.get('hold', False):
        source.ax = None # remove reference to previously used

    ax = kwargs.get('ax', source.ax)
    if ax is None:
        fig, ax = source.figure_setup(**kwargs)

    # kind of plot
    # ------------
    kind = kwargs.get('kind', options.kind_2D)

    # show projections (only useful for maps)
    # ----------------------------------------
    proj = kwargs.get('proj',
                      options.show_projections)  # TODO: tell the axis by title.

    xproj = kwargs.get('xproj', options.show_projection_x)

    yproj = kwargs.get('yproj', options.show_projection_y)

    if (proj or xproj or yproj) and kind in ['map' 'image']:
        # create new axes on the right and on the top of the current axes
        # The first argument of the new_vertical(new_horizontal) method is
        # the height (width) of the axes to be created in inches.
        divider = make_axes_locatable(ax)
        # print divider.append_axes.__doc__
        if proj or xproj:
            axex = divider.append_axes("top", 1.01, pad=0.01, sharex=ax,
                                       frameon=0, yticks=[])
            axex.tick_params(bottom='off', top='off')
            plt.setp(axex.get_xticklabels() + axex.get_yticklabels(),
                     visible=False)
            source.axex = axex

        if proj or yproj:
            axey = divider.append_axes("right", 1.01, pad=0.01, sharey=ax,
                                       frameon=0, xticks=[])
            axey.tick_params(right='off', left='off')
            plt.setp(axey.get_xticklabels() + axey.get_yticklabels(),
                     visible=False)
            source.axey = axey

    # colormap
    # ----------
    cmap = colormap = kwargs.pop('colormap',
                      kwargs.pop('cmap', options.colormap))

    colorbar = kwargs.get('colorbar', True)

    lw = kwargs.get('linewidth', kwargs.get('lw', options.linewidth))

    alpha = kwargs.get('calpha', options.calpha)

    # -------------------------------------------------------------------------
    # plot the source
    # by default contours are plotted
    # -------------------------------------------------------------------------

    s = source.real()

    ylim = kwargs.get("ylim", None)
    zlim = kwargs.get("zlim", None)

    if kind in ['map']:

        # contour plot
        # -------------
        cl = clevels(s.data, **kwargs)
        c = ax.contour(s.x.coords, s.y.coords, s.data, cl, linewidths=lw,
                       alpha=alpha)
        c.set_cmap(cmap)

    elif kind in ['image']:

        kwargs['nlevels'] = 500
        cl = clevels(s.data, **kwargs)
        c = ax.contourf(s.x.coords, s.y.coords, s.data, cl, linewidths=lw,
                        alpha=alpha)
        c.set_cmap(cmap)

    elif kind in ['stack']:

        step = kwargs.get("step", "all")
        normalize = kwargs.get('normalize', None)
        color = kwargs.get('color', 'colormap')

        if not isinstance(step, str):
            showed = np.arange(s.y[0], s.y[-1], float(step))
            ishowed = np.searchsorted(s.y, showed, 'left')
        elif step == 'all':
            ishowed = slice(None)
        else:
            raise ValueError(
                    'step parameter was not recognized. Should be: an int, "all"')

        s = s[ishowed]

        # now plot the collection of lines
        #---------------------------------
        if color == None:
            # very basic plot (likely the faster)
            # use the matplotlib color cycler
            ax.plot(s.x.coords, s.data, lw=lw )

        elif color != 'colormap':
            # just add a color to the line (the same for all)
            ax.plot(s.x.coords, s.data, c=color, lw=lw)

        elif color == 'colormap':
            # here we map the color of each line to the colormap
            if ylim is not None:
                vmin, vmax = ylim
            else:
                vmin, vmax = s.y.coords[0], s.y.coords[-1]
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)  # we normalize to the max time
            if normalize is not None:
                norm.vmax = normalize

            sp = s.sort(inplace=False)
            ys = [sp.data[i] for i in range(len(sp.y.coords))]
            sc = sp.y.coords

            line_segments = LineCollection(
                                        [list(zip(sp.x.coords, y)) for y in ys[::-1]],
                                        # Make a sequence of x,s[i] pairs
                                        # linewidths    = (0.5,1,1.5,2),
                                        linewidths=(lw,),
                                        linestyles='solid',
                                        #alpha=.5,
                             )
            line_segments.set_array(sc[::-1])
            line_segments.set_cmap(colormap)
            line_segments.set_norm(norm)

            ax.add_collection(line_segments)

    # -------------------------------------------------------------------------
    # axis limits and labels
    # -------------------------------------------------------------------------

    # abscissa limits?
    xl = [s.x.coords[0], s.x.coords[-1]]
    xl.sort()
    xlim = list(kwargs.get('xlim', xl))
    xlim.sort()
    xlim[-1] = min(xlim[-1], xl[-1])
    xlim[0] = max(xlim[0], xl[0])

    # reversed x axis?
    #-----------------
    if kwargs.get('x_reverse', s.x.is_reversed):
        xlim.reverse()

    # set the limits
    #---------------
    ax.set_xlim(xlim)

    # ordinates limits?
    #------------------
    if kind in ['stack']:
        # the z axis info
        #----------------
        zl = (np.amin(np.amin(ys)),np.amax(np.amax(ys)))
        zlim = list(kwargs.get('zlim', zl))
        zlim.sort()
        z_reverse = kwargs.get('z_reverse', False)
        if z_reverse:
            zlim.reverse()

        # set the limits
        #---------------
        ax.set_ylim(zlim)

    else:
        # the y axis info
        #----------------
        ylim = list(kwargs.get('ylim', ax.get_ylim()))
        ylim.sort()
        y_reverse = kwargs.get('y_reverse', s.y.is_reversed)
        if y_reverse:
            ylim.reverse()

        # set the limits
        #----------------
        ax.set_ylim(ylim)

    number_x_labels = options.number_of_x_labels
    number_y_labels = options.number_of_y_labels
    ax.xaxis.set_major_locator(MaxNLocator(number_x_labels))
    ax.yaxis.set_major_locator(MaxNLocator(number_y_labels))

    # -------------------------------------------------------------------------
    # labels
    # -------------------------------------------------------------------------



    # x label
    # -------
    xlabel = kwargs.get("xlabel", None)
    if not xlabel:
        xlabel = make_label(s.x, 'x')
    ax.set_xlabel(xlabel)

    # y label
    # --------
    ylabel = kwargs.get("ylabel", None)
    if not ylabel:
        ylabel = make_label(s.y, 'y')

    # z label
    # --------
    zlabel = kwargs.get("zlabel", None)
    if not zlabel:
        zlabel = make_label(s, 'z')


    # do we display the ordinate axis?
    if kwargs.get('show_y', True):
        if kind not in ['stack']:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(zlabel)
    else:
        ax.set_yticks([])

    if colorbar:

        fig = plt.gcf()

        if kind in ['stack']:

            axcb = fig.colorbar(line_segments, ax=ax)
            axcb.set_label(ylabel)
            axcb.set_ticks(np.linspace(int(vmin), int(vmax), 5))


    # do we display the zero line
    if kwargs.get('show_zero', False):
        ax.haxlines()

    source.plot_resume(**kwargs)

    return True


# ===========================================================================
# clevels
# ===========================================================================
def clevels(data, **kwargs):
    """Utility function to determine contours levels
    """
    # avoid circular call to this module
    from spectrochempy.application import plotoptions as options

    # contours
    maximum = data.max()
    minimum = -maximum

    nlevels = kwargs.get('nlevels', options.number_of_contours)
    exponent = kwargs.get('exponent', options.cexponent)
    start = kwargs.get('start', options.cstart)

    if (exponent - 1.00) < .005:
        clevelc = np.linspace(minimum, maximum, nlevels)
        clevelc[clevelc.size / 2 - 1:clevelc.size / 2 + 1] = np.NaN
        return clevelc

    maximum = data.max()
    ms = maximum / nlevels
    for xi in range(100):
        if ms * exponent ** xi > maximum:
            xl = xi
            break
    if start != 0:
        clevelc = [float(start) * ms * exponent ** xi
                   for xi in range(xl)] + [ms * exponent ** xi for xi in
                                           range(xl)]
    else:
        clevelc = [ms * exponent ** xi for xi in range(xl)]
    return sorted(clevelc)
