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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from spectrochempy.preferences.preference_manager import preference_manager as pm

# =============================================================================
# nddataset plot2D functions
# =============================================================================

def plot2D(data, **kwargs):
    """

    Parameters
    ----------
    data : NDDataset to plot

    kwargs : additionnal keywords


    """
    # where to plot?
    ax = data.ax

    # show projections
    proj = kwargs.get('proj', pm.plot.show_projections)

    xproj = kwargs.get('xproj',pm.plot.show_x_projection)

    yproj = kwargs.get('yproj',pm.plot.show_y_projection)

    if proj or xproj or yproj:
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
            data.axex = axex

        if proj or yproj:
            axey = divider.append_axes("right", 1.01, pad=0.01, sharey=ax,
                                       frameon=0, xticks=[])
            axey.tick_params(right='off', left='off')
            plt.setp(axey.get_xticklabels() + axey.get_yticklabels(),
                    visible=False)
            data.axey = axey

    # contour colormap
    cmap = mpl.rcParams['image.cmap'] = kwargs.pop('colormap',
                        kwargs.pop('cmap', pm.plot.colormap))
    pm.plot.colormap =  cmap

    lw = kwargs.get('linewidth', kwargs.get('lw', pm.plot.linewidth))

    alpha = kwargs.get('alpha', pm.plot.contour_alpha)

    # -------------------------------------------------------------------------
    # plot the data
    # -------------------------------------------------------------------------

    # abscissa and ordinate axis
    x = data.x
    y = data.y

    # z axis
    z = data.real()

    # contour levels
    cl = clevels(z.data, **kwargs)

    # plot
    c = ax.contour(x.coords, y.coords, z.data, cl, linewidths=lw, alpha=alpha)
    c.set_cmap(cmap)

    # -------------------------------------------------------------------------
    # axis limits and labels
    # -------------------------------------------------------------------------

    # abscissa limits?
    xlim = list(kwargs.get('xlim', ax.get_xlim()))
    xlim.sort()
    xl = [x.coords[0], x.coords[-1]]
    xl.sort()
    xlim[-1] = min(xlim[-1], xl[-1])
    xlim[0] = max(xlim[0], xl[0])

    # reversed axis?
    reverse = data.meta.reverse
    if kwargs.get('reverse', reverse):
        xlim.reverse()

    # ordinates limits?
    zlim = list(kwargs.get('zlim', kwargs.get('ylim', ax.get_ylim())))
    zlim.sort()

    # set the limits
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)

    number_x_labels = pm.plot.number_x_labels
    number_y_labels = pm.plot.number_y_labels
    ax.xaxis.set_major_locator(MaxNLocator(number_x_labels))
    ax.yaxis.set_major_locator(MaxNLocator(number_y_labels))

    # -------------------------------------------------------------------------
    # labels
    # -------------------------------------------------------------------------

    # x label

    # if xlabel exists or is passed, use this
    xlabel = kwargs.get("xlabel", ax.get_xlabel())

    # else:
    if not xlabel:
        # make a label from title and units
        if x.title:
            label = x.title.replace(' ', '\ ')
        else:
            label = 'x'
        if x.units is not None and str(x.units) !='dimensionless':
            units = "({:~Lx})".format(x.units)
        else:
            units = '(a.u)'

        xlabel = r"$\mathrm{%s\ %s}$"%(label, units)

    ax.set_xlabel(xlabel)

    # y label

    # if ylabel exists or is passed, use this
    ylabel = kwargs.get("ylabel", ax.get_ylabel())

    if not ylabel:

        # make a label from title and units
        if y.title:
            label = y.title.replace(' ', '\ ')
        else:
            label = r"intensity"
        if y.units is not None and str(y.units) !='dimensionless':
            units = r"({:~X})".format(y.units)
        else:
            units = '(a.u.)'
        ylabel = r"$\mathrm{%s\ %s}$" % (label, units)

    # do we display the ordinate axis?
    if kwargs.get('show_y', True):
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticks([])

    # do we display the zero line
    if kwargs.get('show_zero', False):
        ax.haxlines()

# ===========================================================================
# clevels
# ===========================================================================
def clevels(data, **kwargs):
    """Utility function to determine contours levels
    """
    # contours
    maximum = data.max()
    minimum = -maximum

    nlevels = kwargs.get('nlevels', pm.plot.number_of_contour_levels)
    exponent = kwargs.get('exponent', pm.plot.contour_exponent)
    start = kwargs.get('start', pm.plot.contour_start)

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

#-----------------------------------------------------------------
from ..dataset import NDDataset
setattr(NDDataset, 'plot2D', plot2D)
