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
import numpy as np
from matplotlib.ticker import MaxNLocator

from spectrochempy.application import plotoptions as options
from spectrochempy.core.plotters.utils import make_label

__all__ = ['plot_1D']

def plot_1D(source, **kwargs):
    """

    Parameters
    ----------
    source : NDDataset to plot

    reverse: `bool` [optional, default=True]



    kwargs : additionnal keywords


    """
    # where to plot?
    #---------------
    ax = source.ax
    if ax is None:
        fig, ax = source.figure_setup(**kwargs)

    # -------------------------------------------------------------------------
    # plot the source
    # -------------------------------------------------------------------------

    # abscissa axis
    x = source.x

    # ordinates
    y = source.real()

    # offset
    offset = kwargs.get('offset', 0.0)
    y = y - offset

    # plot
    line, = ax.plot(x.coords, y.data)

    # line attributes
    c = kwargs.get('color', kwargs.get('c'))
    if c:
        line.set_color(c)
    lw = kwargs.get('linewidth', kwargs.get('lw', 1.))
    if lw:
        line.set_linewidth(lw)
    ls = kwargs.get('linestyle', kwargs.get('ls', '-'))
    if ls:
        line.set_linestyle(ls)

    # -------------------------------------------------------------------------
    # axis limits and labels
    # -------------------------------------------------------------------------

    # abscissa limits?
    xl = [x.coords[0], x.coords[-1]]
    xlim = list(kwargs.get('xlim', xl))
    xlim.sort()
    xlim[-1] = min(xlim[-1], xl[-1])
    xlim[0] = max(xlim[0], xl[0])

    # reversed axis?
    reverse = source.x.is_reversed
    if kwargs.get('reverse', reverse):
        xlim.reverse()

    # ordinates limits?
    zl = [np.amin(y.data), np.amax(y.data)]
    zlim = list(kwargs.get('zlim',zl))
    zlim.sort()

    # set the limits
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)

    number_x_labels = options.number_of_x_labels # get from config
    number_y_labels = options.number_of_y_labels

    ax.xaxis.set_major_locator(MaxNLocator(number_x_labels))
    ax.yaxis.set_major_locator(MaxNLocator(number_y_labels))

    # -------------------------------------------------------------------------
    # labels
    # -------------------------------------------------------------------------

    # x label

    xlabel = kwargs.get("xlabel", None)
    if not xlabel:
        xlabel = make_label(source.x, 'x')
    ax.set_xlabel(xlabel)

    # z label
    zlabel = kwargs.get("zlabel", None)
    if not zlabel:
        zlabel = make_label(source, 'z')

    ax.set_ylabel(zlabel)

    # do we display the ordinate axis?
    if kwargs.get('show_z', True):
        ax.set_ylabel(zlabel)
    else:
        ax.set_yticks([])

    # do we display the zero line
    if kwargs.get('show_zero', False):
        ax.haxlines()

    source.plot_resume(**kwargs)

    return True