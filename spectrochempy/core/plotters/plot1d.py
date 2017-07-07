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
from ...preferences import preference_manager as pm

def plot1D(data, **kwargs):
    """

    Parameters
    ----------
    data : NDDataset to plot

    kwargs : additionnal keywords


    """
    # where to plot?
    ax = data.ax

    # -------------------------------------------------------------------------
    # plot the data
    # -------------------------------------------------------------------------

    # abscissa axis
    x = data.x

    # ordinates
    y = data.real()

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

    number_x_labels = pm.plot.number_x_labels # get from config
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
            units = "({:~X})".format(x.units)
        else:
            units = '(a.u)'

        xlabel = r"$\mathrm{%s\ %s}$"%(label, units)

    ax.set_xlabel(xlabel)

    # y label

    # if ylabel exists or is passed, use this
    ylabel = kwargs.get("ylabel", ax.get_ylabel())

    if not ylabel:

        # make a label from title and units
        if data.title:
            label = data.title.replace(' ', '\ ')
        else:
            label = r"intensity"
        if data.units is not None and str(data.units) !='dimensionless':
            units = r"({:~X})".format(data.units)
        else:
            units = '(a.u.)'
        ylabel = r"$\mathrm{%s\ %s}$" % (label, units)

    ax.set_ylabel(ylabel)

    # do we display the ordinate axis?
    if kwargs.get('show_y', True):
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticks([])

    # do we display the zero line
    if kwargs.get('show_zero', False):
        ax.haxlines()

#--------------------------------------
from ..dataset import NDDataset
setattr(NDDataset, 'plot1D', plot1D)
