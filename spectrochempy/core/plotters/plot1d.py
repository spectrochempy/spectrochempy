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

"""
Module containing 1D plotting function(s)

"""
import numpy as np
from matplotlib.ticker import MaxNLocator

from spectrochempy.application import plotoptions
from spectrochempy.core.plotters.utils import make_label

__all__ = ['plot_1D']
_methods = __all__[:]


# ------------------------------------------------------------------------------
# plot_1D
# ------------------------------------------------------------------------------
def plot_1D(source, **kwargs):
    """
    Plot of one-dimensional data

    Parameters
    ----------
    source: :class:`~spectrochempy.core.ddataset.nddataset.NDDataset` to plot

    reverse: `bool` or None [optional, default= None/False
        In principle, coordinates run from left to right, except for wavenumbers
        (e.g., FTIR spectra) or ppm (e.g., NMR), that spectrochempy
        will try to guess. But if reverse is set, then this is the
        setting which will be taken into account.

    hold: `bool` [optional, default=`False`]

        If true hold the current figure and ax until a new plot is performed.

    data_only: `bool` [optional, default=`False`]

        Only the plot is done. No addition of axes or label specifications
        (current if any or automatic settings are kept.

    imag: `bool` [optional, default=`False`]

        Show imaginary part. By default only the real part is displayed.

    show_complex: `bool` [optional, default=`False`]

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

    savename: str,

        name of the file to save the figure

    savefig: Bool,

        save the fig if savename is defined,
        should be executed after all other commands

    vshift: float, optional
        vertically shift the line from its baseline


    kwargs : additional keywords

    """
    # where to plot?
    # ---------------

    source._figure_setup(**kwargs)

    # -------------------------------------------------------------------------
    # plot the source
    # -------------------------------------------------------------------------

    # abscissa axis
    x = source.x

    # ordinates (by default we plot real part of the data)
    if not kwargs.get('imag', False) or kwargs.get('show_complex', False):
        z = source.real
    else:
        z = source.imag

    # offset
    offset = kwargs.get('offset', 0.0)
    z = z - offset

    # plot
    line, = source.ax.plot(x.coords, z.data)

    if kwargs.get('show_complex', False):
        zimag = source.imag
        source.ax.plot(x.coords, zimag.data, ls='--')

    if kwargs.get('plot_model', False):
        modeldata = source.modeldata
        source.ax.plot(x.coords, modeldata.T, ls=':', lw='2')
        #TODO: improve this!!!

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

    if kwargs.get('data_only', False):
        # if data only (we will not set axes and labels
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

    # reversed axis?
    reverse = source.x.is_reversed
    if kwargs.get('reverse', reverse):
        xlim.reverse()

    # ordinates limits?
    zl = [np.amin(z.data), np.amax(z.data)]
    zlim = list(kwargs.get('zlim', kwargs.get('ylim', zl)))
    zlim.sort()

    # set the limits
    source.ax.set_xlim(xlim)
    source.ax.set_ylim(zlim)

    number_x_labels = plotoptions.number_of_x_labels  # get from config
    number_y_labels = plotoptions.number_of_y_labels

    source.ax.xaxis.set_major_locator(MaxNLocator(number_x_labels))
    source.ax.yaxis.set_major_locator(MaxNLocator(number_y_labels))

    # -------------------------------------------------------------------------
    # labels
    # -------------------------------------------------------------------------

    # x label

    xlabel = kwargs.get("xlabel", None)
    if not xlabel:
        xlabel = make_label(source.x, 'x')
    source.ax.set_xlabel(xlabel)

    # z label
    zlabel = kwargs.get("zlabel", None)
    if not zlabel:
        zlabel = make_label(source, 'z')

    source.ax.set_ylabel(zlabel)

    # do we display the ordinate axis?
    if kwargs.get('show_z', True):
        source.ax.set_ylabel(zlabel)
    else:
        source.ax.set_yticks([])

    # do we display the zero line
    if kwargs.get('show_zero', False):
        source.ax.haxlines()

    source._plot_resume(**kwargs)

    return True
