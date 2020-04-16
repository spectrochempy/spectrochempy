# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ['cmyk2rgb', 'NBlack', 'NRed', 'NBlue', 'NGreen',
           'figure', 'show', 'get_figure']

import os
import shutil as sh
from pkg_resources import resource_filename

from matplotlib import pyplot as plt
import matplotlib as mpl


# ............................................................................
# color conversion function
def cmyk2rgb(C, M, Y, K):
    """CMYK to RGB conversion
    C,M,Y,K are given in percent.
    The R,G,B values are returned in the range of 0..1.
    """
    C, Y, M, K = C / 100., Y / 100., M / 100., K / 100.

    # The red (R) color is calculated from the cyan (C) and black (K) colors:
    R = (1.0 - C) * (1.0 - K)

    # The green color (G) is calculated from the magenta (M) and black (K) colors:
    G = (1. - M) * (1. - K)

    # The blue color (B) is calculated from the yellow (Y) and black (K) colors:
    B = (1. - Y) * (1. - K)

    return (R, G, B)


# Constants
# ----------------------------------------------------------------------------------------------------------------------

# For color blind people, it is safe to use only 4 colors in graphs:
# see http://jfly.iam.u-tokyo.ac.jp/color/ichihara_etal_2008.pdf
#   Black CMYK=0,0,0,0
#   Red CMYK= 0, 77, 100, 0 %
#   Blue CMYK= 100, 30, 0, 0 %
#   Green CMYK= 85, 0, 60, 10 %
NBlack = (0, 0, 0)
NRed = cmyk2rgb(0, 77, 100, 0)
NBlue = cmyk2rgb(100, 30, 0, 0)
NGreen = cmyk2rgb(85, 0, 60, 10)


# .............................................................................
def figure(**kwargs):
    """
    Method to open a new figure

    Parameters
    ----------
    kwargs : any
        keywords arguments to be passed to the matplotlib figure constructor.

    """
    return get_figure(clear=True, **kwargs)


# .............................................................................
def show():
    """
    Method to force the `matplotlib` figure display

    """
    from spectrochempy import NO_DISPLAY

    if not NO_DISPLAY:

        if get_figure(clear=False):  # True to avoid opening a new one
            plt.show(block=True)


# .............................................................................
def get_figure(clear=True, **kwargs):
    """
    Get the figure where to plot.

    Parameters
    ----------
    clear : bool
        if False the last used figure is used.
    kwargs : any
        keywords arguments to be passed to the matplotlib figure constructor.

    Returns
    -------
    matplotlib figure instance

    """

    n = plt.get_fignums()

    if not n or clear:
        # create a figure
        return plt.figure(**kwargs)

    # a figure already exists - if several we take the last
    return plt.figure(n[-1])
