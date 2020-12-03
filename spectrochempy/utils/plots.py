# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import matplotlib as mpl
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import numpy as np

__all__ = ['cmyk2rgb', 'NBlack', 'NRed', 'NBlue', 'NGreen',
           'figure', 'show', 'get_figure',
           # Plotly specific
           'get_plotly_figure', 'colorscale']


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


# FOR PLOTLY
# .............................................................................
def get_plotly_figure(clear=True, fig=None, **kwargs):
    """
    Get the figure where to plot.

    Parameters
    ----------
    clear : bool
        if False the figure provided in the `fig` parameters is used.
    fig : plotly figure
        if provided, and clear is not True, it will be used for plotting
    kwargs : any
        keywords arguments to be passed to the plotly figure constructor.

    Returns
    -------
    Plotly figure instance

    """

    if clear or fig is None:
        # create a figure
        return go.Figure()

    # a figure already exists - if several we take the last
    return fig


class colorscale:

    def normalize(self, vmin, vmax, cmap='viridis', rev=False, offset=0):
        """
        """
        if rev:
            cmp = cmap + '_r'
        _colormap = plt.get_cmap(cmap)

        _norm = mpl.colors.Normalize(vmin=vmin - offset, vmax=vmax - offset)
        self.scalarMap = mpl.cm.ScalarMappable(norm=_norm, cmap=_colormap)

    def rgba(self, z, offset=0):
        c = np.array(self.scalarMap.to_rgba(z.squeeze() - offset))
        c[0:3] *= 255
        c[0:3] = np.round(c[0:3].astype('uint16'), 0)
        return f'rgba' + str(tuple(c))


colorscale = colorscale()
