# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================
"""
This module should be able to handle a large set of plot types.
"""

__all__ = ["plot_3D"]

__dataset_methods__ = []

# from spectrochempy.core import project_preferences

# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import numpy as np


# =============================================================================
# nddataset plot3D functions
# =============================================================================


def plot_3D(dataset, **kwargs):
    raise NotImplementedError("Not implemented")

    # """
    # 3D Plots of NDDatasets
    #
    # Parameters
    # ----------
    # dataset: :class:`~spectrochempy.ddataset.nddataset.NDDataset` to plot
    #
    # data_only: `bool` [optional, default=`False`]
    #
    #     Only the plot is done. No addition of axes or label specifications
    #     (current if any or automatic settings are kept.
    #
    # method: str [optional among ``surface``, ... (other to be implemented)..., default=``surface``]
    #
    # style: str, optional, default='notebook'
    #     Matplotlib stylesheet (use `available_style` to get a list of available
    #     styles for plotting
    #
    # reverse: `bool` or None [optional, default=None
    #     In principle, coordinates run from left to right, except for wavenumbers
    #     (e.g., FTIR spectra) or ppm (e.g., NMR), that spectrochempy
    #     will try to guess. But if reverse is set, then this is the
    #     setting which will be taken into account.
    #
    # x_reverse: `bool` or None [optional, default=None
    #
    # kwargs: additional keywords
    #
    # {}
    #
    # """.format(dataset._general_parameters_doc_)
    #
    # # get all plot preferences
    # # ------------------------
    #
    # prefs = dataset.preferences
    # if not prefs.style:
    #     # not yet set, initialize with default project preferences
    #     prefs.update(project_preferences.to_dict())
    #
    # # If we are in the GUI, we will plot on a widget: but which one?
    # # ---------------------------------------------------------------
    # #
    #
    # widget = kwargs.get('widget', None)
    #
    # if widget is not None:
    #     if hasattr(widget, 'implements') and widget.implements('PyQtGraphWidget'):
    #         # let's go to a particular treament for the pyqtgraph plots
    #         kwargs['use_mpl'] = use_mpl = False
    #         # we try to have a commmon interface for both plot library
    #         kwargs['ax'] = ax = widget  # return qt_plot_1D(dataset, **kwargs)
    #     else:
    #         # this must be a matplotlibwidget
    #         kwargs['use_mpl'] = use_mpl = True
    #         fig = widget.fig
    #         kwargs['ax'] = ax = fig.gca()
    #
    # # method of plot
    # # ------------
    #
    # method = kwargs.get('method', prefs.method_3D)
    #
    # data_only = kwargs.get('data_only', False)
    # new = dataset.copy()
    #
    # # figure setup
    # # ------------
    #
    # ax = plt.axes(projection='3d')
    #
    # # Other properties
    # # ------------------
    #
    # colorbar = kwargs.get('colorbar', project_preferences.colorbar)
    #
    # cmap = mpl.rcParams['image.cmap']
    #
    # # viridis is the default setting,
    # # so we assume that it must be overwritten here
    # # except if style is grayscale which is a particular case.
    # styles = kwargs.get('style', project_preferences.style)
    #
    # if styles and not "grayscale" in styles and cmap == "viridis":
    #
    #     if method in ['surface']:
    #         cmap = colormap = kwargs.get('colormap',
    #                                      kwargs.get('cmap', project_preferences.colormap_surface))
    #     else:
    #         # other methods to be implemented
    #         pass
    #
    # lw = kwargs.get('linewidth', kwargs.get('lw',
    #                                         project_preferences.pen_linewidth))
    #
    # antialiased = kwargs.get('antialiased', project_preferences.antialiased)
    #
    # rcount = kwargs.get('rcount', project_preferences.rcount)
    #
    # ccount = kwargs.get('ccount', project_preferences.ccount)
    #
    # if method == 'surface':
    #     X, Y = np.meshgrid(new.x.data, new.y.data)
    #     Z = dataset.data
    #
    #     # Plot the surface.
    #     surf = ax.plot_surface(X, Y, Z, cmap=cmap,
    #                            linewidth=lw, antialiased=antialiased,
    #                            rcount=rcount, ccount=ccount)
    #     if not data_only:
    #         ax.set_xlabel(new.x.title)
    #         ax.set_ylabel(new.y.title)
    #         ax.set_zlabel(new.title)
    #
    # return ax
