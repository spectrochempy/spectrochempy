# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to provide a general
# API for displaying, processing and analysing spectrochemical data.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
# =============================================================================



from traitlets import Unicode, List, Bool, observe, Integer, Float, Tuple, \
    Unicode
from traitlets.config.configurable import Configurable

import matplotlib.pyplot as plt

import matplotlib as mpl

__all__ = []


# =============================================================================
# Plot Options
# =============================================================================
class PlotOptions(Configurable):
    """
    All options relative to plotting and views

    """
    import matplotlib.pyplot as plt

    name = Unicode('PlotOptions')

    description = Unicode('Options for plotting datasets')

    # -------------------------------------------------------------------------
    style = Unicode('lcs',
                    help='basic matplotlib style to use').tag(config=True)

    use_latex = Bool(True, help='should we use latex for '
                                'plotting labels and texts?').tag(config=True)

    @observe('use_latex')
    def _use_latex_changed(self, change):
        mpl.rc('text', usetex=change.new)

    @observe('style')
    def _style_changed(self, change):
        plt.style.use(change.new)

    # -------------------------------------------------------------------------

    latex_preamble = List(mpl.rcParams['text.latex.preamble'],
                          help='latex preamble for matplotlib outputs'
                          ).tag(config=True)

    @observe('latex_preamble')
    def _set_latex_preamble(self, change):
        mpl.rcParams['text.latex.preamble'] = change.new

    # -------------------------------------------------------------------------

    do_not_block = Bool(False,
                        help="whether or not we show the plots "
                             "and stop after each of them").tag(config=True)

    offscreen = Bool(False)

    method_2D = Unicode('map', help='default method of plot for 2D').tag(
            config=True)

    colormap = Unicode('jet',
            help='default colormap for contour plots').tag(config=True)

    colormap_stack = Unicode('viridis',
                       help='default colormap for stack plots').tag(
        config=True)

    colormap_transposed = Unicode('magma',
                       help='default colormap for stack plots').tag(
        config=True)

    show_projections = Bool(False, help='show all projections').tag(
            config=True)
    show_projection_x = Bool(False, help='show projection along x').tag(
            config=True)
    show_projection_y = Bool(False, help='show projection along y').tag(
            config=True)

    background_color = Tuple((0.5, 0.5, 0.5)).tag(config=True)
    foreground_color = Tuple((1.0, 1.0, 1.0)).tag(config=True)
    linewidth = Float(.7).tag(config=True)

    number_of_x_labels = Integer(5).tag(config=True)
    number_of_y_labels = Integer(5).tag(config=True)
    number_of_z_labels = Integer(5).tag(config=True)

    number_of_contours = Integer(50).tag(config=True)
    contour_alpha = Float(1, help='Transparency of the contours').tag(config=True)
    contour_start = Float(0.05, help='percentage of the maximum '
                              'for starting contour levels').tag(config=False)

    max_lines_in_stack = Integer(1000, help='maximum number of lines to'
                                       ' plot in a stack plot').tag(config=True)