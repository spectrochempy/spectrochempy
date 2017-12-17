# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
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
class plot_options(Configurable):
    """
    Options relative to plotting and views

    """
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------------
    # attributes
    # ------------------------------------------------------------------------

    name = Unicode('plot_options')

    description = Unicode('Options for plotting datasets')

    # ------------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------------

    # ........................................................................
    style = Unicode('lcs',
                    help='Basic matplotlib style to use').tag(config=True)

    @observe('style')
    def _style_changed(self, change):
        plt.style.use(change.new)

    # ........................................................................
    use_latex = Bool(True,
                     help='Should we use latex for plotting labels and texts?'
                     ).tag(config=True)

    @observe('use_latex')
    def _use_latex_changed(self, change):
        mpl.rc('text', usetex=change.new)

    # ........................................................................
    latex_preamble = List(mpl.rcParams['text.latex.preamble'],
                          help='latex preamble for matplotlib outputs'
                          ).tag(config=True)

    @observe('latex_preamble')
    def _set_latex_preamble(self, change):
        mpl.rcParams['text.latex.preamble'] = change.new

    # -------------------------------------------------------------------------

    method_2D = Unicode('map',
                        help='Default plot methods for 2D'
                        ).tag(config=True)

    colormap = Unicode('jet',
                       help='Default colormap for contour plots'
                       ).tag(config=True)

    colormap_stack = Unicode('viridis',
                             help='Default colormap for stack plots'
                             ).tag(config=True)

    colormap_transposed = Unicode('magma',
                            help='Default colormap for tramsposed stack plots'
                                  ).tag(config=True)

    show_projections = Bool(False,
                            help='Show all projections'
                            ).tag(config=True)

    show_projection_x = Bool(False, help='Show projection along x'
                             ).tag(config=True)

    show_projection_y = Bool(False, help='Show projection along y'
                             ).tag(config=True)

    background_color = Tuple((0.5, 0.5, 0.5), help='Bakground color for plots'
                             ).tag(config=True)

    foreground_color = Tuple((1.0, 1.0, 1.0), help='Foreground color for plots'
                             ).tag(config=True)

    linewidth = Float(.7, help='Default width for lines').tag(config=True)

    number_of_x_labels = Integer(5, help='Number of X labels').tag(config=True)

    number_of_y_labels = Integer(5, help='Number of Y labels').tag(config=True)

    number_of_z_labels = Integer(5, help='Number of Z labels').tag(config=True)

    number_of_contours = Integer(50, help='Number of contours').tag(
        config=True)

    contour_alpha = Float(1, help='Transparency of the contours'
                          ).tag(config=True)

    contour_start = Float(0.05, help='Fraction of the maximum '
                              'for starting contour levels'
                          ).tag(config=True)

    max_lines_in_stack = Integer(1000, help='Maximum number of lines to'
                                       ' plot in a stack plot'
                                 ).tag(config=True)

    do_not_block = Bool(False, help="whether or not we show the plots "
                                    "and stop after each of them"
                        ).tag(config=True)
