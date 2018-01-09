# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================
__all__ = []

import os

from traitlets import Unicode, List, Bool, observe, Integer, Float
from traitlets.config.configurable import Configurable

import matplotlib.pyplot as plt
import matplotlib as mpl


# ============================================================================
class ProjectPreferences(Configurable) :
    """
    Per project preferences

    include plotting and views preference for the incuded datasets

    """

    def __init__(self, **kwargs):
        super(ProjectPreferences, self).__init__(**kwargs)

    # ------------------------------------------------------------------------
    # attributes
    # ------------------------------------------------------------------------

    name = Unicode('PlotterPreferences')

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
    latex_preamble = Unicode(
r"""\usepackage{siunitx}
\sisetup{detect-all}
\usepackage{times} # set the normal font here
\usepackage{sansmath}
# load up the sansmath so that math -> helvet
\sansmath
""",
                          help='Latex preamble for matplotlib outputs'
                          ).tag(config=True, type='text')

    @observe('latex_preamble')
    def _set_latex_preamble(self, change):
        mpl.rcParams['text.latex.preamble'] = change.new.split('\n')

    # -------------------------------------------------------------------------

    method_2D = Unicode('map',
                        help='Default plot methods for 2D'
                        ).tag(config=True)

    colorbar = Bool(True,
                       help='Show color bar for 2D plots'
                       ).tag(config=True)

    colormap = Unicode('jet',
                       help='Default colormap for contour plots'
                       ).tag(config=True)

    colormap_stack = Unicode('viridis',
                             help='Default colormap for stack plots'
                             ).tag(config=True)

    colormap_transposed = Unicode('magma',
                            help='Default colormap for transposed stack plots'
                                  ).tag(config=True)

    show_projections = Bool(False,
                            help='Show all projections'
                            ).tag(config=True)

    show_projection_x = Bool(False, help='Show projection along x'
                             ).tag(config=True)

    show_projection_y = Bool(False, help='Show projection along y'
                             ).tag(config=True)

    background_color = Unicode('#EFEFEF', help='Bakground color for plots'
                              ).tag(config=True, type='color')

    foreground_color = Unicode('#000', help='Foreground color for plots'
                              ).tag(config=True, type='color')

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
                                       ' plot in stack plots'
                                 ).tag(config=True)

