# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
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

from traitlets import Unicode, List, Bool, observe, Integer, Float, Tuple, \
    Unicode
from traitlets.config.configurable import Configurable

import matplotlib as mpl
import matplotlib.pyplot as plt

_classes = ['PlotOptions']


# =============================================================================
# Plot Options
# =============================================================================
class PlotOptions(Configurable):
    """
    All options relative to plotting and views

    """
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

    kind_2D = Unicode('map', help='default kind of plot for 2D').tag(
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

    cexponent = Float(1.2).tag(config=True)
    calpha = Float(1).tag(config=True)
