# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
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


"""Preferences for SpectroChemPy
"""
import logging

# Enthought library imports.
from traits.api import (Bool, Enum, Tuple, Range, List, Int, Float,
        Str, Instance, HasTraits, Trait)
from traitsui.api import View, Group, Item, RGBColorEditor

__all__ = ['RootPreferences', 'NDDatasetPreferences',
           'PlotPreferences']

class RootPreferences(HasTraits):

    #### Preferences ##########################################################

    # Specifies if the splash screen is shown when spectrochempy starts.
    show_splash_screen = Bool(desc='if the splash screen is shown at'
                              ' startup')

    # Specifies if the adder nodes are shown on the spectrochempy tree view.
    open_help_in_light_browser = Bool(
                    desc='if the help pages are opened in a chromeless'
                             'browser window (only works with Firefox')

    # Whether or not to use IPython for the Shell.
    use_ipython = Bool(desc='use IPython for the embedded shell '
                            '(if available)')

    #TODO: separate preference for API and those for GUI

    print_info_on_loading = Bool
    _DO_NOT_BLOCK = Bool

    # logging level
    log_level = Trait('DEBUG',
                      {'DEBUG': logging.DEBUG,
                       'INFO': logging.INFO,
                       'WARNING': logging.WARNING,
                       'ERROR': logging.ERROR,
                       'CRITICAL': logging.CRITICAL,
                       },
                      is_str=True,
                      )

    log_file_level = Trait('DEBUG',
                           {'DEBUG': logging.DEBUG,
                            'INFO': logging.INFO,
                            'WARNING': logging.WARNING,
                            'ERROR': logging.ERROR,
                            'CRITICAL': logging.CRITICAL,
                            },
                           is_str=True,
                           )


    #### Traits UI views ######################################################

    traits_view = View(
                    Group(
                        Group(
                            Item('print_info_on_loading'),
                            Item(name='show_splash_screen'),
                            Item(name='open_help_in_light_browser'),
                            Item(name='use_ipython'),
                            label='General settings',
                            show_border=True,
                            ),
                        Group(
                                Item('log_level'),
                                Item('log_file_level'),
                                label='Logger',
                                show_border=True,
                            ),
                        ),
                    resizable=True
                    )


################################################################################
# `PlotPreferences` class
################################################################################
class PlotPreferences(HasTraits):
    """ Preferences definition for SpectroChemPy. """

    #### Preferences ##########################################################

    # The background color of the renderer.
    background_color = Tuple(Range(0., 1., 1.),
                             Range(0., 1., 1.),
                             Range(0., 1., 1.),
                             editor=RGBColorEditor,
                             desc='the background color of the scene')

    # The foreground color of the renderer.
    foreground_color = Tuple(Range(0., 1., 0.),
                             Range(0., 1., 0.),
                             Range(0., 1., 0.),
                             editor=RGBColorEditor,
                             desc='the foreground color of the scene')

    # Offscreen rendering.
    offscreen = Bool(desc='if plotter should use offscreen rendering'
                          ' (no window will show up in this case)')

    # labels
    number_x_labels = Int(desc='number of label in the x dimension of plots')
    number_y_labels = Int(desc='number of label in the y dimension of plots')

    # 2D
    show_projections = Bool()
    show_x_projection = Bool()
    show_y_projection = Bool()
    number_of_contour_levels = Int(minval=5)
    contour_exponent = Float()
    contour_start = Range(-1.,1,-1.)
    contour_alpha = Range(0,1)
    linewidth = Float()
    colormap = Str()
    use_latex = Bool(desc="whether Latex is used or not in plot's text",
                     label = "use Latex in plot's text")

    #TODO: add tooltips (desc)

    ######################################################################
    # Traits UI view.

    traits_view = View(Group(
            Item('use_latex'),
            Item('background_color'),
            Item('foreground_color'),
            Item('linewidth'),
            Item('number_x_labels'),
            Item('number_y_labels'),
            Item('offscreen'),
            Item('number_of_contour_levels'),
            Item('show_projections'),
            Item('show_x_projection'),
            Item('show_y_projection'),
            Item('contour_exponent'),
            Item('contour_start'),
            Item('contour_alpha'),
            Item('colormap'),

                             ),
                       resizable=True
                      )

################################################################################
# `NDDatasetPreference` class
################################################################################
class NDDatasetPreferences(HasTraits):
    """ Preferences definition for SpectroChemPy. """

    #### Preferences ##########################################################.

    # Specifies the Data directory path.
    path_to_data = Str(desc='data directory path')

    # Specifies the name of the last use file for dataset.
    last_file = Str(desc='the name of the last use file for dataset')

    # The default extension for saving
    default_save_extension = Str(desc='The default extension for saving')

    ######################################################################
    # Traits UI view.

    traits_view = View(Group(
                             Item('path_to_data'),
                             Item('last_file'),
                             Item('default_save_extension')
                             ),
                       resizable=True
                      )
