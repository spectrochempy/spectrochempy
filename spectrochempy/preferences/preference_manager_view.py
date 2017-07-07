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
"""
A view for the SpectroChemPy preferences outside of Envisage.
"""
# adapted from mayavi 
# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2008,  Enthought, Inc.
# License: BSD Style.

# Standard library imports
from os.path import join

# Enthought library imports.
from traits.api import List, Bool
from apptools.preferences.ui.api import PreferencesManager, \
    PreferencesPage
from pyface.api import ImageResource
from pyface.resource.api import resource_path
from apptools.preferences.ui.preferences_node import PreferencesNode

# Local imports.
from spectrochempy.preferences.preferences_page import *

from spectrochempy.preferences.preference_manager import preference_manager
from spectrochempy.logger import log, consolelog, filelog

import os

__all__ = ['view_preferences']

################################################################################
# `PreferenceManagerView` class
################################################################################
class PreferenceManagerView(PreferencesManager):
    """ A preference manager UI for SpectroChemPy, to be used outside of
        Envisage.
    """

    # Path used to search for images
    _image_path = [join(resource_path(), 'images'), ]

    # The icon of the dialog
    icon = ImageResource('preferences.ico', search_path=_image_path)

    # The preference pages displayed
    pages = List(PreferencesPage)

    def _pages_default(self):
        return [
                RootPreferencesPage(
                    preferences=preference_manager.general.preferences),
                NDDatasetPreferencesPage(
                    preferences=preference_manager.dataset.preferences),
                PlotPreferencesPage(
                    preferences=preference_manager.plot.preferences),
                ]

    def dialog_view(self, kind='modal'):
        """ Poor-man's subclassing of view to overload size.
        """
        view = self.trait_view()
        view.width = 0.7
        view.height = 0.5
        view.title = 'SpectroChemPy preferences'
        view.icon = self.icon
        ui = self.edit_traits(
                view=view,
                kind=kind,
                scrollable=True,
                id='spectrochempy.preferences.preference_view')
        return ui

    def _get_root(self):
        """ Subclass the general getter, to work outside of envisage, with
            no well-defined general.
        """
        general = PreferencesNode(page=self.pages[0])
        for page in self.pages:
            general.append(PreferencesNode(page=page))
        return general

    def apply(self):
        super(PreferenceManagerView, self).apply()
        for page in self.pages:
            page.preferences.save()
        log.setLevel(self.pages[0].log_level_)
        consolelog.setLevel(self.pages[0].log_level_)
        filelog.setLevel(self.pages[0].log_file_level_)

    def __call__(self, kind='modal'):
        if not preference_manager.general._DO_NOT_BLOCK: # case we are building the docs or testing
            self.dialog_view(kind)


view_preferences = PreferenceManagerView()




