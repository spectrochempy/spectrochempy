"""A preference manager for all spectrochempy related preferences.

The idea behind this module is that it lets the spectrochempy
library/application use the same preferences by managing them no matter
if spectrochempy is used as an application (via envisage3) or as a library.

The preferences helpers are divided into different categories for
different kinds of preferences.  Currently the following are available.

  - general: for global spectrochempy preferences of the form
    'spectrochempy.preference'.

For more details on the general preferences support in enthought, please
read the documentation for apptools.preferences (part of the AppTools
package).

"""
# adapted from MAyavi preferences
#
# Author: Prabhu Ramachandran <prabhu [at] aero . iitb . ac . in>
# Copyright (c) 2008,  Enthought, Inc.
# License: BSD Style.

# Standard library imports
from os.path import join
import pkg_resources

# Enthought library imports.
from traits.etsconfig.api import ETSConfig
from traits.api import HasTraits, Instance
from traitsui.api import View, Group, Item
from apptools.preferences.api import (ScopedPreferences, IPreferences,
        PreferencesHelper)
from spectrochempy.utils import get_config_dir

# Local imports.
from .preferences_helpers import *
from ..logger import log, consolelog, filelog


################################################################################
# `PreferenceManager` class
################################################################################
class PreferenceManager(HasTraits):

    # The general preferences helper for preferences of the form
    # 'spectrochempy.preference'.
    general = Instance(PreferencesHelper)

    # The dataset preferences helper for preferences of the form
    # 'spectrochempy.dataset.preference'.
    dataset = Instance(PreferencesHelper)

    # The plot preferences helper for preferences of the form
    # 'spectrochempy.plot.preference'.
    plot = Instance(PreferencesHelper)

    # The preferences.
    preferences = Instance(IPreferences)

    ######################################################################
    # Traits UI view.

    traits_view = View(Group(
                           Group(Item(name='general', style='custom'),
                                 show_labels=False, label='Root',
                                 show_border=True
                                ),
                           Group(Item(name='dataset', style='custom'),
                                 show_labels=False, label='NDDataset',
                                 show_border=True,
                                ),
                            Group(Item(name='plot', style='custom'),
                                  show_labels=False, label='Plot',
                                  show_border=True,
                                  ),
                            ),
                       buttons=['OK', 'Cancel'],
                       resizable=True
                      )

    ######################################################################
    # `HasTraits` interface.
    ######################################################################
    def __init__(self, **traits):
        super(PreferenceManager, self).__init__(**traits)

        if 'preferences' not in traits:
            self._load_preferences()

    def _preferences_default(self):
        """Trait initializer."""
        return ScopedPreferences()

    def _general_default(self):
        """Trait initializer."""
        return RootPreferencesHelper(preferences=self.preferences)

    def _dataset_default(self):
        """Trait initializer."""
        return NDDatasetPreferencesHelper(preferences=self.preferences)

    def _plot_default(self):
        """Trait initializer."""
        return PlotPreferencesHelper(preferences=self.preferences)

    ######################################################################
    # Private interface.
    ######################################################################
    def _load_preferences(self):
        """Load the default preferences."""

        # Save current application_home.
        app_home = ETSConfig.get_application_home()
        # Set it to where the spectrochempy preferences are temporarily.
        path = get_config_dir()
        ETSConfig.application_home = path
        try:
            for pkg in ('spectrochempy.preferences', ):
                pref = 'preferences.ini'
                pref_file = pkg_resources.resource_stream(pkg, pref)
                preferences = self.preferences
                default = preferences.node('default/')
                default.load(pref_file)
                pref_file.close()
        finally:
            ETSConfig.application_home = app_home

    def _preferences_changed(self, preferences):
        """Setup the helpers if the preferences trait changes."""
        for helper in (self.general, self.dataset, self.plot):
            helper.preferences = preferences

    def save(self):
        self.preferences.save()

##########################################################
# A Global preference manager that all other modules can use.

preference_manager = PreferenceManager()

