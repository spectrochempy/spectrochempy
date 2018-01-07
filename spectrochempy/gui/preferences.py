# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

# inspired by psyplot_gui.preferences

"""Dialog Preferences widget for SpectroChemPy

This module defines the :class:`DialogPreferences` widget that creates an
interface to the configuration file of SpectroChemPy

"""

from warnings import warn
import os

from ..extern.pyqtgraph.Qt import QtGui, QtCore
from .guiutils import geticon
from .widgets.parametertree import ParameterTree, Parameter

from spectrochempy.application import (app, log,
                                       preferences as general_preferences,
                                       plotter_preferences,
                                       project_preferences)


# ============================================================================
class Preference_Page(object):
    """The base class for the application preference pages"""

    title = None
    icon = None

    def initialize(self):
        raise NotImplementedError

# ============================================================================
class PreferencesTree(ParameterTree):


    def __init__(self, preferences, title=None, *args, **kwargs):
        """
        Parameters
        ----------
        preferences: object
            The Configurable object that contains the preferences

        """
        super(PreferencesTree, self).__init__(*args, **kwargs)
        self.preferences = preferences
        self.title = title

    def initialize(self, title=None, reset=False):
        """Fill the items into the tree"""

        if hasattr(self.preferences, 'traits'):

            pref_traits = self.preferences.traits(config=True)
            # we sorts traits using help text
            # we make a dictionary containing the traits and the current values
            preferences = {o[1]: (o[2] ,
                    getattr(self.preferences, o[1]) if not reset else
                    "RESET_TO_DEFAULT")
                    for o in sorted(
                       [(opt.help, k, opt)  for k,opt in pref_traits.items()]
                                   )}
        else:
            raise ValueError("preferences must be a Configurable object")

        p = Parameter.create(name=title,
                             title=title,
                             type='group',
                             children=preferences)

        self.setParameters(p, showTop=True)

        p.sigTreeStateChanged.connect(self.parameter_changed)

    def parameter_changed(self, par, changes):

        for opt, change, data in changes:
            path = par.childPath(opt)
            if path is not None:
                childName = '.'.join(path)
            else:
                childName = opt.name()

            if change == 'value':
                setattr(self.preferences, childName, data)
                pass


# ============================================================================
class PreferencePageWidget(Preference_Page, QtGui.QWidget):
    """A Widget for a spectrochempy preference page"""

    preferences = None  # implemented in subclass
    tree = None

    @property
    def changed(self):
        return True #bool(next(self.tree.changed_preferences(), None))

    @property
    def icon(self):
        return QtGui.QIcon(geticon('preferences.png'))

    def __init__(self, *args, **kwargs):
        super(PreferencePageWidget, self).__init__(*args, **kwargs)
        self.vbox = vbox = QtGui.QVBoxLayout()

        self.tree = tree = PreferencesTree(self.preferences, parent=self, \
                           showHeader=False, **kwargs)

        vbox.addWidget(self.tree)
        hbox = QtGui.QHBoxLayout()
        vbox.addLayout(hbox)
        self.setLayout(vbox)

    def save_settings_action(self, update=False, target=None):
        """Create an action to save the selected settings in the :attr:`tree`

        Parameters
        ----------
        update: bool
            If True, it is expected that the file already exists and it will be
            updated. Otherwise, existing files will be overwritten
        """
        def func():
            if update:
                meth = QtGui.QFileDialog.getOpenFileName
            else:
                meth = QtGui.QFileDialog.getSaveFileName
            if target is None:
                fname = meth(
                    self, 'Select a file to %s' % (
                        'update' if update else 'create'),
                    self.default_path,
                    'YAML files (*.yml);;'
                    'All files (*)'
                    )
                fname = fname[0]
            else:
                fname = target
            if not fname:
                return
            if update:
                preferences = self.preferences.__class__(defaultParams=self.preferences.defaultParams)
                preferences.load_from_file(fname)
                old_keys = list(preferences)
                selected = dict(self.tree.selected_preferences())
                new_keys = list(selected)
                preferences.update(selected)
                preferences.dump(fname, include_keys=old_keys + new_keys,
                        exclude_keys=[])
            else:
                preferences = self.preferences.__class__(self.tree.selected_preferences(),
                                       defaultParams=self.preferences.defaultParams)
                preferences.dump(fname, exclude_keys=[])

        action = QtGui.QAction('Update...' if update else 'Overwrite...', self)
        action.triggered.connect(func)
        return action

    def initialize(self, preferences=None, reset=False):
        """Initialize the config page

        Parameters
        ----------
        preferences: object

        """
        if preferences is not None:
            self.preferences = preferences
            self.tree.preferences = preferences
        self.tree.initialize(title=self.title, reset=reset)

# ============================================================================
class GeneralPreferencePageWidget(PreferencePageWidget):

    preferences = general_preferences
    title = 'General preferences'


# ============================================================================
class ProjectPreferencePageWidget(PreferencePageWidget):

    preferences = project_preferences
    title = 'Project preferences'


# ============================================================================
class PlotPreferencePageWidget(PreferencePageWidget):

    preferences = plotter_preferences
    title = 'Plotter preferences'


# ============================================================================
class DialogPreferences(QtGui.QDialog):
    """Preferences dialog"""

    @property
    def pages(self):
        return map(self.get_page, range(self.pages_widget.count()))

    def __init__(self, main=None):
        super(DialogPreferences, self).__init__(parent=main)
        self.setWindowTitle('Preferences')
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.setWindowFlags(
            QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint |
            QtCore.Qt.WindowTitleHint |
            QtCore.Qt.WindowStaysOnTopHint)
        # Widgets
        self.pages_widget = QtGui.QStackedWidget()
        self.contents_widget = QtGui.QListWidget()
        self.bt_reset = QtGui.QPushButton('Reset to defaults')

        self.bbox = bbox = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok)

        # Widgets setup
        # Destroying the C++ object right after closing the dialog box,
        # otherwise it may be garbage-collected in another QThread
        # (e.g. the editor's analysis thread in Spyder), thus leading to
        # a segmentation fault on UNIX or an application crash on Windows
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Preferences')
        self.contents_widget.setMovement(QtGui.QListView.Static)
        self.contents_widget.setSpacing(1)
        self.contents_widget.setCurrentRow(0)

        # Layout
        hsplitter = QtGui.QSplitter()
        hsplitter.addWidget(self.contents_widget)
        hsplitter.addWidget(self.pages_widget)
        hsplitter.setStretchFactor(1, 1)

        btnlayout = QtGui.QHBoxLayout()
        btnlayout.addWidget(self.bt_reset)
        btnlayout.addStretch(1)
        btnlayout.addWidget(bbox)

        vlayout = QtGui.QVBoxLayout()
        vlayout.addWidget(hsplitter)
        vlayout.addLayout(btnlayout)

        self.setLayout(vlayout)

        # Signals and slots
        if main is not None:
            self.bt_reset.clicked.connect(main.reset_preferences)
        self.pages_widget.currentChanged.connect(self.current_page_changed)
        self.contents_widget.currentRowChanged.connect(
            self.pages_widget.setCurrentIndex)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)

    def set_current_index(self, index):
        """Set current page index"""
        self.contents_widget.setCurrentRow(index)

    def current_page_changed(self, index):
        preference_page = self.get_page(index)

    def get_page(self, index=None):
        """Return page widget"""
        if index is None:
            widget = self.pages_widget.currentWidget()
        else:
            widget = self.pages_widget.widget(index)
        return widget.widget()


    def add_page(self, widget):
        """Add a new page to the preferences dialog

        Parameters
        ----------
        widget: Preference_Page
            The page to add"""

        scrollarea = QtGui.QScrollArea(self)
        scrollarea.setWidgetResizable(True)
        scrollarea.setWidget(widget)
        self.pages_widget.addWidget(scrollarea)
        item = QtGui.QListWidgetItem(self.contents_widget)
        try:
            item.setIcon(widget.icon)
        except TypeError:
            pass
        item.setText(widget.title)
        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        item.setSizeHint(QtCore.QSize(0, 25))
