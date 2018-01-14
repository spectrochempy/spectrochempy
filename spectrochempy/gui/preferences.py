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
from textwrap import fill

from ..extern.pyqtgraph.Qt import QtGui, QtCore
from .guiutils import geticon
from .widgets.parametertree import ParameterTree, Parameter
from traitlets.config.manager import BaseJSONConfigManager

from spectrochempy.application import (app, log)

from .widgets.commonwidgets import warningMessage


# ============================================================================
class Preference_Page(object):
    """The base class for the application preference pages"""

    title = None
    running_title = None
    tree_title = None
    icon = None

    def initialize(self):
        raise NotImplementedError


# ============================================================================
class PreferencesTree(ParameterTree):

    def __init__(self, preferences, *args, **kwargs):
        """
        Parameters
        ----------
        preferences: object
            The Configurable object that contains the preferences

        """
        super(PreferencesTree, self).__init__(*args, **kwargs)
        self.preferences = preferences

    def initialize(self, title=None):
        """Fill the items into the tree"""

        self.pname = app.last_project

        if hasattr(self.preferences, 'traits'):

            pref_traits = self.preferences.traits(config=True)
            # we sorts traits using help text
            # we make a dictionary containing the traits and the current values
            preferences = {o[1]: (o[2], getattr(self.preferences, o[
                1])) for o in sorted(
                [(opt.help, k, opt) for k, opt in pref_traits.items(

                )])}
        else:
            raise ValueError("preferences must be a Configurable object")

        self.p = p = Parameter.create(name=title, title=title, type='group',
                             children=preferences)

        self.setParameters(p, showTop=True)

        p.sigTreeStateChanged.connect(self.parameter_changed)

        # savestate
        self.savedstate = p.saveState(filter = 'user')

    def parameter_changed(self, par, changes):

        for opt, change, data in changes:
            path = par.childPath(opt)
            if path is not None:
                childname = '.'.join(path)
            else:
                childname = opt.name()

            if change == 'value':
                # Change the values of the preference
                data = self._sanitize(childname, data)
                setattr(self.preferences, childname,data)
                # the main  json configuration file is uddated automatically
                # when parameters changes

    def _sanitize(self, name, data):
        # make a string from special type compatible with traits definition

        # color:
        if name in self.preferences.traits(type='color'):
            if isinstance(data, tuple):
                return '#%02x%02x%02x' % data[:3]
            return data.name()

        return data

# ============================================================================
class PreferencePageWidget(Preference_Page, QtGui.QWidget):
    """A Widget for a spectrochempy preference page"""

    preferences = None  # implemented in subclass
    tree = None

    @property
    def changed(self):
        return self.tree.p.saveState(filter='user')!=self.tree.savedstate

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

    def initialize(self, preferences=None):
        """Initialize the config page

        Parameters
        ----------
        preferences: object

        """
        if preferences is not None:
            self.preferences = preferences
            self.tree.preferences = preferences

        current_project = self.preferences.parent.last_project
        self.running_title = self.title.format(current_project=current_project)
        tree_title = self.tree_title.format(current_project=current_project)
        self.tree.initialize(title=tree_title)


# ============================================================================
class GeneralPreferencePageWidget(PreferencePageWidget):
    preferences = app.general_preferences
    tree_title = title = 'General preferences'


# ============================================================================
class ProjectPreferencePageWidget(PreferencePageWidget):
    preferences = app.project_preferences
    title = 'Project:{current_project}'
    tree_title = 'Project `{current_project}` preferences'


# ============================================================================
class DialogPreferences(QtGui.QDialog):
    """Preferences dialog"""

    reset = False

    def __init__(self, main=None):
        super(DialogPreferences, self).__init__(parent=main)
        self.setWindowTitle('Preferences')
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.setWindowFlags(
            QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint |
            QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowStaysOnTopHint)
        # Widgets
        self.pages_widget = QtGui.QStackedWidget()
        self.contents_widget = QtGui.QListWidget()
        self.bt_reset = QtGui.QPushButton('Reset to defaults')

        self.bbox = bbox = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel)

        # Widgets setup
        # Destroying the C++ object right after closing the dialog box,
        # otherwise it may be garbage-collected in another QThread
        # (e.g. the editor's analysis thread in Spyder), thus leading to
        # a segmentation fault on UNIX or an application crash on Windows
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('SpectroChemPy Configuration')
        self.contents_widget.setMovement(QtGui.QListView.Static)
        self.contents_widget.setSpacing(2)
        self.contents_widget.setCurrentRow(0)

        # Layout
        hsplitter = QtGui.QSplitter()
        hsplitter.addWidget(self.contents_widget)
        hsplitter.addWidget(self.pages_widget)
        hsplitter.setSizes([170, -1])
        hsplitter.setStretchFactor(1,4)

        btnlayout = QtGui.QHBoxLayout()
        btnlayout.addWidget(self.bt_reset)
        btnlayout.addStretch(1)
        btnlayout.addWidget(bbox)

        vlayout = QtGui.QVBoxLayout()
        vlayout.addWidget(hsplitter)
        vlayout.addLayout(btnlayout)

        self.setLayout(vlayout)

        # Signals and slots
        self.bt_reset.clicked.connect(main.reset_preferences)
        self.pages_widget.currentChanged.connect(self.current_page_changed)
        self.contents_widget.currentRowChanged.connect(
            self.pages_widget.setCurrentIndex)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)

    def get_page(self, index=None):
        """Return page widget"""
        if index is None:
            widget = self.pages_widget.currentWidget()
        else:
            widget = self.pages_widget.widget(index)
        return widget.widget()

    @property
    def pages(self):
        return map(self.get_page, range(self.pages_widget.count()))

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
        item.setText(widget.running_title)
        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        item.setSizeHint(QtCore.QSize(0, 25))

    def set_current_index(self, index):
        """Set current page index"""
        self.contents_widget.setCurrentRow(index)

    def current_page_changed(self, index):
        self.current_page = self.get_page(index)

    def reject(self):
        """
        Reject current changes

        """

        if not self.reset and not warningMessage(self,
            message= 'Are you sure to cancel all current preference changes'):
            return

        for page in self.pages:
            page.tree.p.restoreState(page.tree.savedstate)

        super().reject()
