# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

# inspired by psyplot_gui.preferences

"""Preferences widget for SpectroChemPy

This module defines the :class:`Preferences` widget that creates an interface
to the configuration file  of SpectroChemPy

"""

from warnings import warn

from ..extern.pyqtgraph.Qt import QtGui, QtCore
from .guiutils import geticon
from .widgets.parametertree import ParameterTree, Parameter

from spectrochempy.application import app

general_preferences = app.general_preferences
project_preferences = app.project_preferences
plotter_preferences = app.plotter_preferences


# ============================================================================
class Preference_Page(object):
    """The base class for the application preference pages"""

    validChanged = QtCore.pyqtSignal(bool)
    changeProposed = QtCore.pyqtSignal(object)

    title = None
    icon = None
    auto_updates = False

    @property
    def is_valid(self):
        raise NotImplementedError

    @property
    def changed(self):
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError

    def apply_changes(self):
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

    @property
    def top_level_items(self):
        """An iterator over the topLevelItems in this tree"""
        return map(self.topLevelItem, range(self.topLevelItemCount()))

    def initialize(self, title=None):
        """Fill the items into the tree"""

        # we may have passed some config preferences or a dictionary
        if hasattr(self.preferences, 'traits'):
            preferences = self.preferences.traits(config=True)
            # sorting using title
            preferences = {o[1]:o[2] for o in sorted([(opt.help, k, opt) for k,opt in preferences.items()])}
        else:
            preferences = self.preferences

        p = Parameter.create(name=title,
                             title=title,
                             type='group',
                             children=preferences)
        self.setParameters(p, showTop=True)

        p.sigTreeStateChanged.connect(self.parameter_changed)

    def parameter_changed(self, par, changes):

        for opt, change, data in changes:
            path = p.childPath(param)
            if path is not None:
                childName = '.'.join(path)
            else:
                childName = param.name()

            if childName == "Project.usempl":
                log.debug('%s : %s\n' % (childName, data))
                self.usempl = data
                for key in self.open_plots.keys():
                    self.show_or_create_plot_dock(key)
                self.project_item_clicked()

    def set_valid(self, i, b):
        """Set the validation status

        If the validation status changed compared to the old one, the
        :attr:`validChanged` signal is emitted

        Parameters
        ----------
        i: int
            The index of the topLevelItem
        b: bool
            The valid state of the item
        """
        old = self.is_valid
        self.valid[i] = b
        new = self.is_valid
        if new is not old:
            self.validChanged.emit(new)


    def changed_preferences(self, use_items=False):
        """Iterate over the changed preferencesParams

        Parameters
        ----------
        use_items: bool
            If True, the topLevelItems are used instead of the keys

        Yields
        ------
        QTreeWidgetItem or str
            The item identifier
        object
            The proposed value"""
        def equals(item, key, val, orig):
            return val != orig
        for t in self._get_preferences(equals):
            yield t[0 if use_items else 1], t[2]

    def selected_preferences(self, use_items=False):
        """Iterate over the selected preferencesParams

        Parameters
        ----------
        use_items: bool
            If True, the topLevelItems are used instead of the keys

        Yields
        ------
        QTreeWidgetItem or str
            The item identifier
        object
            The proposed value"""
        def is_selected(item, key, val, orig):
            return item.isSelected()
        for t in self._get_preferences(is_selected):
            yield t[0 if use_items else 1], t[2]

    def _get_preferences(self, filter_func=None):
        """Iterate over the preferencesParams

        This function applies the given `filter_func` to check whether the
        item should be included or not

        Parameters
        ----------
        filter_func: function
            A function that accepts the following arguments:

            item
                The QTreeWidgetItem
            key
                The preferencesParams key
            val
                The proposed value
            orig
                The current value

        Yields
        ------
        QTreeWidgetItem
            The corresponding topLevelItem
        str
            The preferencesParams key
        object
            The proposed value
        object
            The current value
        """
        def no_check(item, key, val, orig):
            return True
        preferences = self.preferences
        filter_func = filter_func or no_check
        for item in self.top_level_items:
            key = item.text(0)
            editor = self.itemWidget(item.child(0), self.value_col)
            val = yaml.load(asstring(editor.toPlainText()))
            try:
                val = preferences.validate[key](val)
            except:
                pass
            try:
                include = filter_func(item, key, val, preferences[key])
            except:
                warn('Could not check state for %s key' % key,
                     RuntimeWarning)
            else:
                if include:
                    yield (item, key, val, preferences[key])

    def apply_changes(self):
        """Update the :attr:`preferences` with the proposed changes"""
        new = dict(self.changed_preferences())
        if new != self.preferences:
            self.preferences.update(new)

    def select_changes(self):
        """Select all the items that changed comparing to the current preferencesParams
        """
        for item, val in self.changed_preferences(True):
            item.setSelected(True)


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

        self.tree = tree = preferencesTree(self.preferences, parent=self, \
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

    def initialize(self, preferences=None):
        """Initialize the config page

        Parameters
        ----------
        preferences: object

        """
        if preferences is not None:
            self.preferences = preferences
            self.tree.preferences = preferences
        self.tree.initialize(title=self.title)

    def apply_changes(self):
        """Apply the changes in the config page"""
        self.tree.apply_changes()


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
    title = 'Plotting preferences'


# ============================================================================
class Preferences(QtGui.QDialog):
    """Preferences dialog"""

    @property
    def bt_apply(self):
        return self.bbox.button(QtGui.QDialogButtonBox.Apply)

    @property
    def pages(self):
        return map(self.get_page, range(self.pages_widget.count()))

    def __init__(self, main=None):
        super(Preferences, self).__init__(parent=main)
        self.setWindowTitle('Preferences')

        # Widgets
        self.pages_widget = QtGui.QStackedWidget()
        self.contents_widget = QtGui.QListWidget()
        self.bt_reset = QtGui.QPushButton('Reset to defaults')
        #self.bt_load_plugins = QtGui.QPushButton('Load plugin pages')
        #self.bt_load_plugins.setToolTip(
        #    'Load the preferencesParams for the plugins in separate pages')

        self.bbox = bbox = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Apply |
            QtGui.QDialogButtonBox.Cancel)

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
        #btnlayout.addWidget(self.bt_load_plugins)
        btnlayout.addStretch(1)
        btnlayout.addWidget(bbox)

        vlayout = QtGui.QVBoxLayout()
        vlayout.addWidget(hsplitter)
        vlayout.addLayout(btnlayout)

        self.setLayout(vlayout)

        # Signals and slots
        if main is not None:
            pass #self.bt_reset.clicked.connect(main.reset_preferencesParams)
        #self.bt_load_plugins.clicked.connect(self.load_plugin_pages)
        self.pages_widget.currentChanged.connect(self.current_page_changed)
        self.contents_widget.currentRowChanged.connect(
            self.pages_widget.setCurrentIndex)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        self.bt_apply.clicked.connect(self.apply_clicked)
        self.bt_apply.setEnabled(False)

    def set_current_index(self, index):
        """Set current page index"""
        self.contents_widget.setCurrentRow(index)

    def current_page_changed(self, index):
        preference_page = self.get_page(index)
        self.bt_apply.setVisible(not preference_page.auto_updates)
        self.check_changes(preference_page)

    def get_page(self, index=None):
        """Return page widget"""
        if index is None:
            widget = self.pages_widget.currentWidget()
        else:
            widget = self.pages_widget.widget(index)
        return widget.widget()

    def accept(self):
        """Reimplement Qt method"""
        for preference_page in self.pages:
            if not preference_page.is_valid:
                continue
            preference_page.apply_changes()
        QtGui.QDialog.accept(self)

    def apply_clicked(self):
        # Apply button was clicked
        preference_page = self.get_page()
        if preference_page.is_valid:
            preference_page.apply_changes()
        self.check_changes(preference_page)

    def add_page(self, widget):
        """Add a new page to the preferences dialog

        Parameters
        ----------
        widget: Preference_Page
            The page to add"""
        widget.validChanged.connect(self.bt_apply.setEnabled)
        widget.validChanged.connect(
            self.bbox.button(QtGui.QDialogButtonBox.Ok).setEnabled)
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
        widget.changeProposed.connect(self.check_changes)

    def check_changes(self, preference_page):
        """Enable the apply button if there are changes to the settings"""
        if preference_page != self.get_page():
            return
        self.bt_apply.setEnabled(
            not preference_page.auto_updates and
            preference_page.changed)

    def load_plugin_pages(self):
        """Load the preferencesParams for the plugins in separate pages"""
        validators = psy_preferencesParams.validate
        descriptions = psy_preferencesParams.descriptions
        for ep in psy_preferencesParams._load_plugin_entrypoints():
            plugin = ep.load()
            preferences = getattr(plugin, 'preferencesParams', None)
            if preferences is None:
                preferences = preferencesParams()
            w = PreferencePageWidget(parent=self)
            w.title = 'preferencesParams of ' + ep.module_name
            w.default_path = PsypreferencesParamsWidget.default_path
            w.initialize(preferencesParams=preferences, validators=validators,
                         descriptions=descriptions)
            # use the full preferencesParams after initialization
            w.preferences = psy_preferencesParams
            self.add_page(w)
