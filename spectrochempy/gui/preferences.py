# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================



"""Preferences widget for psyplot_gui

This module defines the :class:`Preferences` widget that creates an interface
to the optionsParams of psyplot and psyplot_gui"""
import yaml
from warnings import warn

from ..extern.pyqtgraph.Qt import QtGui, QtCore
from .guiutils import geticon

# noinspection PyUnresolvedReferences
from spectrochempy.api import (options as general_options,
                               projectsoptions as project_options,
                               plotoptions as plot_options)

class Preference_Page(object):
    """An abstract base class for the application preference pages"""

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


class OptionsTree(QtGui.QTreeWidget):
    """A QTreeWidget that can be used to display a optionsParams instance

    This widget is populated by a :class:`psyplot.config.optionssetup.optionsParams`
    instance and displays whether the values are valid or not"""

    #: A signal that shall be emitted if the validation state changes
    validChanged = QtCore.pyqtSignal(bool)

    #: A signal that is emitted if changes are propsed. It is either emitted
    #: with the parent of this instance (if this is not None) or with the
    #: instance itself
    propose_changes = QtCore.pyqtSignal(object)

    #: The :class:`~psyplot.config.optionssetup.optionsParams` to display
    options = None

    #: list of :class:`bool`. A boolean for each optionsParams key that states
    #: whether the proposed value is valid or not
    valid = []

    value_col = 2

    def __init__(self, options, *args, **kwargs):
        """
        Parameters
        ----------
        options: object
            The Configurable object that contains the options

        """
        super(OptionsTree, self).__init__(*args, **kwargs)
        self.options = options

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_menu)
        self.setColumnCount(self.value_col + 1)
        self.setHeaderLabels(['Options', '', 'Value'])

    @property
    def is_valid(self):
        """True if all the proposed values in this tree are valid"""
        return all(self.valid)

    @property
    def top_level_items(self):
        """An iterator over the topLevelItems in this tree"""
        return map(self.topLevelItem, range(self.topLevelItemCount()))

    def initialize(self):
        """Fill the items of the :attr:`options` into the tree"""
        optionsParams = self.options
        descriptions = self.descriptions
        self.valid = [True] * len(optionsParams)
        validators = self.validators
        vcol = self.value_col
        for i, (key, val) in enumerate(sorted(optionsParams.items())):
            item = QtGui.QTreeWidgetItem(0)
            item.setText(0, key)
            item.setToolTip(0, key)
            item.setIcon(1, QtGui.QIcon(geticon('valid.png')))
            desc = descriptions.get(key)
            if desc:
                item.setText(vcol, desc)
                item.setToolTip(vcol, desc)
            child = QtGui.QTreeWidgetItem(0)
            item.addChild(child)
            self.addTopLevelItem(item)
            editor = QtGui.QTextEdit(self)
            # set maximal height of the editor to 3 rows
            editor.setMaximumHeight(
                4 * QtGui.QFontMetrics(editor.font()).height())
            editor.setPlainText(yaml.dump(val))
            self.setItemWidget(child, vcol, editor)
            editor.textChanged.connect(
                self.set_icon_func(i, item, validators[key]))
        self.resizeColumnToContents(0)
        self.resizeColumnToContents(1)

    def set_icon_func(self, i, item, validator):
        """Create a function to change the icon of one topLevelItem

        This method creates a function that can be called when the value of an
        item changes to display it's valid state. The returned function changes
        the icon of the given topLevelItem depending on
        whether the proposed changes are valid or not and it modifies the
        :attr:`valid` attribute accordingly

        Parameters
        ----------
        i: int
            The index of the topLevelItem
        item: QTreeWidgetItem
            The topLevelItem
        validator: func
            The validation function

        Returns
        -------
        function
            The function that can be called to set the correct icon"""
        def func():
            editor = self.itemWidget(item.child(0), self.value_col)
            s = asstring(editor.toPlainText())
            try:
                val = yaml.load(s)
            except Exception as e:
                item.setIcon(1, QtGui.QIcon(geticon('warning.png')))
                item.setToolTip(1, "Could not parse yaml code: %s" % e)
                self.set_valid(i, False)
                return
            try:
                validator(val)
            except Exception as e:
                item.setIcon(1, QtGui.QIcon(geticon('invalid.png')))
                item.setToolTip(1, "Wrong value: %s" % e)
                self.set_valid(i, False)
            else:
                item.setIcon(1, QtGui.QIcon(geticon('valid.png')))
                self.set_valid(i, True)
            self.propose_changes.emit(self.parent() or self)
        return func

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

    def open_menu(self, position):
        """Open a menu to expand and collapse all items in the tree

        Parameters
        ----------
        position: QPosition
            The position where to open the menu"""
        menu = QtGui.QMenu()
        expand_all_action = QtGui.QAction('Expand all', self)
        expand_all_action.triggered.connect(self.expandAll)
        menu.addAction(expand_all_action)
        collapse_all_action = QtGui.QAction('Collapse all', self)
        collapse_all_action.triggered.connect(self.collapseAll)
        menu.addAction(collapse_all_action)
        menu.exec_(self.viewport().mapToGlobal(position))

    def changed_options(self, use_items=False):
        """Iterate over the changed optionsParams

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
        for t in self._get_options(equals):
            yield t[0 if use_items else 1], t[2]

    def selected_options(self, use_items=False):
        """Iterate over the selected optionsParams

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
        for t in self._get_options(is_selected):
            yield t[0 if use_items else 1], t[2]

    def _get_options(self, filter_func=None):
        """Iterate over the optionsParams

        This function applies the given `filter_func` to check whether the
        item should be included or not

        Parameters
        ----------
        filter_func: function
            A function that accepts the following arguments:

            item
                The QTreeWidgetItem
            key
                The optionsParams key
            val
                The proposed value
            orig
                The current value

        Yields
        ------
        QTreeWidgetItem
            The corresponding topLevelItem
        str
            The optionsParams key
        object
            The proposed value
        object
            The current value
        """
        def no_check(item, key, val, orig):
            return True
        options = self.options
        filter_func = filter_func or no_check
        for item in self.top_level_items:
            key = asstring(item.text(0))
            editor = self.itemWidget(item.child(0), self.value_col)
            val = yaml.load(asstring(editor.toPlainText()))
            try:
                val = options.validate[key](val)
            except:
                pass
            try:
                include = filter_func(item, key, val, options[key])
            except:
                warn('Could not check state for %s key' % key,
                     RuntimeWarning)
            else:
                if include:
                    yield (item, key, val, options[key])

    def apply_changes(self):
        """Update the :attr:`options` with the proposed changes"""
        new = dict(self.changed_options())
        if new != self.options:
            self.options.update(new)

    def select_changes(self):
        """Select all the items that changed comparing to the current optionsParams
        """
        for item, val in self.changed_options(True):
            item.setSelected(True)


class OptionsWidget(Preference_Page, QtGui.QWidget):
    """A preference page for the spectrochempy options"""

    options = None  # implemented in subclass
    tree = None

    @property
    def changeProposed(self):
        return self.tree.changeProposed

    @property
    def validChanged(self):
        return self.tree.validChanged

    @property
    def changed(self):
        return bool(next(self.tree.changed_options(), None))

    @property
    def is_valid(self):
        return self.tree.is_valid

    @property
    def icon(self):
        return QtGui.QIcon(geticon('options.png'))

    def __init__(self, *args, **kwargs):
        super(OptionsWidget, self).__init__(*args, **kwargs)
        self.vbox = vbox = QtGui.QVBoxLayout()

        self.description = QtGui.QLabel(
            '<p>Modify the optionsParams for your need. Changes will not be applied'
            ' until you click the Apply or Ok button.</p>'
            '<p>Values must be entered in yaml syntax</p>', parent=self)
        vbox.addWidget(self.description)
        self.tree = tree = OptionsTree(
            self.options, getattr(self.options, 'validate', None),
            getattr(self.options, 'descriptions', None), parent=self)
        tree.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
        vbox.addWidget(self.tree)

        self.bt_select_all = QtGui.QPushButton('Select All', self)
        self.bt_select_changed = QtGui.QPushButton('Select changes', self)
        self.bt_select_none = QtGui.QPushButton('Clear Selection', self)
        self.bt_export = QtGui.QToolButton(self)
        self.bt_export.setText('Export Selection...')
        self.bt_export.setToolTip('Export the selected optionsParams to a file')
        self.bt_export.setPopupMode(QtGui.QToolButton.InstantPopup)
        self.export_menu = export_menu = QtGui.QMenu(self)
        export_menu.addAction(self.save_settings_action())
        export_menu.addAction(self.save_settings_action(True))
        self.bt_export.setMenu(export_menu)
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.bt_select_all)
        hbox.addWidget(self.bt_select_changed)
        hbox.addWidget(self.bt_select_none)
        hbox.addStretch(1)
        hbox.addWidget(self.bt_export)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        self.bt_select_all.clicked.connect(self.tree.selectAll)
        self.bt_select_none.clicked.connect(self.tree.clearSelection)
        self.bt_select_changed.clicked.connect(self.tree.select_changes)

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
                options = self.options.__class__(defaultParams=self.options.defaultParams)
                options.load_from_file(fname)
                old_keys = list(options)
                selected = dict(self.tree.selected_options())
                new_keys = list(selected)
                options.update(selected)
                options.dump(fname, include_keys=old_keys + new_keys,
                        exclude_keys=[])
            else:
                options = self.options.__class__(self.tree.selected_options(),
                                       defaultParams=self.options.defaultParams)
                options.dump(fname, exclude_keys=[])

        action = QtGui.QAction('Update...' if update else 'Overwrite...', self)
        action.triggered.connect(func)
        return action

    def initialize(self, optionsParams=None, validators=None, descriptions=None):
        """Initialize the config page

        Parameters
        ----------
        optionsParams: dict
            The optionsParams to use. If None, the :attr:`options` attribute of this
            instance is used
        validators: dict
            A mapping from the `optionsParams` key to the corresponding validation
            function for the value. If None, the
            :attr:`~psyplot.config.optionssetup.optionsParams.validate` attribute of the
            :attr:`options` attribute is used
        descriptions: dict
            A mapping from the `optionsParams` key to it's description. If None, the
            :attr:`~psyplot.config.optionssetup.optionsParams.descriptions` attribute of
            the :attr:`options` attribute is used"""
        if optionsParams is not None:
            self.options = optionsParams
            self.tree.options = optionsParams
        if validators is not None:
            self.tree.validators = validators
        if descriptions is not None:
            self.tree.descriptions = descriptions
        self.tree.initialize()

    def apply_changes(self):
        """Apply the changes in the config page"""
        self.tree.apply_changes()


class GeneralOptionsWidget(OptionsWidget):

    options = general_options

    title = 'Application preferences'


class ProjectOptionsWidget(OptionsWidget):

    options = project_options

    title = 'Project preferences'


class PlotOptionsWidget(OptionsWidget):
    options = plot_options

    title = 'Plotting preferences'


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
        #    'Load the optionsParams for the plugins in separate pages')

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
            pass #self.bt_reset.clicked.connect(main.reset_optionsParams)
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
        configpage = self.get_page(index)
        self.bt_apply.setVisible(not configpage.auto_updates)
        self.check_changes(configpage)

    def get_page(self, index=None):
        """Return page widget"""
        if index is None:
            widget = self.pages_widget.currentWidget()
        else:
            widget = self.pages_widget.widget(index)
        return widget.widget()

    def accept(self):
        """Reimplement Qt method"""
        for configpage in self.pages:
            if not configpage.is_valid:
                continue
            configpage.apply_changes()
        QtGui.QDialog.accept(self)

    def apply_clicked(self):
        # Apply button was clicked
        configpage = self.get_page()
        if configpage.is_valid:
            configpage.apply_changes()
        self.check_changes(configpage)

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
        widget.propose_changes.connect(self.check_changes)

    def check_changes(self, configpage):
        """Enable the apply button if there are changes to the settings"""
        if configpage != self.get_page():
            return
        self.bt_apply.setEnabled(
            not configpage.auto_updates and configpage.is_valid and
            configpage.changed)

    def load_plugin_pages(self):
        """Load the optionsParams for the plugins in separate pages"""
        validators = psy_optionsParams.validate
        descriptions = psy_optionsParams.descriptions
        for ep in psy_optionsParams._load_plugin_entrypoints():
            plugin = ep.load()
            options = getattr(plugin, 'optionsParams', None)
            if options is None:
                options = optionsParams()
            w = OptionsWidget(parent=self)
            w.title = 'optionsParams of ' + ep.module_name
            w.default_path = PsyoptionsParamsWidget.default_path
            w.initialize(optionsParams=options, validators=validators,
                         descriptions=descriptions)
            # use the full optionsParams after initialization
            w.options = psy_optionsParams
            self.add_page(w)
