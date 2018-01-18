# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

__all__ = []

# ----------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------

import os
import sys

from functools import partial

# ----------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------

from traitlets import HasTraits, Instance

# ----------------------------------------------------------------------------
# local imports
# ----------------------------------------------------------------------------

from ..extern import pyqtgraph as pg
from ..extern.pyqtgraph.console import ConsoleWidget
from ..extern.pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from ..extern.pyqtgraph.dockarea import DockArea, Dock

from .widgets.projecttreewidget import ProjectTreeWidget
from .logtoconsole import QtHandler, redirectoutput
from .plots import Plots
from .widgets.commonwidgets import warningMessage
from .preferences import (DialogPreferences, ProjectPreferencePageWidget,
                          GeneralPreferencePageWidget, )
from .guiutils import geticon

from ..application import (app, log, DEBUG, __release__, long_description)

from ..core import Project, Script, NDDataset


# ============================================================================
class _metaclass_mixin(QtWidgets.QWidget.__class__, HasTraits.__class__):
    # This is necessary to be able to mix HasTraits with QtWidget class.
    pass


# ============================================================================
class MainWindow(HasTraits, Plots, QtGui.QMainWindow,
                 metaclass=_metaclass_mixin):
    project = Instance(Project)
    "Current project instance"

    subproject = Instance(Project)
    "Current sub-project instance"

    dataset = Instance(NDDataset)
    "Current NDDataset instance"

    script = Instance(Script)
    "Current Script instance"

    dlg_preference_pages = []

    # ------------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------------

    def __init__(self, **kwargs):

        super().__init__()

        # first thing to do : get a project (or the last project)
        autoload = app.general_preferences.autoload_project
        self.last = app.last_project

        if self.last and autoload:
            # load last
            self.last = app.last_project
        else:
            # create or load the default project
            self.last = 'DEFAULT'

        try:
            self.project = self.load_project(self.last)
        except:
            self.project = Project(name=self.last)
            self.project.save()

        app.last_project = self.last

        # window and other Qt settings
        # ----------------------------
        PX_FACTOR = QtWidgets.QApplication.instance().PX_FACTOR = \
            QtGui.QPaintDevice.logicalDpiY(
            self) / 96
        self.ww, self.wh = ww, wh = int(1440 * PX_FACTOR), int(900 * PX_FACTOR)
        self.resize(ww, wh)

        # Main area
        # ---------
        self.area = area = DockArea()
        self.setCentralWidget(area)

        self.setWindowIcon(QtGui.QIcon(geticon(app.icon)))
        # TODO: create an app to get the icon working:
        # see http://doc.qt.io/qt-5/appicon.html

        self.setWindowTitle(app.name)

        # Create status bar
        # -----------------
        self.statusbar = self.statusBar()
        self.statusbar.showMessage('Welcome to %s' % app.name)

        # Create progress bar
        # -------------------
        # Will be activated (and will appear in the status bar) for long
        # process
        self.progressbar = QtGui.QProgressBar()
        self.progressbar.setMaximumWidth(250)
        self.progressbar.setVisible(False)
        self.statusbar.addPermanentWidget(self.progressbar)

        # Create docked windows
        # ---------------------
        self._create_docks()

        # Create preferences pages
        # ------------------------
        # tuple (page, preferences section)
        self.dlg_preference_pages = [
            (GeneralPreferencePageWidget, 'general_preferences'),
            (ProjectPreferencePageWidget, 'project_preferences'), ]

        # Create menubar
        # --------------
        self._append_menubar()

        # eventually load a previous layout
        # ---------------------------------
        self.save_layout(default=True)

        self.load_layout()

    # ------------------------------------------------------------------------
    # Docks creation
    # ------------------------------------------------------------------------

    def _create_plot_area(self):

        dplots = Dock("plots", size=(self.ww * self._ratio,
                                                   self.wh * .80),
                                    closable=False, hideTitle=True)
        text = QtWidgets.QLabel("""
            <html>
            <center>
            <p>
            Select a dataset in the project tree, if any,<br/> 
            or add one to display it.</p>
            <p>
            Create a new project ... <font color='#00C'>%s</font> <br/>
            Open an existing project <font color='#00C'>%s</font> <br/>
            </p>
            </center>
            </html>
            """ % (QtGui.QKeySequence(QtGui.QKeySequence.New).toString(
            QtGui.QKeySequence.NativeText),
                   QtGui.QKeySequence(QtGui.QKeySequence.Open).toString(
                       QtGui.QKeySequence.NativeText)))
        dplots.addWidget(text)

        return dplots


    def _create_docks(self):

        ww, wh = self.ww, self.wh
        self._ratio = ratio = .72

        # plots
        # ------
        self.dplots = dplots = self._create_plot_area()

        # console
        # --------
        self.dconsole = dconsole = Dock("Console", size=(ww * ratio, wh * .20),
                                        closable=False)
        self.wconsole = ConsoleWidget()
        dconsole.addWidget(self.wconsole)
        dconsole.hideTitleBar()

        if app.general_preferences.log_level != DEBUG:
            # production

            # log to this console
            handler = QtHandler()
            log.handlers = [handler]  # addHandler(handler) #

            if True:  # TODO: obviously change this to some options
                redirectoutput(console=self.wconsole)

        # project window
        # ---------------

        dproject = Dock("Project", size=(ww * (1. - ratio), wh * 1.0),
                        closable=False)
        self.wproject = ProjectTreeWidget(project=self.project,
                                          showHeader=False)
        dproject.addWidget(self.wproject)

        self.wproject.itemClicked.connect(self.project_item_clicked)

        # FlowChart window
        # ----------------

        # self.dflowchart = dflowchart = Dock("FlowChart", size=(ww * .20,
        #                                                     wh * .50),
        #                   closable=False)
        # ## Create an empty flowchart with a single input and output
        # self.fc = fc = Flowchart(terminals={
        #     'dataIn': {'io': 'in'}, 'dataOut': {'io': 'out'}
        # })
        # w5 = fc.widget()
        # dflowchart.addWidget(w5)

        # set dock layout
        # ----------------

        self.area.addDock(dproject, 'left')
        self.area.addDock(dplots, 'right', dproject)
        self.area.addDock(dconsole, 'bottom', dplots)

        # self.area.addDock(dflowchart, 'bottom', dproject)

    # ------------------------------------------------------------------------
    # General setting for the window layout and geometry
    # ------------------------------------------------------------------------
    # We use the QT settings, for these very specific preferences, which are
    #  not used in the normal API

    def save_layout(self, default=False):
        settings = QtCore.QSettings()
        defstr = ''
        if default:
            defstr = 'def_'
        settings.setValue(defstr + "geometry", self.saveGeometry())
        settings.setValue(defstr + "areastate", self.area.saveState())
        del settings

    def load_layout(self, default=False):
        settings = QtCore.QSettings()
        defstr = ''
        if default:
            defstr = 'def_'

        self.restoreGeometry(settings.value(defstr + "geometry",
                                            settings.value("def_geometry")))
        self.ww, self.wh = self.frameGeometry().width(), self.frameGeometry(

        ).height()

        def checktab(item):
            # print(item)
            if hasattr(item, '__iter__') and not isinstance(item, str):
                for i in item:
                    if checktab(i):
                        return True
            else:
                if 'tab' in item:
                    return True
            return False

        try:
            s = settings.value(defstr + "areastate")
            if not checktab(s['main']):
                # if tab in the state This main
                # that the application was not closed properly. In thsi
                # case we use the default.
                s = settings.value("def_areastate")
                self.area.restoreState(s)
        except:
            try:
                self.area.restoreState(settings.value("def_areastate"))
            except:
                log.error("can't restore docking state")

    # --------------------------------------------------------------------
    # Help and preferences actions
    # --------------------------------------------------------------------

    def about(self):
        """About the tool"""
        QtGui.QMessageBox.about(self, "SpectroChemPy {}".format(__release__),
                                long_description)

    def edit_preferences(self):

        if hasattr(self, 'dlg_preferences'):
            try:
                self.dlg_preferences.close()
            except RuntimeError:
                pass
        self.dlg_preferences = dlg = DialogPreferences(self)

        for Page, prefs in self.dlg_preference_pages:
            page = Page(dlg)
            preferences = getattr(app, prefs)
            page.initialize(preferences=preferences)
            dlg.add_page(page)

        dlg.set_current_index(1)
        ww = self.frameGeometry().width()
        wh = self.frameGeometry().height()
        dlg.resize(ww * .95, wh * .8)

        dlg.exec()

        if app.project_preferences.updated or \
            app.general_preferences.updated:
            app.project_preferences.updated = False
            app.general_preferences.updated = False
            opens = list(self.open_plots.keys())
            for key in opens:
                self.show_or_create_plot_dock(key, update=True)


    def reset_preferences(self):
        """
        Reset preferences to default values

        """

        if not warningMessage(self, message='Are you sure to reset to the '
                                            'default? All previous changes '
                                            'will '
                                            'be lost.'):
            return

        if hasattr(self, 'dlg_preferences'):
            try:
                self.dlg_preferences.reset = True
                self.dlg_preferences.close()
            except RuntimeError:
                pass

        log.debug("RESET")

        # we init all preference to their default
        app.reset_config = True
        app.init_all_preferences()
        pass

    # ------------------------------------------------------------------------
    # Project actions
    # ------------------------------------------------------------------------

    def load_project(self, fname, **kwargs):
        proj = Project.load(fname, **kwargs)
        proj.meta['project_file'] = fname
        return proj

    def open_project(self, *args, **kwargs):

        # if self.project.name != 'DEFAULT':
        self.close_project()

        if not kwargs.get('new', False):
            # get an existing project (limited to those present in the
            # project directory

            def getproject():
                directory = app.general_preferences.project_directory
                items = (f.split('.')[0].upper() for f in os.listdir(directory)
                         if f.endswith('.pscp'))

                item, ok = QtGui.QInputDialog.getItem(self, "select a project",
                                                      "list of available "
                                                      "projects", items, 0,
                                                      False)

                if ok and item:
                    return str(item)

            projectname = getproject()
            if projectname:
                proj = self.load_project(projectname, **kwargs)
            else:
                return

        else:
            # create a new project
            dlg = QtGui.QInputDialog(self)
            dlg.setWindowTitle('Create a new project')
            dlg.setInputMode(QtGui.QInputDialog.TextInput)
            dlg.setLabelText('Enter the project name:')
            dlg.resize(300, 120)
            ok = dlg.exec()
            projectname = dlg.textValue()

            if ok and projectname:
                proj = Project(name=str(projectname).upper())
            else:
                return

        if not kwargs.get('main', True):
            # get current project if we need to add a subproject
            self.subproject = proj
            self.project.add_project(proj)

        else:
            # if it is a main project, replace the previous displayed
            # project by this one
            self.project = proj

        app.last_project = self.project.name

        self.update_project_widget(self.project)

    def save_project(self, *args, **kwargs):
        main = kwargs.get('main', True)
        new = kwargs.get('new', False)
        if main:
            # save main project

            if not new:  # normal save
                self.project.save()

            else:  # save as
                dlg = QtGui.QInputDialog(self)
                dlg.setWindowTitle('Save as a new project')
                dlg.setInputMode(QtGui.QInputDialog.TextInput)
                dlg.setLabelText('Enter the new project name:')
                dlg.resize(300, 120)
                ok = dlg.exec()
                projectname = dlg.textValue()

                if ok and projectname:
                    proj = self.project.copy()
                    proj.name = str(projectname).upper()
                    proj.meta['project_file'] = proj.name
                    self.project = proj
                    self.project.save()
                    self.update_project_widget(proj)
                else:
                    return
        else:
            print()
            pass

    def close_project(self, *args, **kwargs):

        if app.general_preferences.show_close_dialog:
            b = _CloseProjectDialog(self, "Close current project ...")
            ret = b.exec_()
            if ret == QtWidgets.QMessageBox.Cancel:
                return
            if ret == QtWidgets.QMessageBox.Discard:
                return

        self.project.save()

    def remove_subproject(self):
        pass

    # ------------------------------------------------------------------------
    # Actions on project items
    # ------------------------------------------------------------------------

    # ........................................................................
    def update_project_widget(self, project):
        self.wproject.setproject(project)

    # ........................................................................
    def get_branches(self, item, branches=[]):
        # Check if top level item is selected or child selected
        if self.wproject.indexOfTopLevelItem(item) == -1:
            name = item.text(0)
            branches.insert(0, name)
            return self.get_branches(item.parent(), branches)
        else:
            return branches

    # ........................................................................
    def project_item_clicked(self):
        """
        When an item is clicked in the project window, some actions can be
        performed, e.g., plot the corresponding data.

        """
        sel = self.wproject.currentItem()
        if sel:
            branches = []
            branches = self.get_branches(sel, branches)
            if branches:
                try:
                    if sel.text(1) == "NDDataset":
                        # make a plot of the data
                        key = '.'.join(branches)
                        log.debug('plot %s' % key)
                        self.show_or_create_plot_dock(key)
                except Exception as e:
                    log.error(e)

    # ------------------------------------------------------------------------
    # Actions on matplotlib plot canvas
    # ------------------------------------------------------------------------

    def updatestatusbar(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.statusbar.showMessage(("x=%.4f y=%.4f" % (x, y)), 0)

    # ------------------------------------------------------------------------
    # close event
    # ------------------------------------------------------------------------

    def closeEvent(self, evt):

        self.save_layout()

        if app.general_preferences.show_close_dialog:
            b = _CloseProjectDialog(self, "Close Spectrochempy...")
            ret = b.exec_()
            if ret == QtWidgets.QMessageBox.Cancel:
                return evt.ignore()
            if ret == QtWidgets.QMessageBox.Discard:
                return evt.accept()

        # close open plots
        for key in list(self.open_plots.keys()):
            # get the existing ones
            dp, wplot, _ = self.open_plots[key]
            dp.close()
            del wplot

        self.project.save()

        return evt.accept()

    # ------------------------------------------------------------------------
    # Menubar
    # ------------------------------------------------------------------------

    def _append_menubar(self):

        # --------------------------------------------------------------------
        # MENU Project
        # --------------------------------------------------------------------

        project_menu = QtGui.QMenu('Project', parent=self)

        new_mp_action = QtGui.QAction('New project', self)
        new_mp_action.setShortcut(QtGui.QKeySequence.New)
        new_mp_action.setStatusTip('Create a new project')
        new_mp_action.triggered.connect(
            partial(self.open_project, main=True, new=True))
        project_menu.addAction(new_mp_action)

        open_mp_action = QtGui.QAction('Open project', self)
        open_mp_action.setShortcut(QtGui.QKeySequence.Open)
        open_mp_action.setStatusTip('Load an existing project')
        open_mp_action.triggered.connect(
            partial(self.open_project, main=True, new=False))
        project_menu.addAction(open_mp_action)

        save_mp_action = QtGui.QAction('Save project', self)
        save_mp_action.setStatusTip('Save the entire project')
        save_mp_action.setShortcut(QtGui.QKeySequence.Save)
        save_mp_action.triggered.connect(
            partial(self.save_project, main=True, new=False))
        project_menu.addAction(save_mp_action)

        save_mp_as_action = QtGui.QAction('Save project as ...', self)
        save_mp_as_action.setStatusTip('Save the entire project with a new '
                                       'name')
        save_mp_as_action.triggered.connect(
            partial(self.save_project, main=True, new=True))
        project_menu.addAction(save_mp_as_action)

        close_mp_action = QtGui.QAction('Close project', self)
        close_mp_action.setStatusTip(
            'Close the current project, subprojects and all opened datasets')
        close_mp_action.setShortcut(QtGui.QKeySequence.Close)
        close_mp_action.triggered.connect(
            partial(self.close_project, main=True))
        project_menu.addAction(close_mp_action)

        # ....................................................................
        project_menu.addSeparator()

        subproject_menu = QtGui.QMenu('Subprojects ...', parent=self)
        project_menu.addMenu(subproject_menu)

        new_sp_action = QtGui.QAction('Add new subproject', self)
        new_sp_action.setShortcut(
            QtGui.QKeySequence('Ctrl+Shift+N', QtGui.QKeySequence.NativeText))
        new_sp_action.setStatusTip('Create a new subproject and add it to the '
                                   'current project')
        new_sp_action.triggered.connect(
            partial(self.open_project, main=False, new=True))
        subproject_menu.addAction(new_sp_action)

        open_sp_action = QtGui.QAction('Add existing project...', self)
        open_sp_action.setShortcut(
            QtGui.QKeySequence('Ctrl+Shift+O', QtGui.QKeySequence.NativeText))
        open_sp_action.setStatusTip('Load a project and add it to the current'
                                    ' main project')
        open_sp_action.triggered.connect(
            partial(self.open_project, main=False, new=False))
        subproject_menu.addAction(open_sp_action)

        save_sp_action = QtGui.QAction('Save selected subproject', self)
        save_sp_action.setStatusTip(
            'Save the selected sub project into a file')
        save_sp_action.setShortcut(
            QtGui.QKeySequence('Ctrl+Shift+N', QtGui.QKeySequence.NativeText))
        save_sp_action.triggered.connect(
            partial(self.save_project, main=False, new=False))
        subproject_menu.addAction(save_sp_action)

        save_sp_as_action = QtGui.QAction('Save selected subproject as ...',
                                          self)
        save_sp_as_action.setStatusTip('Save only selected subproject to a '
                                       'file with a new name')
        save_sp_as_action.triggered.connect(
            partial(self.save_project, main=False, new=True))
        subproject_menu.addAction(save_sp_as_action)

        remove_sp_action = QtGui.QAction('Remove selected subproject', self)
        remove_sp_action.setStatusTip('Remove the selected subproject')
        # remove_sp_action.setShortcut(QtGui.QKeySequence.Close)
        remove_sp_action.triggered.connect(self.remove_subproject)
        subproject_menu.addAction(remove_sp_action)

        # ....................................................................
        project_menu.addSeparator()

        # Export figures

        export_project_menu = QtGui.QMenu('Export figures', parent=self)
        project_menu.addMenu(export_project_menu)

        export_mp_action = QtGui.QAction('All', self)
        export_mp_action.setStatusTip(
            'Pack all the data of the main project into one folder')
        # export_mp_action.triggered.connect(export_mp)
        export_mp_action.setShortcut(
            QtGui.QKeySequence('Ctrl+E', QtGui.QKeySequence.NativeText))
        export_project_menu.addAction(export_mp_action)

        export_sp_action = QtGui.QAction('Selected', self)
        export_sp_action.setStatusTip(
            'Pack all the data of the current sub project into one folder')
        export_sp_action.setShortcut(
            QtGui.QKeySequence('Ctrl+Shift+E', QtGui.QKeySequence.NativeText))
        # export_sp_action.triggered.connect(export_sp)
        export_project_menu.addAction(export_sp_action)

        project_menu.addSeparator()

        #  Quit

        if sys.platform != 'darwin':  # mac os makes this anyway
            quit_action = QtGui.QAction('Quit', self)
            quit_action.triggered.connect(
                QtCore.QCoreApplication.instance().quit)
            quit_action.setShortcut(QtGui.QKeySequence.Quit)
            project_menu.addAction(quit_action)

        self.menuBar().addMenu(project_menu)

        # --------------------------------------------------------------------
        # Console menu
        # --------------------------------------------------------------------

        console_menu = QtGui.QMenu('Console', self)
        console_menu.addActions(self.wconsole.actions())
        self.menuBar().addMenu(console_menu)

        # --------------------------------------------------------------------
        # Windows menu
        # --------------------------------------------------------------------

        windows_menu = QtGui.QMenu('View', self)
        self.menuBar().addMenu(windows_menu)
        windows_menu.addSeparator()
        window_layouts_menu = QtGui.QMenu('Window layouts', self)

        restore_layout_action = QtGui.QAction('Restore previous layout', self)
        restore_layout_action.triggered.connect(self.load_layout)
        window_layouts_menu.addAction(restore_layout_action)

        reset_layout_action = QtGui.QAction('Reset to default layout', self)
        reset_layout_action.triggered.connect(
            partial(self.load_layout, default=True))
        window_layouts_menu.addAction(reset_layout_action)

        save_layout_action = QtGui.QAction('Save layout', self)
        save_layout_action.triggered.connect(self.save_layout)
        window_layouts_menu.addAction(save_layout_action)

        windows_menu.addMenu(window_layouts_menu)

        # --------------------------------------------------------------------
        # Help menu
        # --------------------------------------------------------------------

        help_menu = QtGui.QMenu('Help', parent=self)
        self.menuBar().addMenu(help_menu)

        # --------------------------------------------------------------------
        # Preferences
        # --------------------------------------------------------------------

        help_action = QtGui.QAction('Preferences', self)
        help_action.triggered.connect(lambda: self.edit_preferences())
        help_action.setShortcut(QtGui.QKeySequence.Preferences)
        help_menu.addAction(help_action)

        # --------------------------------------------------------------------
        # About
        # --------------------------------------------------------------------

        about_action = QtGui.QAction('About', self)
        about_action.triggered.connect(self.about)
        help_menu.addAction(about_action)

        self.menuBar().setNativeMenuBar(
            False)  #  this put the menu in the window itself in OSX,
        # as in windows.  # when running application from pycharm, it helps
        # to have immediate  # access to menu  # Indeed, on mac they are not
        #  accessible until unfocused the window


# ============================================================================
class _CloseProjectDialog(QtWidgets.QMessageBox):

    def __init__(self, mainWindow, message):
        QtWidgets.QMessageBox.__init__(self, mainWindow)
        self.setIcon(QtWidgets.QMessageBox.Warning)
        self.setText(message)
        self.setInformativeText("Save current project?")
        self.setStandardButtons(
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard |
            QtWidgets.QMessageBox.Cancel)

        c = QtWidgets.QCheckBox("don't ask me again")
        c.clicked.connect(self.change_ask_again)

        self.layout().addWidget(c, 4, 0, 7, 0)

    def change_ask_again(self, val):
        app.general_preferences.show_close_dialog = not val


# ============================================================================
if __name__ == '__main__':

    pass
