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
import logging
from functools import partial

# ----------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------

from traitlets import HasTraits

# ----------------------------------------------------------------------------
# local imports
# ----------------------------------------------------------------------------

from ..extern import pyqtgraph as pg
from ..extern.pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from ..extern.pyqtgraph.dockarea import DockArea, Dock
from ..extern.pyqtgraph.flowchart import Flowchart, Node
from ..extern.pyqtgraph.flowchart import library as fclib
from ..extern.pyqtgraph.flowchart.library.common import CtrlNode
from ..extern.pyqtgraph import console

from .widgets.projecttreewidget import ProjectTreeWidget
from .widgets.matplotlibwidget import MatplotlibWidget
from .logtoconsole import QtHandler, redirectoutput
from .plots import Plots
from .preferences import (Preferences, ProjectOptionsWidget,
                        GeneralOptionsWidget, PlotOptionsWidget)
from .guiutils import geticon

from ..application import app
from ..projects.project import Project

log = app.log
general_options = app.general_options
project_options = app.project_options

# set flags to change for the final usage
__DEV__ = True

# =============================================================================
class MainWindow(QtGui.QMainWindow, Plots):

    #: current project instance
    project = None

    #: current dataset instance
    dataset = None

    preference_pages = []

    # ........................................................................
    def __init__(self, show=True):

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        super(MainWindow, self).__init__()

        # Main area
        # ---------
        self.area = area = DockArea()
        self.setCentralWidget(area)

        self.setWindowIcon(QtGui.QIcon(geticon('scp.png')))

        self.setWindowTitle('SpectroChemPy')

        # Create progress bar
        # -------------------
        self.progressbar = QtGui.QProgressBar()
        self.progressbar.setMaximumWidth(250)
        self.progressbar.setVisible(False)

        # Create status bar
        # -----------------
        self.statusbar = self.statusBar()
        self.statusbar.showMessage('Welcome to SpectroChemPy')
        self.statusbar.addPermanentWidget(self.progressbar)

        # Create docked windows
        # ---------------------
        self._create_docks()

        # Create Menubar and preferences
        # ------------------------------
        self._append_menubar_and_preferences()

        self.preference_pages.extend([GeneralOptionsWidget,
                                      ProjectOptionsWidget,
                                      PlotOptionsWidget])

        # show window
        # --------------------------------------------------------------------
        if show:
            self.show()  # Maximized()

    # ........................................................................
    def _create_docks(self):

        # rescale relative
        PX_FACTOR = QtWidgets.QApplication.instance().PX_FACTOR = \
            QtGui.QPaintDevice.logicalDpiY(
            self) / 96

        ww, wh = 1440 * PX_FACTOR, 900 * PX_FACTOR

        # --------------------------------------------------------------------
        # Console
        # --------------------------------------------------------------------
        self.dplots = dplots = Dock("plots", size=(ww * .80, wh * .80))
        dplots.hideTitleBar()

        self.dconsole = dconsole = Dock("Console", size=(ww * .80, wh * .20),
                                        closable=False)
        self.wconsole = pg.console.ConsoleWidget()
        dconsole.addWidget(self.wconsole)
        dconsole.hideTitleBar()

        if not __DEV__:
            # production
            # log to this console
            handler = QtHandler()
            log.handlers = [handler]  # addHandler(handler) #

            if True:  # TODO: obviously change this to some options
                redirectoutput(console=self.wconsole)

            general_options.log_level = logging.WARNING
        else:
            # developpement
            general_options.log_level = logging.DEBUG

        # --------------------------------------------------------------------
        # project window
        # --------------------------------------------------------------------

        dproject = Dock("Project", size=(ww * .20, wh * .50), closable=False)
        d = None
        startup_project = general_options.startup_project
        if startup_project:
            d = self.load_project(startup_project)
        self.wproject = ProjectTreeWidget(project=d, showHeader=False)
        dproject.addWidget(self.wproject)

        self.wproject.itemClicked.connect(self.project_item_clicked)
        #self.wproject.itemEntered.connect(self.project_item_activated)
        # self.wproject.itemSelectionChanged.connect(
        # self.project_item_activated)

        # currentItemChanged(::QTreeWidgetItem *,::QTreeWidgetItem *)
        # itemActivated(::QTreeWidgetItem *, int)
        # itemChanged(::QTreeWidgetItem *, int)
        # itemClicked(::QTreeWidgetItem *, int)
        # itemCollapsed(::QTreeWidgetItem *)
        # itemDoubleClicked(::QTreeWidgetItem *, int)
        # itemEntered(::QTreeWidgetItem *, int)
        # itemExpanded(::QTreeWidgetItem *)
        # itemPressed(::QTreeWidgetItem *, int)
        # itemSelectionChanged()

        # --------------------------------------------------------------------
        # FlowChart window
        # --------------------------------------------------------------------

        self.dflowchart = dflowchart = Dock("FlowChart", size=(ww * .20,
                                                            wh * .50),
                          closable=False)
        ## Create an empty flowchart with a single input and output
        self.fc = fc = Flowchart(terminals={
            'dataIn': {'io': 'in'}, 'dataOut': {'io': 'out'}
        })
        w5 = fc.widget()
        dflowchart.addWidget(w5)


        # --------------------------------------------------------------------
        # set layout
        # --------------------------------------------------------------------

        self.area.addDock(dproject, 'left')
        self.area.addDock(dplots, 'right')
        self.area.addDock(dflowchart, 'bottom', dproject)
        self.area.addDock(dconsole, 'bottom', dplots)

        self.save_layout()
        self.resize(ww, wh)

    @property
    def project_dir(self):
        return project_options.projects_directory

    def load_project(self, fname, **kwargs):
        proj = Project.load(fname, **kwargs)
        proj.meta['project_file'] = fname
        return proj

    def save_layout(self):
        global layout
        layout = self.area.saveState()

    def load_layout(self):
        global layout
        self.area.restoreState(layout)

    # ........................................................................
    def _append_menubar_and_preferences(self):

        def open_mp(*args, **kwargs):
            open_project(main=True)

        def open_sp(self, *args, **kwargs):
            self._open_project(main=False)

        def open_project(self, *args, **kwargs):
            fname = QtGui.QFileDialog.getOpenFileName(self, 'Project file',
                self.project_dir, 'SpectroChemPy project files (*.pscp);;'
                                  'All files (*)')
            fname = fname[0]
            if not fname:
                return
            proj = self.load_project(fname, **kwargs)
            self.update_project_widget(proj)

        def save_mp(*args, **kwargs):
            self._save_project(main=True)

        def save_sp(*args, **kwargs):
            self.save_project(main=False)

        def export_mp(*args, **kwargs):
            export_project(main=True)

        def export_sp(self, *args, **kwargs):
            export_project(main=False)

        def about():
            """About the tool"""
            # versions = {
            #     key: d['version'] for key, d in psyplot.get_versions(
            # False).items()
            #     }
            # versions.update(psyplot_gui.get_versions()['requirements'])
            # versions.update(psyplot._get_versions()['requirements'])
            # versions['github'] = 'https://github.com/Chilipp/psyplot'
            # versions['author'] = psyplot.__author__
            about = QtGui.QMessageBox.about(self, "About SpectroChemPy",
                                            """ <strong> 
            the about info </strong>
                """)
            # % versions)

        def edit_preferences(exec_=None):

            if hasattr(self, 'preferences'):
                try:
                    self.preferences.close()
                except RuntimeError:
                    pass
            self.preferences = dlg = Preferences(self)

            for Page in self.preference_pages:
                page = Page(dlg)
                page.initialize()
                dlg.add_page(page)

            available_width = 0.667 * \
                             QtGui.QDesktopWidget().availableGeometry().width()
            width = dlg.sizeHint().width()
            height = dlg.sizeHint().height()
            # The preferences window should cover at least one third of the screen
            dlg.resize(max(available_width, width), height)
            if exec_:
                dlg.exec_()

        # MENU FILE
        # -------------------------------------------------------------------
        file_menu = QtGui.QMenu('File', parent=self)

        # Open project
        # --------------------------------------------------------------------
        open_project_menu = QtGui.QMenu('Open project', self)
        file_menu.addMenu(open_project_menu)

        open_mp_action = QtGui.QAction('New main project', self)
        open_mp_action.setShortcut(QtGui.QKeySequence.Open)
        open_mp_action.setStatusTip('Open a new main project')
        open_mp_action.triggered.connect(open_mp)
        open_project_menu.addAction(open_mp_action)

        open_sp_action = QtGui.QAction('Add to current', self)
        open_sp_action.setShortcut(
            QtGui.QKeySequence('Ctrl+Shift+O', QtGui.QKeySequence.NativeText))
        open_sp_action.setStatusTip('Load a project as a sub project '
                                         'and add it to the current main '
                                         'project')
        open_sp_action.triggered.connect(open_sp)
        open_project_menu.addAction(open_sp_action)

        self.menuBar().addMenu(file_menu)

        # Save project
        # --------------------------------------------------------------------

        save_project_menu = QtGui.QMenu('Save project', parent=self)
        file_menu.addMenu(save_project_menu)

        save_mp_action = QtGui.QAction('All', self)
        save_mp_action.setStatusTip('Save the entire project into a file')
        save_mp_action.setShortcut(QtGui.QKeySequence.Save)
        save_mp_action.triggered.connect(save_mp)
        save_project_menu.addAction(save_mp_action)

        save_sp_action = QtGui.QAction('Selected', self)
        save_sp_action.setStatusTip(
            'Save the selected sub project into file')
        save_sp_action.triggered.connect(save_sp)
        save_project_menu.addAction(save_sp_action)

        # Save project as
        # --------------------------------------------------------------------

        save_project_as_menu = QtGui.QMenu('Save project as', parent=self)
        file_menu.addMenu(save_project_as_menu)

        save_mp_as_action = QtGui.QAction('All', self)
        save_mp_as_action.setStatusTip(
            'Save the entire project into a file')
        save_mp_as_action.setShortcut(QtGui.QKeySequence.SaveAs)
        save_mp_as_action.triggered.connect(
            partial(save_mp, new_fname=True))
        save_project_as_menu.addAction(save_mp_as_action)

        save_sp_as_action = QtGui.QAction('Selected', self)
        save_sp_as_action.setStatusTip(
            'Save the selected sub project into a file')
        save_sp_as_action.triggered.connect(
            partial(save_sp, new_fname=True))
        save_project_as_menu.addAction(save_sp_as_action)

        # Export figures
        # --------------------------------------------------------------------

        export_project_menu = QtGui.QMenu('Export figures', parent=self)
        file_menu.addMenu(export_project_menu)

        export_mp_action = QtGui.QAction('All', self)
        export_mp_action.setStatusTip(
            'Pack all the data of the main project into one folder')
        export_mp_action.triggered.connect(export_mp)
        export_mp_action.setShortcut(
            QtGui.QKeySequence('Ctrl+E', QtGui.QKeySequence.NativeText))
        export_project_menu.addAction(export_mp_action)

        export_sp_action = QtGui.QAction('Selected', self)
        export_sp_action.setStatusTip(
            'Pack all the data of the current sub project into one folder')
        export_sp_action.setShortcut(
            QtGui.QKeySequence('Ctrl+Shift+E', QtGui.QKeySequence.NativeText))
        export_sp_action.triggered.connect(export_sp)
        export_project_menu.addAction(export_sp_action)

        # Close project
        # --------------------------------------------------------------------

        file_menu.addSeparator()

        close_project_menu = QtGui.QMenu('Close project', parent=self)
        file_menu.addMenu(close_project_menu)

        close_mp_action = QtGui.QAction('Main project', self)
        close_mp_action.setShortcut(
            QtGui.QKeySequence('Ctrl+Shift+W', QtGui.QKeySequence.NativeText))
        close_mp_action.setStatusTip(
            'Close the main project and delete all data and plots out of '
            'memory')
        # TODO : self.close_mp_action.triggered.connect(
        #        lambda : psy.close(psy.gcp(True).num))
        close_project_menu.addAction(close_mp_action)

        close_sp_action = QtGui.QAction('Only selected', self)
        close_sp_action.setStatusTip(
            'Close the selected arrays project and delete all data and plots '
            'out of memory')
        close_sp_action.setShortcut(QtGui.QKeySequence.Close)
        # TODO: close_sp_action.triggered.connect(
        #        lambda : psy.gcp().close(True, True))
        close_project_menu.addAction(close_sp_action)

        #  Quit
        # --------------------------------------------------------------------

        if sys.platform != 'darwin':  # mac os makes this anyway
            quit_action = QtGui.QAction('Quit', self)
            quit_action.triggered.connect(
                QtCore.QCoreApplication.instance().quit)
            quit_action.setShortcut(QtGui.QKeySequence.Quit)
            file_menu.addAction(quit_action)

        self.menuBar().addMenu(file_menu)

        # ######################## Console menu ###############################

        console_menu = QtGui.QMenu('Console', self)
        console_menu.addActions(self.wconsole.actions())
        self.menuBar().addMenu(console_menu)

        # ######################## Windows menu ###############################

        windows_menu = QtGui.QMenu('View', self)
        self.menuBar().addMenu(windows_menu)
        windows_menu.addSeparator()
        window_layouts_menu = QtGui.QMenu('Window layouts', self)

        restore_layout_action = QtGui.QAction('Restore previous layout',
                                                   self)
        restore_layout_action.triggered.connect(self.load_layout)
        window_layouts_menu.addAction(restore_layout_action)

        save_layout_action = QtGui.QAction('Save layout', self)
        save_layout_action.triggered.connect(self.save_layout)
        window_layouts_menu.addAction(save_layout_action)

        windows_menu.addMenu(window_layouts_menu)

        # ############################ Help menu ##############################

        help_menu = QtGui.QMenu('Help', parent=self)
        self.menuBar().addMenu(help_menu)

        # -------------------------- Preferences ------------------------------

        help_action = QtGui.QAction('Preferences', self)
        help_action.triggered.connect(lambda: edit_preferences(True))
        help_action.setShortcut(QtGui.QKeySequence.Preferences)
        help_menu.addAction(help_action)

        # ---------------------------- About ----------------------------------

        about_action = QtGui.QAction('About', self)
        about_action.triggered.connect(about)
        help_menu.addAction(about_action)

        # self.menuBar().setNativeMenuBar(False)  # this put the menu in the
        #  window itself in OSX, as in windows.


    # ------------------------------------------------------------------------
    # Actions on matplotlib plot canvas
    # ------------------------------------------------------------------------

    def updatestatusbar(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.statusbar.showMessage(("x=%.4f y=%.4f" % (x, y)), 0)

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
                    if sel.text(
                            1) == "NDDataset":  # sinstance(data, NDDataset):
                        # make a plot of the data
                        key = '.'.join(branches)
                        log.debug('add %s' % key)
                        self.show_or_create_plot_dock(key)
                except Exception as e:
                    log.error(e)

    # ------------------------------------------------------------------------
    # Starts and run methods
    # ------------------------------------------------------------------------

    @classmethod
    def run(cls, *args, **kwargs):
        """
        Create a mainwindow and open the given files or project

        This class method creates a new mainwindow instance and sets the
        global :attr:`mainwindow` variable.

        Parameters
        ----------
        %(MainWindow.open_external_files.parameters)s
        %(MainWindow.parameters)s

        Notes
        -----
        - There can be only one mainwindow at the time
        - This method does not create a QApplication instance! See
          :meth:`run_app`

        See Also
        --------
        run_app
        """

        show = kwargs.get('show', True)
        mainwindow = cls(show=show)

        # here we can process the command line parameters
        # TODO

        return mainwindow

    @classmethod
    def start(cls, *args, **kwargs):
        """
        Create a QApplication, open the given files or project and enter the
        mainloop

        Parameters
        ----------
        %(MainWindow.run.parameters)s

        See Also
        --------
        run
        """

        gui = QtGui.QApplication(sys.argv)
        cls.run(*args, **kwargs)

        sys.exit(gui.exec_())


# =============================================================================
if __name__ == '__main__':

    MainWindow.start()
