# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""
This module is the main entry-point for the GUI application launched from the
terminal command line

"""

#For compatibility with the pyQtgtaph library we copy the Qt.py part
# corresponding to PyQt5. indeed, we cann use as later
# import spectrochempy.extern.pyqtgraph.qt because we want a Splash screen
# before loading the spectrochempy API
# We're using PyQt5 which has a different structure so we're going to use a shim to
# recreate the Qt4 structure for Qt5

from PyQt5 import QtGui, QtCore, QtWidgets

try:
    from PyQt5 import QtSvg
except ImportError:
    pass
try:
    from PyQt5 import QtOpenGL
except ImportError:
    pass
try:
    from PyQt5 import QtTest

    QtTest.QTest.qWaitForWindowShown = QtTest.QTest.qWaitForWindowExposed
except ImportError:
    pass

# Re-implement deprecated APIs

__QGraphicsItem_scale = QtWidgets.QGraphicsItem.scale


def scale(self, *args):
    if args:
        sx, sy = args
        tr = self.transform()
        tr.scale(sx, sy)
        self.setTransform(tr)
    else:
        return __QGraphicsItem_scale(self)


QtWidgets.QGraphicsItem.scale = scale


def rotate(self, angle):
    tr = self.transform()
    tr.rotate(angle)
    self.setTransform(tr)


QtWidgets.QGraphicsItem.rotate = rotate


def translate(self, dx, dy):
    tr = self.transform()
    tr.translate(dx, dy)
    self.setTransform(tr)


QtWidgets.QGraphicsItem.translate = translate


def setMargin(self, i):
    self.setContentsMargins(i, i, i, i)


QtWidgets.QGridLayout.setMargin = setMargin


def setResizeMode(self, *args):
    self.setSectionResizeMode(*args)


QtWidgets.QHeaderView.setResizeMode = setResizeMode

QtGui.QApplication = QtWidgets.QApplication
QtGui.QGraphicsScene = QtWidgets.QGraphicsScene
QtGui.QGraphicsObject = QtWidgets.QGraphicsObject
QtGui.QGraphicsWidget = QtWidgets.QGraphicsWidget

QtGui.QApplication.setGraphicsSystem = None

# Import all QtWidgets objects into QtGui
for o in dir(QtWidgets):
    if o.startswith('Q'):
        setattr(QtGui, o, getattr(QtWidgets, o))

VERSION_INFO = 'PyQt5 ' + QtCore.PYQT_VERSION_STR + ' Qt ' + QtCore.QT_VERSION_STR

import sip


def isQObjectAlive(obj):
    return not sip.isdeleted(obj)

QtCore.Signal = QtCore.pyqtSignal

import sys, time

def main_gui():

    # if 'darwin' in sys.platform:
    #   if QtGui.QApplication.setGraphicsSystem:
    #      QtGui.QApplication.setGraphicsSystem('raster')  # to avoid the warning  (but on sierra it doesnt
    gui = QtGui.QApplication(sys.argv)


    # Create and display the splash screen
    splash_pix = QtGui.QPixmap('gui/ressources/scpy_splash.png')

    splash = QtGui.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
    splash.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint |
                          QtCore.Qt.FramelessWindowHint)
    splash.setEnabled(False)

    # adding progress bar
    progressBar = QtGui.QProgressBar(splash)
    progressBar.setMaximum(10)
    progressBar.setGeometry(0, splash_pix.height() - 50, splash_pix.width(),
                            20)
    progressBar.setVisible(False)

    splash.show()
    splash.showMessage("<br/><h1><font color='blue'>"
                       "Welcome to<br/>SpectroChemPy</font></h1>",
                       QtCore.Qt.AlignTop | QtCore.Qt.AlignCenter,
                       QtCore.Qt.black)
    gui.processEvents()
    time.sleep(1)

    splash.clearMessage()

    splash.showMessage("<br/><h3><font color='blue'>"
                       "Loading API...</font></h3>",
                       QtCore.Qt.AlignTop | QtCore.Qt.AlignCenter,
                       QtCore.Qt.black)
    gui.processEvents()

    progressBar.setVisible(True)
    
    for i in range(1, 11):
        progressBar.setValue(i)
        t = time.time()
        while time.time() < t + 0.1:
            gui.processEvents()


    from spectrochempy.gui.gui import MainWindow

    w = MainWindow()
    w.show()
    time.sleep(.2)
    splash.finish(w)

    sys.exit(gui.exec())


main_gui()