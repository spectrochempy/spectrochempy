# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================


import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ...extern.pyqtgraph.Qt import QtGui, QtCore

from spectrochempy.application import app

__all__ = ['MatplotlibWidget']


class MatplotlibWidget(QtGui.QWidget):
    """
    Implements a Matplotlib figure inside a QWidget.
    Use getFigure() and redraw() to interact with matplotlib.
    
    Example::
    
        mw = MatplotlibWidget()
        subplot = mw.getFigure().add_subplot(111)
        subplot.plot(x,y)
        mw.draw()
    """
    
    def __init__(self, parent=None,
                 background='default',
                 dpi=100,
                 **kargs):

        if background=='default':
            background = app.project_preferences.background_color

        QtGui.QWidget.__init__(self)
        #a=236./255.

        self.fig = Figure(dpi=dpi, facecolor=background) #(a,a,a))
        self.setSizePolicy(QtGui.QSizePolicy.Expanding,
                           QtGui.QSizePolicy.Expanding)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addWidget(self.toolbar)
        self.vbox.addWidget(self.canvas)
        
        self.setLayout(self.vbox)

    def getFigure(self):
        return self.fig
        
    def draw(self):
        self.canvas.draw()

    def clear(self):
        self.fig.clf()

    def plot(self, *args, **kargs):
        fig = self.fig
        clear = kargs.get('clear', False)
        params = kargs.get('params', None)

        if clear:
            self.clear()

        if len(args) == 1:
            data = args[0]
            data.plot(ax=fig.gca())
        else:
            raise NotImplementedError
            # TODO: make this class as close as possible to that thovided by pyqtgraph

        self.draw()
