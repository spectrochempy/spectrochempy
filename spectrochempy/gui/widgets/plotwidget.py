# -*- coding: utf-8 -*-
"""
PlotWidget.py -  Convenience class--GraphicsView widget displaying a single PlotItem
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infomation.
"""
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ...extern.pyqtgraph.Qt import QtGui, QtCore
from ...extern import pyqtgraph as pg
from ...extern.pyqtgraph.WidgetGroup import WidgetGroup
from ...extern.pyqtgraph.widgets.GraphicsView import GraphicsView

from ..pyqtgraph.graphicsItems.PlotItems import PlotItem
from ..pyqtgraph.graphicsItems.PlotDataItem import PlotDataItem
from ..pyqtgraph.graphicsItems.PlotItems.plotConfigTemplate_pyqt5 import Ui_Form

from spectrochempy.application import app,log

__all__ = ['PyQtGraphWidget', 'MatplotlibWidget']


class PyQtGraphWidget(GraphicsView):
    
    # signals wrapped from PlotItem / ViewBox
    sigRangeChanged = QtCore.Signal(object, object)
    sigTransformChanged = QtCore.Signal(object)
    
    """
    :class:`GraphicsView <pyqtgraph.GraphicsView>` widget with a single 
    :class:`PlotItem <pyqtgraph.PlotItem>` inside.
    
    The following methods are wrapped directly from PlotItem: 
    :func:`addItem <pyqtgraph.PlotItem.addItem>`, 
    :func:`removeItem <pyqtgraph.PlotItem.removeItem>`, 
    :func:`clear <pyqtgraph.PlotItem.clear>`, 
    :func:`setXRange <pyqtgraph.ViewBox.setXRange>`,
    :func:`setYRange <pyqtgraph.ViewBox.setYRange>`,
    :func:`setRange <pyqtgraph.ViewBox.setRange>`,
    :func:`autoRange <pyqtgraph.ViewBox.autoRange>`,
    :func:`setXLink <pyqtgraph.ViewBox.setXLink>`,
    :func:`setYLink <pyqtgraph.ViewBox.setYLink>`,
    :func:`viewRect <pyqtgraph.ViewBox.viewRect>`,
    :func:`setMouseEnabled <pyqtgraph.ViewBox.setMouseEnabled>`,
    :func:`enableAutoRange <pyqtgraph.ViewBox.enableAutoRange>`,
    :func:`disableAutoRange <pyqtgraph.ViewBox.disableAutoRange>`,
    :func:`setAspectLocked <pyqtgraph.ViewBox.setAspectLocked>`,
    :func:`setLimits <pyqtgraph.ViewBox.setLimits>`,
    :func:`register <pyqtgraph.ViewBox.register>`,
    :func:`unregister <pyqtgraph.ViewBox.unregister>`
    
    
    For all 
    other methods, use :func:`getPlotItem <pyqtgraph.PlotWidget.getPlotItem>`.
    """

    def implements(self, name=None):
        if name is None:
            return ['PyQtGraphWidget']
        else:
            return name == 'PyQtGraphWidget'

    def __init__(self,
                 data=None,
                 parent=None,
                 background='default',
                 **kargs):
        """When initializing PlotWidget, *parent* and *background* are passed to 
        :func:`GraphicsWidget.__init__() <pyqtgraph.GraphicsWidget.__init__>`
        and all others are passed
        to :func:`PlotItem.__init__() <pyqtgraph.PlotItem.__init__>`."""

        # put some pyqtgraph configuration parameters from general preferences
        # --------------------------------------------------------------------
        pg.setConfigOption('background',
                           app.general_preferences.background_color)
        pg.setConfigOption('foreground',
                           app.general_preferences.foreground_color)
        pg.setConfigOption('antialias', app.general_preferences.antialias)
        pg.setConfigOption('leftButtonPan',
                           app.general_preferences.leftbuttonpan)

        # initialize the widget
        # ---------------------
        GraphicsView.__init__(self, parent, background=background)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding,
                           QtGui.QSizePolicy.Expanding)
        self.enableMouse(False)
        self.plotItem = PlotItem(**kargs)
        self.setCentralItem(self.plotItem)
        ## Explicitly wrap methods from plotItem
        ## NOTE: If you change this list, update the documentation above as well.
        for m in ['addItem', 'removeItem', 'autoRange', 'clear', 'setXRange', 
                  'setYRange', 'setRange', 'setAspectLocked', 'setMouseEnabled', 
                  'setXLink', 'setYLink', 'enableAutoRange', 'disableAutoRange', 
                  'setLimits', 'register', 'unregister', 'viewRect']:
            setattr(self, m, getattr(self.plotItem, m))
        #QtCore.QObject.connect(self.plotItem, QtCore.SIGNAL('viewChanged'), self.viewChanged)
        self.plotItem.sigRangeChanged.connect(self.viewRangeChanged)

        # if we have passed data, we plot it immediately
        # -----------------------------------------------
        if data is not None:
            data.plot(widget = self,
                      clear = True)

    def close(self):
        self.plotItem.close()
        self.plotItem = None
        #self.scene().clear()
        #self.mPlotItem.close()
        self.setParent(None)
        super().close()

    def __getattr__(self, attr):  ## implicitly wrap methods from plotItem
        if hasattr(self.plotItem, attr):
            m = getattr(self.plotItem, attr)
            if hasattr(m, '__call__'):
                return m
        raise NameError(attr)
    
    def viewRangeChanged(self, view, range):
        #self.emit(QtCore.SIGNAL('viewChanged'), *args)
        self.sigRangeChanged.emit(self, range)

    def widgetGroupInterface(self):
        return (None, PyQtGraphWidget.saveState, PyQtGraphWidget.restoreState)

    def saveState(self):
        return self.plotItem.saveState()
        
    def restoreState(self, state):
        return self.plotItem.restoreState(state)
        
    def getPlotItem(self):
        """Return the PlotItem contained within."""
        return self.plotItem

    def plot(self, *args, **kargs):
        """
        Add and return a new plot.
        See :func:`PlotDataItem.__init__ <pyqtgraph.PlotDataItem.__init__>`
        for data arguments

        Extra allowed arguments are:
            clear    - clear all plots before displaying new data
            params   - meta-parameters to associate with this data

        """

        clear = kargs.get('clear', False)
        params = kargs.get('params', None)

        if clear:
            self.clear()

        item = PlotDataItem(*args, **kargs)

        if params is None:
            params = {}

        self.addItem(item, params=params)

        return item


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

    def implements(self, name=None):
        if name is None:
            return ['MatplotlibWidget']
        else:
            return name == 'MatplotlibWidget'

    def __init__(self,
                 data=None,
                 parent=None,
                 background='default',
                 dpi=100,
                 **kargs):

        if background =='default':
            background = app.general_preferences.background_color

        QtGui.QWidget.__init__(self)

        self.setSizePolicy(QtGui.QSizePolicy.Expanding,
                           QtGui.QSizePolicy.Expanding)

        self.fig = Figure(dpi=dpi, facecolor=background)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addWidget(self.toolbar)
        self.vbox.addWidget(self.canvas)

        self.setLayout(self.vbox)

        if data is not None:
            data.plot(widget = self,
                      clear = True)

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

    def __getattr__(self, item):

        log.debug('%s not implemented ye t '%item)
        return self._tempo

    def _tempo(self, *args, **kwargs):
        pass