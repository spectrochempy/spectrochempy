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
from matplotlib.rcsetup import cycler
import matplotlib as mpl

from ...extern.pyqtgraph.Qt import QtGui, QtCore
from ...extern import pyqtgraph as pg
from ...extern.pyqtgraph.WidgetGroup import WidgetGroup
from ...extern.pyqtgraph.widgets.GraphicsView import GraphicsView

from ..pyqtgraph.graphicsItems.PlotItems import PlotItem
from ..pyqtgraph.graphicsItems.PlotDataItem import PlotDataItem
from ..pyqtgraph.graphicsItems.PlotItems.plotConfigTemplate_pyqt5 import Ui_Form

from ...application import app,log
from ...utils import getdocfrom

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

    _prop_cycle = None

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

        clear = kargs.get('clear', True)
        params = kargs.get('params', None)

        if clear:
            self.clear()

        # make a correspndence between some matplotlib parameters and pyqtgraph
        kargs['symbol']= kargs.get('marker', None)
        kargs['symbolSize'] = kargs.get('markersize', self.prefs.markersize)
        kargs['symbolPen'] = kargs.get('mfc', "#000000")
        kargs['symbolBrush'] = kargs.get('mec', self.prefs.pen_color)

        item = PlotDataItem(*args, **kargs)

        if params is None:
            params = {}

        self.addItem(item, params=params)

        return (item,)

    ##### addition to perform similar command used in matplotlib #####
    ##### tentative emulation of the plt.Axes object

    @property
    def lines(self):
        return self.plotItem.items

    @getdocfrom(mpl.axes.Axes.set_prop_cycle)
    def set_prop_cycle(self, *args, **kwargs):

        if args and kwargs:
            raise TypeError("Cannot supply both positional and keyword "
                            "arguments to this method.")
        if len(args) == 1 and args[0] is None:
            self._prop_cycle = None
        else:
            self._prop_cycle = cycler(*args, **kwargs)

    @getdocfrom(mpl.axes.Axes.set_xscale)
    def set_xscale(self, xscale, **kwargs):
        xlog = (xscale == 'log')
        self.plotItem.axes['top']['item'].setLogMode(xlog)
        self.plotItem.axes['bottom']['item'].setLogMode(xlog)

    @getdocfrom(mpl.axes.Axes.set_yscale)
    def set_yscale(self, yscale, **kwargs):
        ylog = (yscale == 'log')
        self.plotItem.axes['left']['item'].setLogMode(ylog)
        self.plotItem.axes['right']['item'].setLogMode(ylog)

    @getdocfrom(mpl.axes.Axes.set_xlim)
    def set_xlim(self, *xlim):
        if len(xlim)==1:
            xlim = xlim[0]
        mi, ma = xlim
        if mi > ma:
            self.plotItem.invertX(True)
            mi, ma = ma, mi
        self.plotItem.setXRange(mi, ma)
        return xlim

    @getdocfrom(mpl.axes.Axes.set_ylim)
    def set_ylim(self, *ylim):
        if len(ylim)==1:
            ylim = ylim[0]
        mi, ma = ylim
        if mi > ma:
            self.plotItem.invertY(True)
            mi, ma = ma, mi
        self.plotItem.setYRange(mi, ma)
        return ylim

    def set_xlabel(self, label):
        self.setLabel('bottom', label)  # TODO:mathTex_to_QPixmap(label))

    def set_ylabel(self, label):
        self.setLabel('left', label)


def mathTex_to_QPixmap(mathTex, fs=10):

    #---- set up a mpl figure instance ----

    fig = mpl.figure.Figure()
    fig.patch.set_facecolor('none')
    fig.set_canvas(FigureCanvas(fig))
    renderer = fig.canvas.get_renderer()

    #---- plot the mathTex expression ----

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.patch.set_facecolor('none')
    t = ax.text(0, 0, mathTex, ha='left', va='bottom', fontsize=fs)

    #---- fit figure size to text artist ----

    fwidth, fheight = fig.get_size_inches()
    fig_bbox = fig.get_window_extent(renderer)

    text_bbox = t.get_window_extent(renderer)

    tight_fwidth = text_bbox.width * fwidth / fig_bbox.width
    tight_fheight = text_bbox.height * fheight / fig_bbox.height

    fig.set_size_inches(tight_fwidth, tight_fheight)

    #---- convert mpl figure to QPixmap ----

    buf, size = fig.canvas.print_to_buffer()
    qimage = QtGui.QImage.rgbSwapped(QtGui.QImage(buf, size[0], size[1],
                                                  QtGui.QImage.Format_ARGB32))
    qpixmap = QtGui.QPixmap(qimage)

    return qpixmap

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

        self.draw()

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

