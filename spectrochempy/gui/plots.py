# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================
import numpy as np
import copy

import matplotlib.pyplot as plt

from ..extern.pyqtgraph.parametertree import (Parameter,
                                                       ParameterTree, )
from ..extern.pyqtgraph.dockarea import Dock
from ..extern.pyqtgraph.functions import mkPen
from .widgets.PlotWidget import PlotWidget
from .widgets.matplotlibwidget import MatplotlibWidget

from spectrochempy.application import app

# --------------------------------------------------------------------
# Plot window
# --------------------------------------------------------------------

class Plots(object):

    dataset = None
    open_plots = {}
    usempl = False

    def show_or_create_plot_dock(self, key):
        """
        Utility function to create docked plots
    
        Parameters
        ----------
        self : class
            reference to the mainwindow
        branches : list(str)
            the branch elements in the project tree
    
    
        """

        opens = self.open_plots

        self.usempl = app.project_preferences.usempl

        if key in opens.keys():
            # get the existing ones
            dp, wplot, usempl = opens[key]
            if usempl != self.usempl:
                dp.close()
                del wplot
            else:
                # nothing to do, except to raise the corresponding tab
                dp.raiseDock()
                return

        if key not in opens.keys():
            # We do not use else, here because if one plot has been deleted it
            #  is not anymore in the keys - so we need to check this
            # we need to create a dock object.
            dp = Dock(key, size=(self.ww * self._ratio, self.wh),
                      closable=True)
            dp.sigClosed.connect(self.plot_closed)

        # --------------------------------------------------------------------
        # Select the required plotwidget
        # --------------------------------------------------------------------

        if self.usempl:
            # use matplotlib (The slower option)
            wplot = MatplotlibWidget()
        else:
            # or use the Qt's GraphicsView framework offered by Pyqtgraph
            wplot = PlotWidget(name=key)

        # --------------------------------------------------------------------
        # Get the dataset to plot and plot it
        # --------------------------------------------------------------------
        data = self.wproject.project
        for item in key.split('.'):
            data = data[item]

        wplot.plot(data)

            #wplot.plot(data,
            #           clear=True)
                       #pen=mkPen('#FFCC33', width=3),
                       #symbol='o',
                       #symbolSize=7 )

        # --------------------------------------------------------------------
        # update dockarea
        # --------------------------------------------------------------------

        dp.addWidget(wplot)
        try:
            self.area.addDock(dp, 'above', self.dplots)
        except:
            self.area.addDock(dp, 'above', opens[list(opens.keys())[-1]][0])
        opens[key]= (dp, wplot, self.usempl)
        if hasattr(self, 'dplots'):
            self.dplots.close()
            del self.dplots

    def plot_closed(self, dock):

        del self.open_plots[dock.name()]
        if not self.open_plots:
            # recreate the void area for plots
            self.dplots = self._create_plot_area()
            self.area.addDock(self.dplots, 'top', self.dconsole)


