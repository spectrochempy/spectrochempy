# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

from ..extern.pyqtgraph.dockarea import Dock
from spectrochempy.gui.widgets.plotwidget import (MatplotlibWidget,
                                                  PyQtGraphWidget)

from spectrochempy.application import app

# --------------------------------------------------------------------
# Plot window
# --------------------------------------------------------------------

class Plots(object):

    dataset = None
    open_plots = {}
    usempl = False

    def show_or_create_plot_dock(self, key, update=False):
        """
        Utility function to create docked plots
    
        Parameters
        ----------
        self : class
            reference to the mainwindow
        key : str
            the dataset key in the project tree
        update : bool
            If true, redraw is enforced
    
        """

        opens = self.open_plots

        self.usempl = app.general_preferences.usempl

        if key in opens.keys():
            # get the existing ones
            dp, wplot, usempl = opens[key]
            if usempl != self.usempl or update:
                # delte old for redrawing
                dp.close()
                del wplot
            else:
                # nothing to do, except to raise the corresponding tab,
                # if necessary
                try:
                    dp.raiseDock()
                finally:
                    return

        if key not in opens.keys():
            # We do not use else, here because if one plot has been deleted it
            #  is not anymore in the keys - so we need to check this
            # we need to create a dock object.

            dp = Dock(key, size=(self.ww * self._ratio, self.wh),
                      closable=True)
            dp.sigClosed.connect(self.plot_closed)

        # --------------------------------------------------------------------
        # Get the dataset to plot
        # --------------------------------------------------------------------
        data = self.wproject.project
        for item in key.split('.'):
            data = data[item]

        # --------------------------------------------------------------------
        # Select the required plotwidget
        # --------------------------------------------------------------------

        if self.usempl:
            # use matplotlib (The slower option)
            wplot = MatplotlibWidget(data=data, name=key)
        else:
            # or use the Qt's GraphicsView framework offered by Pyqtgraph
            wplot = PyQtGraphWidget(data=data, name=key)

        # --------------------------------------------------------------------
        # update dockarea
        # --------------------------------------------------------------------

        dp.addWidget(wplot)

        if hasattr(self, 'dplots'):
            self.area.addDock(dp, 'above', self.dplots)
        else:
            # get the last dockked tab
            dtb = opens[list(opens.keys())[-1]][0]
            self.area.addDock(dp, 'above', dtb)

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


