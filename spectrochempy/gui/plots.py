# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

import matplotlib.pyplot as plt

from ..extern.pyqtgraph.parametertree import (Parameter,
                                                       ParameterTree, )
from ..extern.pyqtgraph.dockarea import Dock
from ..extern.pyqtgraph.functions import mkPen
from ..extern.pyqtgraph import PlotWidget
from .widgets.matplotlibwidget import MatplotlibWidget


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
            # we need to create a dock object.
            dp = Dock(key, size=(1000, 600), closable=True)
            dp.sigClosed.connect(self.plot_closed)

        # get the real object
        data = self.wproject.project
        for item in key.split('.'):
            data = data[item]

        if self.usempl:
            # or use matplotlib (The slower option)
            wplot = MatplotlibWidget()
            fig = wplot.getFigure()
            fig.clf()
            ax = fig.gca()
            # simplify plot for better interactivity (#TODO:options)
            plt.rcParams['path.simplify']=True
            plt.rcParams['path.simplify_threshold'] = 1.
            data.plot(ax=ax)
            wplot.draw()

        else:
            # use the Qt's GraphicsView framework offered by Pyqtgraph
            wplot = PlotWidget(title="")
            wplot.plot(data,
                       clear=True,
                       pen=mkPen('#FFCC33', width=3),
                       symbol='o',
                       symbolSize=7 )

        # --------------------------------------------------------------------
        # update dockarea
        # --------------------------------------------------------------------

        dp.addWidget(wplot)
        self.area.addDock(dp, 'above', self.dplots)
        opens[key]= (dp, wplot, self.usempl)


    def plot_closed(self, dock):

        del self.open_plots[dock.name()]


    def show_options(self):
        # ------------------------------------------------------------------------
        # Option window
        # ------------------------------------------------------------------------

        doptions = Dock("Options", size=(ww * .20, wh * .50), closable=False)
        self.options = t = ParameterTree(showHeader=False)
        doptions.addWidget(t)

        # """
        # 'type': None,
        #     'readonly': False,
        #     'visible': True,
        #     'enabled': True,
        #     'renamable': False,
        #     'removable': False,
        #     'strictNaming': False,  # forces name to be usable as a python
        #  variable
        #     'expanded': True,
        #     'title': None,
        #     """

        self.usempl = True

        self.scatter = True
        self.line = True
        self.stacked = True
        self.map = False
        self.image = False

        usempl = {
                'name': 'usempl',
                'title': 'Use Matplotlib?',
                'type': 'bool',
                'value': self.usempl,
                'tip': "Matplotlib is slower than the QtGraphicView,"
                       " but render nicer figures for publishing.",
                }

        dicprojet = {
            # PROJECT
            'name': 'Project',
            'type': 'group',
            'strictNaming': True,
            'children': [usempl]
            }

        scatter = {
                    'name': 'scatter',
                    'title': 'scatter plot',
                    'type': 'bool',
                    'value': self.scatter,
                    'tip': "type of 1D plot (scatter)",
                  }

        line = {
                    'name': 'line',
                    'title': 'line plot',
                    'type': 'bool',
                    'value': self.line,
                    'tip': "type of 1D plot (line)",
                }

        stacked = {
                    'name': 'stacked',
                    'title': 'stacked plot',
                    'type': 'bool',
                    'value': self.stacked,
                    'tip': "type of 2D plot (stacked)",
                  }

        mapped = {
                    'name': 'map',
                    'title': 'mapped plot',
                    'type': 'bool',
                    'value': self.map,
                    'tip': "type of 2D plot (map)",
                }

        imaged = {
                    'name': 'image',
                    'title': 'image',
                    'type': 'bool',
                    'value': self.image,
                    'tip': "type of 2D plot (image)"
                }

        dic1d = {   # 1D
                    'name': '1D',
                    'visible': (self.dataset_showed and
                    self.current_dataset.ndim==1),
                    'type': 'group',
                    'children': [scatter, line]
                }

        dic2d = {   # 2D
                    'name': '2D',
                    'visible': (self.dataset_showed
            and self.current_dataset.ndim==2),
                    'type': 'group',
                    'children': [stacked, mapped, imaged ]
                }

        dicdataset = {
            # DATASET
            'name': 'Dataset',
            'visible': self.dataset_showed,
            'type': 'group',
            'strictNaming': True,
            'children': [dic1d, dic2d]
            }

        # Todo: add nd

        params = [dicprojet]

        ## Create tree of Parameter objects

        p = Parameter.create(name='params', type='group', children=params)
        t.setParameters(p, showTop=False)

        ## If anything changes in the tree, print a message

        def change(param, changes):

            for param, change, data in changes:
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

        p.sigTreeStateChanged.connect(change)

        def valueChanging(param, value):
            self.wconsole.write(
                "Value changing (not finalized): %s %s" % (param, value))

        # Too lazy for recursion:
        for child in p.children():
            child.sigValueChanging.connect(valueChanging)
            for ch2 in child.children():
                ch2.sigValueChanging.connect(valueChanging)

        self.area.addDock(doptions, 'above', self.dflowchart)
