# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================


from ...extern.pyqtgraph.Qt import QtGui, QtCore

from ...core.projects.project import Project, NDDataset #, Script, Meta

__all__ = []

class ProjectTreeWidget(QtGui.QTreeWidget) :
    """
    Widget for displaying spectrochempy projects

    """

    project = None

    def __init__(self, parent=None, project=None, showHeader=True) :
        """
        Parameters
        ----------
        parent : object
        project : SpectroChemPy Project object

        """
        QtGui.QTreeWidget.__init__(self, parent)
        self.setVerticalScrollMode(self.ScrollPerPixel)
        self.setproject(project)
        self.setColumnCount(2)
        self.setHeaderLabels(['name', 'type'])
        self.setHeaderHidden(not showHeader)

    def setproject(self, project) :
        """
        Parameters
        ----------
        project: SpectroChemPy project object

        """
        self.clear()
        self.project = project
        self._buildtree(project, self.invisibleRootItem())
        self.expandToDepth(3)
        self.resizeColumnToContents(0)

    def _buildtree(self, obj, parent, name='') :

        if obj is None:
            node = QtGui.QTreeWidgetItem(['No project was loaded yet', ''])
            parent.addChild(node)
            return

        typeStr = type(obj).__name__
        if typeStr == 'Project' :
            name = obj.name
        node = QtGui.QTreeWidgetItem([name, typeStr])
        parent.addChild(node)

        if isinstance(obj, Project):
            for k in obj.allnames:
                self._buildtree(obj[k], node, k)
        elif isinstance(obj, NDDataset):
            node.setFlags(
                QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            node.setCheckState(0, QtCore.Qt.Unchecked)
        else:
            return
