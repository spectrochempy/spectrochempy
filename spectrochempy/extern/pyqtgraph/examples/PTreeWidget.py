# -*- coding: utf-8 -*-

"""
Simple use of DataTreeWidget to display a structure of nested dicts, lists, and arrays
"""

import initExample ## Add path to library (just for examples; you do not need this)

from spectrochempy_gui.widgets.projecttreewidget import ProjectTreeWidget
from spectrochempy.extern.pyqtgraph.Qt import QtCore, QtGui
import numpy as np

from spectrochempy.api import Project


app = QtGui.QApplication([])

d = Project.load('HIZECOKE')

tree = ProjectTreeWidget(project=d, hideRoot=False)
tree.show()
tree.setWindowTitle('pyqtgraph example: DataTreeWidget')
tree.resize(600,600)


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()