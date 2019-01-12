# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

import sys
import os
from spectrochempy.extern.pyqtgraph.Qt import QtGui, QtCore
from spectrochempy.gui.widgets.commonwidgets import OpenFileName, OpenExistingDirectory
from spectrochempy.application import project_preferences, general_preferences as prefs

class FilePickerWidget(QtGui.QWidget):
    """
    A file picker widget
    """

    def __init__(self, typ='file'):

        QtGui.QWidget.__init__(self)
        self.typ = typ
        # vertical layout for widgets
        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        # Create a label which displays the path to our chosen file
        self.qed = QtGui.QLineEdit()
        self.vbox.addWidget(self.qed)

        # Create a push button labelled 'choose' and add it to our layout
        btn = QtGui.QPushButton('Choose %s'%self.typ, self)
        self.vbox.addWidget(btn, 0, QtCore.Qt.AlignRight)

        # Connect the clicked signal to the get_fname handler
        btn.clicked.connect(self.get_fname)
        self.editingFinished = self.qed.editingFinished
        self.text = self.qed.text
        self.setText = self.qed.setText
        self.textChanged = self.qed.textChanged

    def get_fname(self):
        """
        Handler called when 'choose file' is clicked
        """
        t = self.typ
        text = self.text()
        if os.path.exists(text) and os.path.isdir(text):
            directory = text
        elif os.path.exists(text):
            directory = os.path.dirname(text)
        elif t=='project':
            directory = project_preferences.project_directory
        else:
            directory = prefs.datadir

        # when only folders are requested
        if t == 'folder':
            fname = OpenExistingDirectory(self,
                                          directory=directory,
                                          caption ="Select project's folder")

        # else we want a file
        elif t == 'project':
            fname = OpenFileName(self,
                                 directory=directory,
                                 caption="Select a project",
                                 filters = ["Project files (*.pscp)"])
        else:
            fname = OpenFileName(self,
                                 caption ="Select a file",
                                 directory=self.text(),
                                 filters=["SpectroChemPy files (*.scp;*.scpy)",
                                         "Project files (*.pscp)",
                                         "All Files (*)"])

        if fname:
            self.setText(fname)
        else:
            self.setText(self.text())

        self.editingFinished.emit()

# =============================================================================
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    gui = FilePickerWidget()
    gui.show()
    x= app.exec()
    print(x)