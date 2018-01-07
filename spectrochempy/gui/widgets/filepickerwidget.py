# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

import sys
from spectrochempy.extern.pyqtgraph.Qt import QtGui, QtCore

class FilePickerWidget(QtGui.QWidget):
    """
    A file picker widget
    """

    def __init__(self, typ='file'):

        QtGui.QWidget.__init__(self)
        self.typ = typ
        # vertical layout for widgets
        self.hbox = QtGui.QHBoxLayout()
        self.setLayout(self.hbox)

        # Create a label which displays the path to our chosen file
        self.qed = QtGui.QLineEdit()
        self.hbox.addWidget(self.qed)

        # Create a push button labelled 'choose' and add it to our layout
        btn = QtGui.QPushButton('Choose %s'%self.typ, self)
        self.hbox.addWidget(btn)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding,
                           QtGui.QSizePolicy.Expanding)

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
        dlg = QtGui.QFileDialog()
        if t == 'folder':
            dlg.setFileMode(QtGui.QFileDialog.Directory)
            dlg.setOption(QtGui.QFileDialog.ShowDirsOnly, True)
            fname = dlg.getExistingDirectory(self, "Select project's folder")
            if fname:
                self.setText(fname)
            else:
                self.setText(self.text)
            return

        elif t == 'project':
            dlg.setFilter("Project files (*.pscp)")
            fname = dlg.getOpenFileName(self, "Select project")
        else:
            fname = dlg.getOpenFileName(self, "Select file")

        if fname:
            self.setText(fname[0])
        else:
            self.setText(self.text)


# =============================================================================
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    gui = FilePickerWidget()
    gui.show()
    app.exec_()