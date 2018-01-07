# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

from ...extern.pyqtgraph.Qt import QtGui, QtCore

QMessageBox = QtGui.QMessageBox
QFileDialog = QtGui.QFileDialog

def warningMessage(parent, message='Warning'):
    msgBox = QMessageBox(QMessageBox.Warning,
                               "Warning", message,
                               QMessageBox.NoButton, parent)
    msgBox.addButton("&Ok", QMessageBox.AcceptRole)
    msgBox.addButton("&Cancel", QMessageBox.RejectRole)
    return (msgBox.exec() == QMessageBox.AcceptRole)


def OpenExistingDirectory(parent,
                          caption='Select a folder',
                          directory=''):

    options = QFileDialog.DontResolveSymlinks | QFileDialog.ShowDirsOnly
    directory = QFileDialog.getExistingDirectory(parent,
                                                 caption=caption,
                                                 directory=directory,
                                                 options=options)
    if directory:
        return directory


def OpenFileName(parent=None,
                 directory = '',
                 caption='Select file',
                 filters=["All Files (*)","Text Files (*.txt)"]):

    options = QFileDialog.Options()
    options |= QFileDialog.AnyFile
    # options |= QFileDialog.DontUseNativeDialog
    filename, _ = QFileDialog.getOpenFileName(parent,
                                              caption=caption,
                                              directory=directory,
                                              filter=';;'.join(filters),
                                              options=options)
    if filename:
        return filename


def OpenMultipleFileNames(
                 parent=None,
                 directory = '',
                 caption='Select file(s)',
                 filters=["All Files (*)","Text Files (*.txt)"]):
    options = QFileDialog.Options()
    # options |= QFileDialog.DontUseNativeDialog
    files, _ = QFileDialog.getOpenFileNames(parent,
                                            caption=caption,
                                            directory=directory,
                                            filter=';;'.join(filters),
                                            options=options)
    if files:
        return files


def SaveFileName(parent=None,
                 filename = '',
                 caption='Select file',
                 filters=["All Files (*)","Text Files (*.txt)"]):
    options = QFileDialog.Options()
    # options |= QFileDialog.DontUseNativeDialog
    filename, _ = QFileDialog.getSaveFileName(parent,
                                              caption,
                                              filename,
                                              filters,
                                              options=options)
    if filename:
        return filename

#
# def criticalMessage(self):
#     reply = QMessageBox.critical(self, "QMessageBox.critical()",
#                                  Dialog.MESSAGE,
#                                  QMessageBox.Abort | QMessageBox.Retry |
#                                  QMessageBox.Ignore)
#     if reply == QMessageBox.Abort:
#         self.criticalLabel.setText("Abort")
#     elif reply == QMessageBox.Retry:
#         self.criticalLabel.setText("Retry")
#     else:
#         self.criticalLabel.setText("Ignore")
#
#
# def informationMessage(self):
#     reply = QMessageBox.information(self, "QMessageBox.information()",
#                                     Dialog.MESSAGE)
#     if reply == QMessageBox.Ok:
#         self.informationLabel.setText("OK")
#     else:
#         self.informationLabel.setText("Escape")
#
#
# def questionMessage(self):
#     reply = QMessageBox.question(self, "QMessageBox.question()",
#                                  Dialog.MESSAGE,
#                                  QMessageBox.Yes | QMessageBox.No |
#                                  QMessageBox.Cancel)
#     if reply == QMessageBox.Yes:
#         self.questionLabel.setText("Yes")
#     elif reply == QMessageBox.No:
#         self.questionLabel.setText("No")
#     else:
#         self.questionLabel.setText("Cancel")
#
# def errorMessage(self):
#     self.errorMessageDialog.showMessage("This dialog shows and remembers "
#                                         "error messages. If the checkbox "
#                                         "is checked (as it is by "
#                                         "default), the shown message "
#                                         "will be shown again, but if the "
#                                         "user unchecks the box the message "
#                                         "will not appear again if "
#                                         "QErrorMessage.showMessage() is "
#                                         "called with the same message.")
#     self.errorLabel.setText("If the box is unchecked, the message won't "
#                             "appear again.")
#

# =============================================================================
if __name__ == '__main__':
    pass
