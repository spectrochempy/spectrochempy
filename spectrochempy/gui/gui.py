import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, \
    QFileDialog
from PyQt5.QtGui import QIcon


class Gui(QWidget):
    def __init__(self):

        super().__init__()
        self.title = 'SpectroChemPy'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

    def openFileNameDialog(self,
                           directory="",
                           filter=""):

        filters = "SpectroChemPy (*.scp);;All Files (*)"
        if filter:
            filters = filter + ";;" + filters

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,
                                                  caption="Open ...",
                                                  directory=directory,
                                                  filter=filters,
                                                  options=options)
        if fileName:
            return fileName

    def openFileNamesDialog(self,
                            directory="",
                            filter=""):

        filters = "SpectroChemPy (*.scp);;All Files (*)"
        if filter:
            filters =  filter + ";;" + filters

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,
                                                caption="Open Multiples files ...",
                                                directory=directory,
                                                filter=filters,
                                                options=options)
        if files:
            return files

    def saveFileDialog(self,
                       directory="",
                       filter=""):

        filters = "SpectroChemPy (*.scp);;All Files (*)"
        if filter:
            filters = filter + ";;" + filters

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,
                                                  caption="Save file ...",
                                                  directory=directory,
                                                  filter=filters,
                                                  options=options)
        if fileName:
            return fileName


if __name__ == '__main__':
    guiapp = QApplication(sys.argv)

    ex = Gui()
    print((ex.openFileNameDialog()))

    # guiapp.exec_()
