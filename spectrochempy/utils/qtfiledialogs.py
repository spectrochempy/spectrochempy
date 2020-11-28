# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

# standalone dialogs

__all__ = []

from PyQt5 import QtWidgets

QFileDialog = QtWidgets.QFileDialog

def OpenExistingDirectory(parent=None,
                          caption='Select a folder',
                          directory=''):


    options = QFileDialog.DontResolveSymlinks | QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog
    directory = QFileDialog.getExistingDirectory(parent,
                                                 caption=caption,
                                                 directory=directory,
                                                 options=options)
    if directory:
        return directory


def OpenFileName(parent=None,
                 directory='',
                 caption='Select file',
                 filters=["All Files (*)"]):

    options = QFileDialog.AnyFile | QFileDialog.DontUseNativeDialog
    filename, _ = QFileDialog.getOpenFileName(parent,
                                              caption=caption,
                                              directory=directory,
                                              filter=';;'.join(filters),
                                              options=options)
    if filename:
        return filename


def OpenMultipleFileNames(
        parent=None,
        directory='',
        caption='Select file(s)',
        filters=["All Files (*)"]):

    options = QFileDialog.DontUseNativeDialog
    files, _ = QFileDialog.getOpenFileNames(parent,
                                            caption=caption,
                                            directory=directory,
                                            filter=';;'.join(filters),
                                            options=options)
    if files:
        return files


def savedialog(  filename='',
                 caption='Save as...',
                 selectedfilter = '',
                 filters=["All Files (*)"]):

    options = QFileDialog.DontUseNativeDialog
    options |= QFileDialog.DontConfirmOverwrite  # bug : this seems to work only with DontUseNativeDialog on OSX.
                                                 # TODO: Check on windows and Linux
                                                 # second problems: if we confirm owerwrite here a new dialog is opened,
                                                 # and thus the main one do not close on exit!
    filename, _ = QFileDialog.getSaveFileName(parent = None,
                                              caption=caption,
                                              directory=str(filename),
                                              initialFilter = selectedfilter,
                                              filter=';;'.join(filters),
                                              options=options)
    if filename:
        return filename


def opendialog(single=True,
               directory='',
               caption='',
               filters=["All Files (*)"]
               ):

    if filters == 'directory':
        if not caption:
            caption = 'Select a folder'
        f = OpenExistingDirectory(caption=caption, directory=str(directory))
    elif single:
        if not caption:
            caption = 'Select file'
        f = OpenFileName(caption=caption, directory=str(directory), filters=filters)
    else:
        if not caption:
            caption = 'Select file(s)'
        f = OpenMultipleFileNames(caption=caption, directory=str(directory), filters=filters)

    from spectrochempy.utils.file import pathclean
    return pathclean(f)


# ======================================================================================================================
if __name__ == '__main__':
    pass

