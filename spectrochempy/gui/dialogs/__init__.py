# standalone dialogs

__all__ = []

from spectrochempy.gui.widgets.commonwidgets import *
from spectrochempy.extern.pyqtgraph.Qt import QtGui

import sys

def opendialog(single=True,
                   parent=None,
                   directory='',
                   caption='',
                   filters=["All Files (*)", "Text Files (*.txt)"]
                   ):
    """

    Parameters
    ----------
    single : Bool, optional, default: True
        Open single or multiple files
    directory : str, optional
        Folder where to start selection of files or folder
    caption : str, optional
        Caption for the dialog window, optional
    filters : str or list of str, optional, default : ["All Files (*)", "Text Files (*.txt)"]
        Type of files to select. if filters='directory', then the dialog is to select
        an existing directory

    Returns
    -------
    filename : str
        Filename or folder name

    """
    gui = QtGui.QApplication(sys.argv)
    if filters == 'directory':
        if not caption:
            caption = 'Select a folder'
        f = OpenExistingDirectory(caption=caption, directory=directory)
    elif single:
        if not caption:
            caption = 'Select file'
        f = OpenFileName(directory=directory, caption=caption, filters=filters)
    else:
        if not caption:
            caption = 'Select file(s)'
        f = OpenMultipleFileNames(directory=directory, caption=caption,
                                  filters=filters)

    gui.exit()
    return f

if __name__ == '__main__':
    f = opendialog()
    print(f)
    f = opendialog(single='False', caption='xxxx')
    print(f)
    f = opendialog(filters='directory')
    print(f)
