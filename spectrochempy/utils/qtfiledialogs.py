# standalone dialogs


__all__ = []


def QFileDialog():
    # delayed import
    from PyQt5 import QtWidgets
    return QtWidgets.QFileDialog

QFileDialog = QFileDialog()


def OpenExistingDirectory(parent=None,
                          caption='Select a folder',
                          directory=''):

    if QFileDialog is None:
        return

    options = QFileDialog.DontResolveSymlinks | QFileDialog.ShowDirsOnly
    directory = QFileDialog.getExistingDirectory(parent,
                                                 caption=caption,
                                                 directory=directory,
                                                 options=options)
    if directory:
        return directory


def OpenFileName(parent=None,
                 directory='',
                 caption='Select file',
                 filters=["All Files (*)", "Text Files (*.txt)"]):

    if QFileDialog is None:
        return

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
        directory='',
        caption='Select file(s)',
        filters=["All Files (*)", "Text Files (*.txt)"]):

    if QFileDialog is None:
        return

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
                 filename='',
                 caption='Select file',
                 filters=["All Files (*)", "Text Files (*.txt)"]):

    if QFileDialog is None:
        return

    options = QFileDialog.Options()
    # options |= QFileDialog.DontUseNativeDialog
    filename, _ = QFileDialog.getSaveFileName(parent,
                                              caption,
                                              filename,
                                              filters,
                                              options=options)
    if filename:
        return filename


def opendialog(single=True,
               parent=None,
               directory='',
               caption='',
               filters=["All Files (*)", "Text Files (*.txt)"]
               ):
    """

    Parameters
    ----------
    single : Bool, optional, default=True
        Open single or multiple files
    directory : str, optional
        Folder where to start selection of files or folder
    caption : str, optional
        Caption for the dialog window, optional
    filters : str or list of str, optional, default=["All Files (*)", "Text Files (*.txt)"]
        Type of files to select. if filters='directory', then the dialog is to select
        an existing directory

    Returns
    -------
    filename : str
        Filename or folder name

    """
    # gui = QtGui.QApplication(sys.argv)

    if QFileDialog is None:
        return

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

    return f


# ======================================================================================================================
if __name__ == '__main__':
    pass
