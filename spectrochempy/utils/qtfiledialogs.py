# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""
Dialog utilities

"""

__all__ = ['open_dialog', 'save_dialog']

from PyQt5 import QtWidgets

QFileDialog = QtWidgets.QFileDialog


# ------------------------------------------------------------------------------------------------------------------
# Private functions
# ------------------------------------------------------------------------------------------------------------------

def _open_existing_directory(parent=None,
                             caption='Select a folder',
                             directory=''):
    options = QFileDialog.DontResolveSymlinks | QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog
    directory = QFileDialog.getExistingDirectory(parent,
                                                 caption=caption,
                                                 directory=directory,
                                                 options=options)
    if directory:
        return directory


# noinspection PyRedundantParentheses
def _open_filename(parent=None,
                   directory='',
                   caption='Select file',
                   filters=("All Files (*)")):
    options = QFileDialog.DontUseNativeDialog
    filename, _ = QFileDialog.getOpenFileName(parent,
                                              caption=caption,
                                              directory=directory,
                                              filter=';;'.join(filters),
                                              options=options)
    if filename:
        return filename


# noinspection PyRedundantParentheses
def _open_multiple_filenames(
        parent=None,
        directory='',
        caption='Select file(s)',
        filters=("All Files (*)")):
    """
    Return one or several files to open

    """
    options = QFileDialog.DontUseNativeDialog
    files, _ = QFileDialog.getOpenFileNames(parent,
                                            caption=caption,
                                            directory=directory,
                                            filter=';;'.join(filters),
                                            options=options)
    if files:
        return files


# ------------------------------------------------------------------------------------------------------------------
# Public functions
# ------------------------------------------------------------------------------------------------------------------
# noinspection PyRedundantParentheses
def save_dialog(filename='',
                caption='Save as...',
                selected_filter='',
                filters=("All Files (*)")):
    """
    Return a file where to save

    """
    options = QFileDialog.DontUseNativeDialog
    options |= QFileDialog.DontConfirmOverwrite  # bug : this seems to work only with DontUseNativeDialog on OSX.
    # TODO: Check on windows and Linux
    # second problems: if we confirm overwrite here a new dialog is opened,
    # and thus the main one do not close on exit!
    filename, _ = QFileDialog.getSaveFileName(parent=None,
                                              caption=caption,
                                              directory=str(filename),
                                              initialFilter=selected_filter,
                                              filter=';;'.join(filters),
                                              options=options)
    if filename:
        return filename


# noinspection PyRedundantParentheses
def open_dialog(single=True,
                directory='',
                filters=("All Files (*)")
                ):
    """
    Return one or several files to open

    """
    if directory is None:
        directory = ''
    if filters == 'directory':
        caption = 'Select a folder'
        f = _open_existing_directory(caption=caption, directory=str(directory))
    elif single:
        caption = 'Select file'
        f = _open_filename(caption=caption, directory=str(directory), filters=filters)
    else:
        caption = 'Select file(s)'
        f = _open_multiple_filenames(caption=caption, directory=str(directory), filters=filters)

    from spectrochempy.utils.file import pathclean
    return pathclean(f)


# ======================================================================================================================
if __name__ == '__main__':
    pass
