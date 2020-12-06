# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""
Dialog utilities

"""
import sys

__all__ = ['open_dialog', 'save_dialog']

try:
    from PyQt5 import QtWidgets

    GUI = QtWidgets.QApplication(sys.argv)
    QFileDialog = QtWidgets.QFileDialog
    USEQT = True

except ImportError:
    # Qt not found - use Tkinter
    import tkinter as tk
    from tkinter import filedialog

    USEQT = False


# ------------------------------------------------------------------------------------------------------------------
# Private functions
# ------------------------------------------------------------------------------------------------------------------

class _QTFileDialogs:

    @classmethod
    def _open_existing_directory(cls,
                                 parent=None,
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
    @classmethod
    def _open_filename(cls,
                       parent=None,
                       directory='',
                       caption='Select file',
                       filters=None):
        options = QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(parent,
                                                  caption=caption,
                                                  directory=directory,
                                                  filter=';;'.join(filters),
                                                  options=options)
        if filename:
            return filename

    # noinspection PyRedundantParentheses
    @classmethod
    def _open_multiple_filenames(
            parent=None,
            directory='',
            caption='Select file(s)',
            filters=None):
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

    @classmethod
    def _save_filename(cls,
                       filename='',
                       caption='Save as...',
                       selected_filter='',
                       filters=None):

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


class _TKFileDialogs:

    def __init__(self):
        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)
        root.geometry('0x0+0+0')
        root.deiconify()
        root.lift()
        root.focus_force()
        self.root = root

    def _open_existing_directory(self,
                                 parent=None,
                                 caption='Select a folder',
                                 directory=''):

        directory = filedialog.askdirectory(parent=self.root, initialdir=directory, title=caption)

        if directory:
            return directory

    @staticmethod
    def filetypes(filters):
        # convert QT filters to TK
        import re

        regex = r"(.*)\((.*)\)"
        filetypes = []
        for filter in filters:
            matches = re.finditer(regex, filter)
            match = list(matches)[0]
            g = list(match.groups())
            g[1] = g[1].replace('[0-9]', '')
            g[1] = g[1].replace('1[r|i]', '*.*')
            g[1] = g[1].replace('2[r|i]*', '*.*')
            g[1] = g[1].replace('3[r|i]*', '*.*')
            g[1] = g[1].replace(' ', ',')
            g[1] = tuple(set(g[1].split(',')))
            filetypes.append((g[0], (g[1])))
        return filetypes

    # noinspection PyRedundantParentheses
    def _open_filename(self,
                       parent=None,
                       directory='',
                       caption='Select file',
                       filters=None):

        filename = filedialog.askopenfilename(
                parent=self.root,
                filetypes=self.filetypes(filters),
                title='Select file to open')

        self.root.destroy()
        if filename:
            return filename

    # noinspection PyRedundantParentheses
    def _open_multiple_filenames(
            self,
            parent=None,
            directory='',
            caption='Select file(s)',
            filters=None):
        """
        Return one or several files to open

        """

        filename = filedialog.askopenfilenames(
                parent=self.root,
                filetypes=self.filetypes(filters)+[("all files", ('*') )],
                title='Select file(s) to open')

        self.root.destroy()
        if filename:
            return filename


    def _save_filename(self,
                       filename='',
                       caption='Save as...',
                       selected_filter='',
                       filters=None):

        from spectrochempy.utils import pathclean
        dftext = ''
        directory = '.'
        if filename:
            filename = pathclean(filename)
            directory = filename.parent
            dftext = filename.suffix

        if not dftext and selected_filter:
            print()

        if not dftext:
            dftext = '.scp'

        # -defaultextension, -filetypes, -initialdir, -initialfile, -message, -parent, -title, -typevariable, -command, or -confirmoverwrite
        filename = filedialog.asksaveasfilename(parent=self.root,
                                                  title=caption,
                                                  initialdir=str(directory),
                                                  initialfile=filename.name,
                                                  defaultextension=dftext,
                                                  filetypes=self.filetypes(filters))
        if filename:
            return pathclean(filename)


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
    if USEQT:
        f = _QTFileDialogs._save_filename(filename,
                                          caption,
                                          selected_filter,
                                          filters)
    else:
        f = _TKFileDialogs()._save_filename(filename,
                                          caption,
                                          selected_filter,
                                          filters)

    from spectrochempy.utils.file import pathclean
    return pathclean(f)


# noinspection PyRedundantParentheses
def open_dialog(single=True,
                directory='',
                filters=("All Files (*)")
                ):
    """
    Return one or several files to open

    """
    if USEQT:
        klass = _QTFileDialogs
    else:
        klass = _TKFileDialogs()

    if directory is None:
        directory = ''
    if filters == 'directory':
        caption = 'Select a folder'
        f = klass._open_existing_directory(caption=caption, directory=str(directory))
    elif single:
        caption = 'Select file'
        f = klass._open_filename(caption=caption, directory=str(directory), filters=filters)
    else:
        caption = 'Select file(s)'
        f = klass._open_multiple_filenames(caption=caption, directory=str(directory), filters=filters)

    from spectrochempy.utils.file import pathclean
    return pathclean(f)


# ======================================================================================================================
if __name__ == '__main__':
    pass
