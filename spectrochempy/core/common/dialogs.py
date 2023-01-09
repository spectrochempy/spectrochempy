# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implement File I/O Dialogs
"""

from os import environ

from spectrochempy.core import error_, preferences
from spectrochempy.utils import import_optional_dependency

__all__ = ["open_dialog", "save_dialog"]

# Set flags
USE_QT = preferences.use_qt or environ.get("SCPY_GUI", None) == "RUNNING"

if USE_QT:  # pragma: no cover

    try:
        pyqt = import_optional_dependency("PyQt5.QtWidgets")
        FileDialog = pyqt.QFileDialog

    except ImportError as e:
        error_(e)
        USE_QT = False
        from tkinter import filedialog

else:

    from tkinter import filedialog


# ------------------------------------------------------------------------
# Private functions
# ------------------------------------------------------------------------


class _QTFileDialogs:  # pragma: no cover
    @classmethod
    def _open_existing_directory(
        cls, parent=None, caption="Select a folder", directory=None
    ):

        if directory is None:
            directory = str(preferences.datadir)

        options = FileDialog.DontResolveSymlinks | FileDialog.ShowDirsOnly
        directory = FileDialog.getExistingDirectory(
            parent=parent, caption=caption, directory=directory, options=options
        )

        if directory:
            return directory

        return None

    # noinspection PyRedundantParentheses
    @classmethod
    def _open_filename(
        cls, parent=None, directory=None, caption="Select file", filters=None
    ):

        if directory is None:
            directory = str(preferences.datadir)

        filename, _ = FileDialog.getOpenFileName(
            parent=parent,
            caption=caption,
            directory=directory,
            filter=";;".join(filters),
        )
        if filename:
            return filename

        return None

    # noinspection PyRedundantParentheses
    @classmethod
    def _open_multiple_filenames(
        cls, parent=None, directory=None, caption="Select file(s)", filters=None
    ):
        """
        Return one or several files to open
        """

        if directory is None:
            directory = str(preferences.datadir)

        files, _ = FileDialog.getOpenFileNames(
            parent=parent,
            caption=caption,
            directory=directory,
            filter=";;".join(filters),
        )
        if files:
            return files

        return None

    @classmethod
    def _save_filename(
        cls,
        parent=None,
        filename=None,
        caption="Save as...",
        filters=None,
    ):

        directory = str(filename)

        options = (
            FileDialog.DontConfirmOverwrite
        )  # bug : this seems to work only with DontUseNativeDialog on OSX.
        # TODO: Check on windows and Linux
        # second problems: if we confirm overwrite here a new dialog is opened,
        # and thus the main one do not close on exit!
        filename, _ = FileDialog.getSaveFileName(
            parent=parent,
            caption=caption,
            directory=directory,
            filter=";;".join(filters),
            options=options,
        )
        if filename:
            return filename

        return None


class _TKFileDialogs:  # pragma: no cover
    def __init__(self):
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)
        root.geometry("0x0+0+0")
        root.deiconify()
        root.lift()
        root.focus_force()
        self.root = root

    @staticmethod
    def _open_existing_directory(parent=None, caption="Select a folder", directory=""):

        directory = filedialog.askdirectory(
            # parent=parent,
            initialdir=directory,
            title=caption,
        )

        if directory:
            return directory

        return None

    @staticmethod
    def filetypes(filters):
        # convert QT filters to TK
        import re

        regex = r"(.*)\((.*)\)"
        filetypes = []
        for _filter in filters:
            matches = re.finditer(regex, _filter)
            match = list(matches)[0]
            g = list(match.groups())
            g[1] = g[1].replace("[0-9]", "")
            g[1] = g[1].replace("1[r|i]", "*.*")
            g[1] = g[1].replace("2[r|i]*", "*.*")
            g[1] = g[1].replace("3[r|i]*", "*.*")
            g[1] = g[1].replace(" ", ",")
            g[1] = tuple(set(g[1].split(",")))
            filetypes.append((g[0], (g[1])))
        return filetypes

    # noinspection PyRedundantParentheses
    def _open_filename(
        self,
        parent=None,
        filters=None,
    ):

        filename = filedialog.askopenfilename(
            # parent=parent,
            filetypes=self.filetypes(filters),
            title="Select file to open",
        )

        if parent is not None:
            parent.destroy()

        if filename:
            return filename

        return None

    # noinspection PyRedundantParentheses
    def _open_multiple_filenames(self, parent=None, filters=None):
        """
        Return one or several files to open
        """
        filename = filedialog.askopenfilenames(
            # parent=parent,
            filetypes=self.filetypes(filters) + [("all files", ("*"))],
            title="Select file(s) to open",
        )

        if parent is not None:
            parent.destroy()

        if filename:
            return filename

        return None

    def _save_filename(
        self,
        # parent=None,
        filename="",
        caption="Save as...",
        filters=None,
    ):

        from spectrochempy.utils import pathclean

        dftext = ""
        directory = "."
        if filename:
            filename = pathclean(filename)
            directory = filename.parent
            dftext = filename.suffix

        if not dftext:
            dftext = ".scp"

        # -defaultextension, -filetypes, -initialdir, -initialfile, -message, -parent,
        # -title, -typevariable,
        # -command, or -confirmoverwrite
        filename = filedialog.asksaveasfilename(
            # parent=parent,
            title=caption,
            initialdir=str(directory),
            initialfile=filename.name,
            defaultextension=dftext,
            filetypes=self.filetypes(filters),
        )
        #        if parent is not None:
        #            parent.destroy

        if filename:
            return pathclean(filename)

        return None


# --------------------------------------------------------------------------------------
# Public functions
# --------------------------------------------------------------------------------------

# noinspection PyRedundantParentheses
def save_dialog(
    filename=None, caption="Save as...", filters=("All Files (*)"), **kwargs
):  # pragma: no cover
    """
    Return a file where to save.
    """
    if USE_QT:
        parent = kwargs.pop(
            "Qt_parent", None
        )  # in case this is launched from spectrochempy_gui

        _ = pyqt.QApplication([])
        f = _QTFileDialogs._save_filename(
            parent=parent,
            filename=filename,
            caption=caption,
            filters=filters,
        )
    else:
        f = _TKFileDialogs()._save_filename(
            filename=filename, caption=caption, filters=filters
        )

    from spectrochempy.utils import pathclean

    return pathclean(f)


# noinspection PyRedundantParentheses
def open_dialog(
    single=True, directory=None, filters=("All Files (*)"), **kwargs
):  # pragma: no cover
    """
    Return one or several files to open.
    """
    if USE_QT:
        parent = kwargs.pop(
            "Qt_parent", None
        )  # in case this is launched from spectrochempy_gui

        _ = pyqt.QApplication([])
        klass = _QTFileDialogs
    else:
        klass = _TKFileDialogs()
        parent = klass.root

    if directory is None:
        directory = ""
    if filters == "directory":
        caption = "Select a folder"
        f = klass._open_existing_directory(
            parent=parent, caption=caption, directory=str(directory)
        )
    elif single:
        f = klass._open_filename(parent=parent, filters=filters)
    else:
        f = klass._open_multiple_filenames(parent=parent, filters=filters)

    from spectrochempy.utils import pathclean

    return pathclean(f)
