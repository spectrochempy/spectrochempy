# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
File I/O Dialog Module.

This module implements file dialog functionality for SpectroChemPy, supporting both
Qt and Tkinter backends. It provides:

- Open file dialogs (single and multiple file selection)
- Save file dialogs
- Directory selection dialogs

The implementation automatically selects between Qt and Tkinter based on:
- User preferences (preferences.use_qt)
- Environment variable SCPY_GUI
- Availability of PyQt5
"""

import re
from os import environ
from pathlib import Path

from spectrochempy.application import error_
from spectrochempy.core import preferences
from spectrochempy.utils.file import pathclean
from spectrochempy.utils.optional import import_optional_dependency

__all__ = ["open_dialog", "save_dialog"]

# Set flags
USE_QT = preferences.use_qt or environ.get("SCPY_GUI", None) == "RUNNING"

if USE_QT:  # pragma: no cover
    try:
        pyqt = import_optional_dependency("PyQt5.QtWidgets")
        FileDialog = pyqt.QFileDialog

    except ImportError as e:
        error_(ImportError, e)
        USE_QT = False
        from tkinter import filedialog

else:
    from tkinter import filedialog


# --------------------------------------------------------------------------------------
# Private functions
# --------------------------------------------------------------------------------------
class _QTFileDialogs:  # pragma: no cover
    """Qt-based file dialog implementations."""

    @classmethod
    def _open_existing_directory(
        cls,
        parent: object | None = None,
        caption: str = "Select a folder",
        directory: str | None = None,
    ) -> str | None:
        """
        Open dialog for selecting an existing directory.

        Parameters
        ----------
        parent : Optional[object]
            Parent widget
        caption : str
            Dialog window title
        directory : Optional[str]
            Initial directory to show

        Returns
        -------
        Optional[str]
            Selected directory path or None if cancelled

        """
        if directory is None:
            directory = str(preferences.datadir)

        options = FileDialog.DontResolveSymlinks | FileDialog.ShowDirsOnly
        directory = FileDialog.getExistingDirectory(
            parent=parent,
            caption=caption,
            directory=directory,
            options=options,
        )

        if directory:
            return directory

        return None

    @classmethod
    def _open_filename(
        cls,
        parent: object | None = None,
        directory: str | None = None,
        caption: str = "Select file",
        filters: list[str] | None = None,
    ) -> str | None:
        """
        Open dialog for selecting a file.

        Parameters
        ----------
        parent : Optional[object]
            Parent widget
        directory : Optional[str]
            Initial directory to show
        caption : str
            Dialog window title
        filters : Optional[List[str]]
            List of file type filters

        Returns
        -------
        Optional[str]
            Selected file path or None if cancelled

        """
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

    @classmethod
    def _open_multiple_filenames(
        cls,
        parent: object | None = None,
        directory: str | None = None,
        caption: str = "Select file(s)",
        filters: list[str] | None = None,
    ) -> list[str] | None:
        """
        Return one or several files to open.

        Parameters
        ----------
        parent : Optional[object]
            Parent widget
        directory : Optional[str]
            Initial directory to show
        caption : str
            Dialog window title
        filters : Optional[List[str]]
            List of file type filters

        Returns
        -------
        Optional[List[str]]
            List of selected file paths or None if cancelled

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
        parent: object | None = None,
        filename: str | None = None,
        caption: str = "Save as...",
        filters: list[str] | None = None,
    ) -> str | None:
        """
        Open dialog for saving a file.

        Parameters
        ----------
        parent : Optional[object]
            Parent widget
        filename : Optional[str]
            Default filename to suggest
        caption : str
            Dialog window title
        filters : Optional[List[str]]
            List of file type filters

        Returns
        -------
        Optional[str]
            Selected save file path or None if cancelled

        """
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
    """Tkinter-based file dialog implementations."""

    def __init__(self):
        """Initialize Tkinter root window."""
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
    def _open_existing_directory(
        parent: object | None = None,
        caption: str = "Select a folder",
        directory: str = "",
    ) -> str | None:
        """
        Open dialog for selecting an existing directory.

        Parameters
        ----------
        parent : Optional[object]
            Parent widget
        caption : str
            Dialog window title
        directory : str
            Initial directory to show

        Returns
        -------
        Optional[str]
            Selected directory path or None if cancelled

        """
        directory = filedialog.askdirectory(
            # parent=parent,
            initialdir=directory,
            title=caption,
        )

        if directory:
            return directory

        return None

    @staticmethod
    def filetypes(filters: list[str]) -> list[tuple[str, tuple[str, ...]]]:
        """
        Convert Qt filter patterns to Tkinter format.

        Parameters
        ----------
        filters : List[str]
            List of Qt-style filter patterns

        Returns
        -------
        List[Tuple[str, Tuple[str, ...]]]
            Tkinter-compatible file type specifications

        """
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

    def _open_filename(
        self,
        parent: object | None = None,
        filters: list[str] | None = None,
    ) -> str | None:
        """
        Open dialog for selecting a file.

        Parameters
        ----------
        parent : Optional[object]
            Parent widget
        filters : Optional[List[str]]
            List of file type filters

        Returns
        -------
        Optional[str]
            Selected file path or None if cancelled

        """
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

    def _open_multiple_filenames(
        self,
        parent: object | None = None,
        filters: list[str] | None = None,
    ) -> list[str] | None:
        """
        Return one or several files to open.

        Parameters
        ----------
        parent : Optional[object]
            Parent widget
        filters : Optional[List[str]]
            List of file type filters

        Returns
        -------
        Optional[List[str]]
            List of selected file paths or None if cancelled

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
        filename: str = "",
        caption: str = "Save as...",
        filters: list[str] | None = None,
    ) -> Path | None:
        """
        Open dialog for saving a file.

        Parameters
        ----------
        parent : Optional[object]
            Parent widget
        filename : str
            Default filename to suggest
        caption : str
            Dialog window title
        filters : Optional[List[str]]
            List of file type filters

        Returns
        -------
        Optional[Path]
            Selected save file path or None if cancelled

        """
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


def save_dialog(
    filename: str | Path | None = None,
    caption: str = "Save as...",
    filters: tuple[str, ...] = ("All Files (*)",),
    **kwargs,
) -> Path | None:  # pragma: no cover
    """
    Display a save file dialog.

    Parameters
    ----------
    filename : Optional[Union[str, Path]]
        Default filename to suggest
    caption : str
        Dialog window title
    filters : Tuple[str, ...]
        File type filters in Qt format
    **kwargs
        Additional arguments passed to dialog implementation

    Returns
    -------
    Optional[Path]
        Selected save file path or None if cancelled

    """
    if USE_QT:
        parent = kwargs.pop(
            "Qt_parent",
            None,
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
            filename=filename,
            caption=caption,
            filters=filters,
        )

    return pathclean(f)


def open_dialog(
    single: bool = True,
    directory: str | Path | None = None,
    filters: str | tuple[str, ...] = ("All Files (*)",),
    **kwargs,
) -> Path | list[Path] | None:  # pragma: no cover
    """
    Display an open file/directory dialog.

    Parameters
    ----------
    single : bool
        If True, allow selecting only one file
    directory : Optional[Union[str, Path]]
        Initial directory to show
    filters : Union[str, Tuple[str, ...]]
        File type filters in Qt format, or "directory" for directory selection
    **kwargs
        Additional arguments passed to dialog implementation

    Returns
    -------
    Optional[Union[Path, List[Path]]]
        Selected path(s) or None if cancelled

    """
    if USE_QT:
        parent = kwargs.pop(
            "Qt_parent",
            None,
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
            parent=parent,
            caption=caption,
            directory=str(directory),
        )
    elif single:
        f = klass._open_filename(parent=parent, filters=filters)
    else:
        f = klass._open_multiple_filenames(parent=parent, filters=filters)

    return pathclean(f)
