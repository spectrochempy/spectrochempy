# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
Package defining the *core* methods of the |scpy| API.

Most the API methods such as plotting, processing, analysis, etc...
"""

# warnings.simplefilter('ignore', (DeprecationWarning,
#                                 FutureWarning, UserWarning))

import os
from os import environ
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy

from ..optional import import_optional_dependency

warnings.filterwarnings("ignore")

# ======================================================================================================================
# Tells here the methods or object we allow to import from this library
# ======================================================================================================================

__all__ = [
    # Useful librairies alias for the end user avoiding to load them when wild card import is used
    # --------------------------------------------------------------------------------------------
    "np",
    "plt",
    "scipy",
    "os",
    "sys",
    "mpl",
    # methods and objects from other packages will be added
    # later on this module (see below)
]

# ======================================================================================================================
# logging functions
# ======================================================================================================================

from ..utils import pstr  # noqa: E402


def _format_args(*args, **kwargs):
    stg = ""
    for arg in args:
        stg += pstr(arg, **kwargs) + " "
    return stg.replace("\0", "").strip()


def print_(*args, **kwargs):
    """
    Formatted printing.
    """
    stg = _format_args(*args, **kwargs)
    print(stg)


# ------------------------------------------------------------------
def info_(*args, **kwargs):
    """
    Formatted info message.
    """
    stg = _format_args(*args, **kwargs)
    app.logs.info(stg)


# ------------------------------------------------------------------
def debug_(*args, **kwargs):
    """
    Formatted debug message.
    """
    stg = _format_args(*args, **kwargs)
    try:
        app.logs.debug(stg)
    except NameError:  # pragma: no cover
        # works only if app if already loaded
        pass


# ------------------------------------------------------------------
def error_(*args, **kwargs):
    """
    Formatted error message.
    """
    stg = ""
    if not isinstance(args[0], str):
        stg += type(args[0]).__name__ + ": "
    stg += _format_args(*args, **kwargs)
    app.logs.error(stg)


# ------------------------------------------------------------------
def warning_(*args, **kwargs):
    """
    Formatted warning message.
    """
    stg = _format_args(*args, **kwargs)
    app.logs.warning(stg)


__all__ += ["info_", "debug_", "error_", "warning_", "print_"]

# ======================================================================================================================
# Progress bar
# ======================================================================================================================
PBAR_COUNT = 0

USE_TQDM = (
    environ.get("USE_TQDM", "Yes") == "Yes"
    and "DOC_BUILDING" not in environ
    and "/bin/scpy" not in sys.argv[0]
)

if USE_TQDM:
    from tqdm import tqdm

    pbar = tqdm(total=1211)
    pbar.set_description("Loading SpectroChemPy API")
    val_tqdm = [1, 39, 52, 83, 83, 89, 92, 93, 94, 95, 96, 97, 98, 99, 100]


def _pbar_update(close=None):
    global PBAR_COUNT

    if USE_TQDM:

        if close:
            pbar.clear()
            pbar.close()
            PBAR_COUNT = 0
        else:
            pbar.update(val_tqdm[PBAR_COUNT])
            PBAR_COUNT += 1
    else:
        pass


# ======================================================================================================================
# loading module libraries
# here we also construct the __all__ list automatically
# ======================================================================================================================

_pbar_update()
from ..application import SpectroChemPy  # noqa: E402

app = SpectroChemPy()
__all__ += ["app"]

from ..application import (  # noqa: E402
    __version__ as version,
    __release__ as release,
    __copyright__ as copyright,
    __license__ as license,
    __release_date__ as release_date,
    __author__ as authors,
    __contributor__ as contributors,
    __url__ as url,
    DEBUG,
    WARNING,
    ERROR,
    CRITICAL,
    INFO,
)

preferences = app.preferences
plot_preferences = app.plot_preferences
description = app.description
long_description = app.long_description
config_manager = app.config_manager
config_dir = app.config_dir
reset_preferences = app.reset_preferences


# datadir = app.datadir


def set_loglevel(level=WARNING):
    preferences.log_level = level


def get_loglevel():
    return preferences.log_level


__all__ += [
    # Helpers
    "DEBUG",
    "WARNING",
    "ERROR",
    "CRITICAL",
    "INFO",
    "preferences",
    "plot_preferences",
    "config_manager",
    "config_dir",
    "reset_preferences",
    "set_loglevel",
    "get_loglevel",
    # Info
    "copyright",
    "version",
    "release",
    "license",
    "url",
    "release_date",
    "authors",
    "contributors",
    "description",
    "long_description",
]

# IPython methods
# ------------------------------------------------------------------
# we put them before so that we can eventually overwrite them

_pbar_update()

# constants
# ------------------------------------------------------------------
from ..utils import show, MASKED, NOMASK, EPSILON, INPLACE  # noqa: E402

__all__ += ["show", "MASKED", "NOMASK", "EPSILON", "INPLACE"]

# dataset
# ------------------------------------------------------------------
_pbar_update()
from .dataset import api  # noqa: E402
from .dataset.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# plotters
# ------------------------------------------------------------------
_pbar_update()
from .plotters import api  # noqa: E402
from .plotters.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# processors
# ------------------------------------------------------------------
_pbar_update()
from .processors import api  # noqa: E402
from .processors.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# readers
# ------------------------------------------------------------------
_pbar_update()
from .readers import api  # noqa: E402
from .readers.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# writers
# ------------------------------------------------------------------
_pbar_update()
from .writers import api  # noqa: E402
from .writers.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# units
# ------------------------------------------------------------------
_pbar_update()
from ..units import units  # noqa: E402
from ..units.units import *  # noqa: E402,F403,F401

__all__ += units.__all__

# databases
# ------------------------------------------------------------------
_pbar_update()
from ..databases import api  # noqa: E402
from ..databases.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# analysis
# ------------------------------------------------------------------
_pbar_update()
from .analysis import api  # noqa: E402
from .analysis.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# fitting
# ------------------------------------------------------------------
_pbar_update()
from .fitting import api  # noqa: E402
from .fitting.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# project
# ------------------------------------------------------------------
_pbar_update()
from .project import api  # noqa: E402
from .project.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# script
# ------------------------------------------------------------------
_pbar_update()
from .scripts import api  # noqa: E402
from .scripts.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# widgets
# ------------------------------------------------------------------
_pbar_update()
from ..widgets import api  # noqa: E402
from ..widgets.api import *  # noqa: E402,F403,F401

__all__ += api.__all__


# Helpers
# ------------------------------------------------------------------


def APIref():
    """
    Helper to display public objects and methods from the API
    """
    a = __all__[:]
    a = sorted(a)
    return a


APIref = APIref()

__all__.append("APIref")

# START THE app

_pbar_update()

_pbar_update(close=True)

_started = app.start()

warnings.filterwarnings(action="ignore", module="matplotlib", category=UserWarning)

# ------------------------------------------------------------------
# File Dialogs
# ------------------------------------------------------------------

__all__ += ["open_dialog", "save_dialog"]

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
            parent=parent, initialdir=directory, title=caption
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
            parent=parent,
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
            parent=parent,
            filetypes=self.filetypes(filters) + [("all files", ("*"))],
            title="Select file(s) to open",
        )

        if parent is not None:
            parent.destroy()

        if filename:
            return filename

        return None

    def _save_filename(
        self, parent=None, filename="", caption="Save as...", filters=None
    ):

        from ..utils import pathclean

        dftext = ""
        directory = "."
        if filename:
            filename = pathclean(filename)
            directory = filename.parent
            dftext = filename.suffix

        if not dftext:
            dftext = ".scp"

        # -defaultextension, -filetypes, -initialdir, -initialfile, -message, -parent, -title, -typevariable,
        # -command, or -confirmoverwrite
        filename = filedialog.asksaveasfilename(
            parent=parent,
            title=caption,
            initialdir=str(directory),
            initialfile=filename.name,
            defaultextension=dftext,
            filetypes=self.filetypes(filters),
        )
        if parent is not None:
            parent.destroy

        if filename:
            return pathclean(filename)

        return None


# ------------------------------------------------------------------------
# Public functions
# ------------------------------------------------------------------------

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

    from ..utils import pathclean

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

    from ..utils import pathclean

    return pathclean(f)


# ======================================================================================================================
if __name__ == "__main__":
    pass
