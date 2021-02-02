# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================
"""
Package defining the *core* methods of the |scpy| API.

Most the API methods such as plotting, processing, analysis, etc...
"""

# ----------------------------------------------------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------------------------------------------------

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
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ======================================================================================================================
# Tells here the methods or object we allow to import from this library
# ======================================================================================================================

__all__ = [

        # Useful librairies alias for the end user avoiding to load them when wild card import is used
        # --------------------------------------------------------------------------------------------

        'np', 'plt', 'scipy', 'os', 'sys', 'mpl',

        # methods and objects from other packages will be added
        # later on this module (see below)

]

# ======================================================================================================================
# logging functions
# ======================================================================================================================

from spectrochempy.utils import pstr  # noqa: E402


def print_(*args, **kwargs):
    """
    Formatted printing
    """
    s = ""
    for a in args:
        s += pstr(a, **kwargs) + ' '
    s = s.replace('\0', '').strip()
    print(s)


# ----------------------------------------------------------------------------------------------------------------------
def info_(*args, **kwargs):
    s = ""
    for a in args:
        s += pstr(a, **kwargs) + ' '
    s = s.replace('\0', '').strip()
    app.logs.info(s)


# ----------------------------------------------------------------------------------------------------------------------
def debug_(*args, **kwargs):
    s = ""
    for a in args:
        s += pstr(a, **kwargs) + ' '
        s = s.replace('\0', '').strip()
    try:
        app.logs.debug(s)
    except NameError:
        # works only if app if already loaded
        pass


# ----------------------------------------------------------------------------------------------------------------------
def error_(*args, **kwargs):
    s = ''
    if not isinstance(args[0], str):
        s += type(args[0]).__name__ + ": "
    for a in args:
        s += pstr(a, **kwargs) + ' '
        s = s.replace('\0', '').strip()
    app.logs.error(s)


# ----------------------------------------------------------------------------------------------------------------------
def warning_(*args, **kwargs):
    s = ""
    for a in args:
        s += pstr(a, **kwargs) + ' '
        s = s.replace('\0', '').strip()
    app.logs.warning(s)


__all__ += ['info_', 'debug_', 'error_', 'warning_', 'print_']

# ======================================================================================================================
# Progress bar
# ======================================================================================================================
pbar_count = 0

USE_TQDM = environ.get('USE_TQDM', 'Yes') == 'Yes' and 'DOC_BUILDING' not in environ and "/bin/scpy" not in sys.argv[0]

if USE_TQDM:
    pbar = tqdm(total=1211)
    pbar.set_description('Loading SpectroChemPy API')
    val_tqdm = [1, 39, 52, 83, 83, 89, 92, 93, 94, 95, 96, 97, 98, 99, 100]


def _pbar_update(close=None):
    global pbar_count

    if USE_TQDM:

        if close:
            pbar.clear()
            pbar.close()
            pbar_count = 0
        else:
            pbar.update(val_tqdm[pbar_count])
            pbar_count += 1
    else:
        pass


# ======================================================================================================================
# loading module libraries
# here we also construct the __all__ list automatically
# ======================================================================================================================

_pbar_update()
from spectrochempy.application import SpectroChemPy  # noqa: E402

app = SpectroChemPy()
__all__ += ['app']

from spectrochempy.application import (  # noqa: E402
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
project_preferences = app.project_preferences
dataset_preferences = app.dataset_preferences
matplotlib_preferences = app.matplotlib_preferences
description = app.description
long_description = app.long_description
config_manager = app.config_manager
config_dir = app.config_dir


# datadir = app.datadir

def set_loglevel(level=WARNING):
    preferences.log_level = level


def get_loglevel():
    return preferences.log_level


__all__ += [
        # Helpers
        'DEBUG',
        'WARNING',
        'ERROR',
        'CRITICAL',
        'INFO',
        'project_preferences',
        'dataset_preferences',
        'preferences',
        'matplotlib_preferences',
        'config_manager',
        'config_dir',
        'set_loglevel',
        'get_loglevel',

        # Info
        'copyright',
        'version',
        'release',
        'license',
        'url',
        'release_date',
        'authors',
        'contributors',
        'description',
        'long_description'
        ]

# IPython methods
# ----------------------------------------------------------------------------------------------------------------------
# we put them before so that we can eventually overwrite them

_pbar_update()

# constants
# ----------------------------------------------------------------------------------------------------------------------
from spectrochempy.utils import show, MASKED, NOMASK, EPSILON, INPLACE  # noqa: E402

__all__ += ['show', 'MASKED', 'NOMASK', 'EPSILON', 'INPLACE']

# dataset
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update()
from spectrochempy.core.dataset.api import *  # noqa: E402,F403,F401
from spectrochempy.core.dataset import api  # noqa: E402

__all__ += api.__all__

# plotters
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update()
from spectrochempy.core.plotters.api import *  # noqa: E402,F403,F401
from spectrochempy.core.plotters import api  # noqa: E402

__all__ += api.__all__

# processors
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update()
from spectrochempy.core.processors.api import *  # noqa: E402,F403,F401
from spectrochempy.core.processors import api  # noqa: E402

__all__ += api.__all__

# readers
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update()
from spectrochempy.core.readers.api import *  # noqa: E402,F403,F401
from spectrochempy.core.readers import api  # noqa: E402

__all__ += api.__all__

# writers
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update()
from spectrochempy.core.writers.api import *  # noqa: E402,F403,F401
from spectrochempy.core.writers import api  # noqa: E402

__all__ += api.__all__

# units
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update()
from spectrochempy.units.units import *  # noqa: E402,F403,F401
from spectrochempy.units import units  # noqa: E402

__all__ += units.__all__

# databases
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update()
from spectrochempy.databases.api import *  # noqa: E402,F403,F401
from spectrochempy.databases import api  # noqa: E402

__all__ += api.__all__

# analysis
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update()
from spectrochempy.core.analysis.api import *  # noqa: E402,F403,F401
from spectrochempy.core.analysis import api  # noqa: E402

__all__ += api.__all__

# fitting
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update()
from spectrochempy.core.fitting.api import *  # noqa: E402,F403,F401
from spectrochempy.core.fitting import api  # noqa: E402

__all__ += api.__all__

# project
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update()
from spectrochempy.core.project.api import *  # noqa: E402,F403,F401
from spectrochempy.core.project import api  # noqa: E402

__all__ += api.__all__

# script
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update()
from spectrochempy.core.scripts.api import *  # noqa: E402,F403,F401
from spectrochempy.core.scripts import api  # noqa: E402

__all__ += api.__all__

# widgets
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update()
from spectrochempy.widgets.api import *  # noqa: E402,F403,F401
from spectrochempy.widgets import api  # noqa: E402

__all__ += api.__all__


# Helpers
# ----------------------------------------------------------------------------------------------------------------------

def APIref():
    """
    Helper to display public objects and methods from the API
    """
    a = __all__[:]
    a = sorted(a)
    return a


APIref = APIref()

__all__.append('APIref')

# START THE app

_pbar_update()

_pbar_update(close=True)

_started = app.start()

warnings.filterwarnings(action='ignore', module='matplotlib', category=UserWarning)

# ----------------------------------------------------------------------------------------------------------------------
# File Dialogs
# ----------------------------------------------------------------------------------------------------------------------

# can not be in utils due to circular imports
__all__ += ['open_dialog', 'save_dialog']

USE_QT = preferences.use_qt

if USE_QT:

    try:
        from PyQt5 import QtWidgets

        GUI = QtWidgets.QApplication(sys.argv)
        FileDialog = QtWidgets.QFileDialog

    except ImportError:
        # Qt not found - use Tkinter
        USE_QT = False

if not USE_QT:
    from tkinter import filedialog


# ------------------------------------------------------------------------------------------------------------------
# Private functions
# ------------------------------------------------------------------------------------------------------------------

class _QTFileDialogs:

    @classmethod
    def _open_existing_directory(cls,
                                 parent=None,
                                 caption='Select a folder',
                                 directory=''):
        options = FileDialog.DontResolveSymlinks | FileDialog.ShowDirsOnly | FileDialog.DontUseNativeDialog
        directory = FileDialog.getExistingDirectory(parent,
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
        options = FileDialog.DontUseNativeDialog
        filename, _ = FileDialog.getOpenFileName(parent,
                                                 caption=caption,
                                                 directory=directory,
                                                 filter=';;'.join(filters),
                                                 options=options)
        if filename:
            return filename

    # noinspection PyRedundantParentheses
    @classmethod
    def _open_multiple_filenames(
            cls,
            parent=None,
            directory='',
            caption='Select file(s)',
            filters=None):
        """
        Return one or several files to open
        """
        options = FileDialog.DontUseNativeDialog
        files, _ = FileDialog.getOpenFileNames(parent,
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

        options = FileDialog.DontUseNativeDialog
        options |= FileDialog.DontConfirmOverwrite  # bug : this seems to work only with DontUseNativeDialog on OSX.
        # TODO: Check on windows and Linux
        # second problems: if we confirm overwrite here a new dialog is opened,
        # and thus the main one do not close on exit!
        filename, _ = FileDialog.getSaveFileName(parent=None,
                                                 caption=caption,
                                                 directory=str(filename),
                                                 initialFilter=selected_filter,
                                                 filter=';;'.join(filters),
                                                 options=options)
        if filename:
            return filename


class _TKFileDialogs:

    def __init__(self):
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)
        root.geometry('0x0+0+0')
        root.deiconify()
        root.lift()
        root.focus_force()
        self.root = root
        # self.root.mainloop()


    def _open_existing_directory(self,
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
    def _open_filename(self, filters=None):

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
            filters=None):
        """
        Return one or several files to open
        """

        filename = filedialog.askopenfilenames(
                parent=self.root,
                filetypes=self.filetypes(filters) + [("all files", ('*'))],
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
            raise Exception("Save error")

        if not dftext:
            dftext = '.scp'

        # -defaultextension, -filetypes, -initialdir, -initialfile, -message, -parent, -title, -typevariable,
        # -command, or -confirmoverwrite
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
    if USE_QT:
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
    if USE_QT:
        klass = _QTFileDialogs
    else:
        klass = _TKFileDialogs()

    if directory is None:
        directory = ''
    if filters == 'directory':
        caption = 'Select a folder'
        f = klass._open_existing_directory(caption=caption, directory=str(directory))
    elif single:
        f = klass._open_filename(filters=filters)
    else:
        f = klass._open_multiple_filenames(filters=filters)

    from spectrochempy.utils.file import pathclean

    return pathclean(f)


# ======================================================================================================================
if __name__ == '__main__':
    pass
