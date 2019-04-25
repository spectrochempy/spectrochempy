# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""
Package defining the *core* methods of the |scpy| API such as plotting,
processing, analysis, etc...

"""

# ----------------------------------------------------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------------------------------------------------

import warnings

# warnings.simplefilter('ignore', (DeprecationWarning,
#                                 FutureWarning, UserWarning))

import os
import sys
import time

# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy

# ======================================================================================================================
# Tells here the methods or object we allow to import from this library
# ======================================================================================================================

__all__ = [
    # Useful librairies alias for the end user avoiding to load them
    # --------------------------------------------------------------

    'np', 'plt', 'scipy', 'os', 'sys', 'mpl',

    ### methods and objects from other packages will be added
    ### later on this module (see below)

]

# ======================================================================================================================
# loading module libraries
# here we also construct the __all__ list automatically
# ======================================================================================================================

from spectrochempy.application import SpectroChemPy

app = SpectroChemPy()
#__all__ += ['app']

# ======================================================================================================================
# logging functions
# ======================================================================================================================

from spectrochempy.utils import pstr

def print_(*args, **kwargs):
    """
    Formatted printing
    """
    s = ""
    for a in args:
        s += pstr(a, **kwargs)
    s = s.replace('\0', '')
    print(s)

# ----------------------------------------------------------------------------------------------------------------------
def info_(*args, **kwargs):
    s = ""
    for a in args:
        s += pstr(a, **kwargs)
    s = s.replace('\0', '')
    app.log.info(s)


# ----------------------------------------------------------------------------------------------------------------------
def debug_(*args, **kwargs):
    s = ""
    for a in args:
        s += pstr(a, **kwargs)
        s = s.replace('\0', '')
    app.log.debug(s)


# ----------------------------------------------------------------------------------------------------------------------
def error_(*args, **kwargs):
    s = ""
    for a in args:
        s += pstr(a, **kwargs)
        s = s.replace('\0', '')
    app.log.error(f"ERROR: {s}")


# ----------------------------------------------------------------------------------------------------------------------
def warning_(*args, **kwargs):
    s = ""
    for a in args:
        s += pstr(a, **kwargs)
        s = s.replace('\0', '')
    app.log.warning(f"WARNING: {s}")

__all__ += ['info_', 'debug_', 'error_', 'warning_', 'print_']


# check for sys.gui_splash flag which exist only if application is lauched
# in the GUI mode.
# (See comment on `sys.gui_splash` in launch_gui)
def _update(i, text):
    if hasattr(sys, 'gui_splash'):
        sys.gui_splash(i, text)
        time.sleep(.15)  # intentionally limit the speed.
    else:
        info_(text)

_update(1, 'Load API ...')

from spectrochempy.application import (
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
    available_styles,
)

general_preferences = app.general_preferences
project_preferences = app.project_preferences
description = app.description
long_description = app.long_description
config_manager = app.config_manager
config_dir = app.config_dir

# datadir = app.datadir

def set_loglevel(level=WARNING):
    general_preferences.log_level = level

def get_loglevel():
    return general_preferences.log_level

__all__ += [
    ### Helpers
    'DEBUG',
    'WARNING',
    'ERROR',
    'CRITICAL',
    'INFO',
    'project_preferences',
    'general_preferences',
    'config_manager',
    'config_dir',
    'available_styles',
    'set_loglevel',
    'get_loglevel',

    ### Info
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

# load the default style
# print("mpl_config_dir", mpl.get_configdir(), plt.style.available)
try:
    plt.style.use(app.project_preferences.style)
except:
    pass  # if the scp styles are not yet installed, ignore it

# IPython methods
# ----------------------------------------------------------------------------------------------------------------------
# we put them before so that we can eventually overwrite them

_update(3, 'IPython loading ...')

from IPython.core.display import *
from IPython.core import display

__all__.extend(display.__all__)

from IPython.lib.display import *
from IPython.lib import display

__all__.extend(display.__all__)

"""
This packages contains most of the core methods expose in the spectrochempy 
API.

"""
# constants
# ----------------------------------------------------------------------------------------------------------------------
from spectrochempy.utils import show, MASKED, NOMASK, EPSILON, INPLACE

__all__ += ['show', 'MASKED', 'NOMASK', 'EPSILON', 'INPLACE']

# optional libraries
# ----------------------------------------------------------------------------------------------------------------------

try:
    import sympy as sym
    HAS_SYMPY__ = True
    __all__.append('sym')
except ImportError:
    HAS_SYMPY = True

__all__.append('HAS_SYMPY')

try:
    import sklearn as skl
    
    HAS_SCIKITLEARN = True
    __all__.append('skl')
except ImportError:
    HAS_SCIKITLEARN = False

__all__.append('HAS_SCIKITLEARN')

try:
    import xarray as xr
    HAS_XARRAY = True
    __all__.append('xr')
except:
    HAS_XARRAY = False

__all__.append('HAS_XARRAY')

try:
    import pandas as pd
    HAS_PANDAS = True
    __all__.append('pd')
except:
    HAS_PANDAS = False

__all__.append('HAS_PANDAS')

# dataset
# ----------------------------------------------------------------------------------------------------------------------
_update(4, 'Load NDDataset ...')
from spectrochempy.core.dataset.api import *
from spectrochempy.core.dataset import api

__all__ += api.__all__

# plotters
# ----------------------------------------------------------------------------------------------------------------------
_update(5, 'Load plotters ...')
from spectrochempy.core.plotters.api import *
from spectrochempy.core.plotters import api

__all__ += api.__all__

# processors
# ----------------------------------------------------------------------------------------------------------------------
_update(6, 'Load processors ...')
from spectrochempy.core.processors.api import *
from spectrochempy.core.processors import api

__all__ += api.__all__

# readers
# ----------------------------------------------------------------------------------------------------------------------
_update(7, 'Load readers ...')
from spectrochempy.core.readers.api import *
from spectrochempy.core.readers import api

__all__ += api.__all__

# writers
# ----------------------------------------------------------------------------------------------------------------------
_update(8, 'Load writers ...')
from spectrochempy.core.writers.api import *
from spectrochempy.core.writers import api

__all__ += api.__all__

# units
# ----------------------------------------------------------------------------------------------------------------------
_update(9, 'Load units ...')
from spectrochempy.units.units import *
from spectrochempy.units import units

__all__ += units.__all__

# databases
# ----------------------------------------------------------------------------------------------------------------------
_update(10, 'Load database ...')
from spectrochempy.databases.api import *
from spectrochempy.databases import api

__all__ += api.__all__

# analysis
# ----------------------------------------------------------------------------------------------------------------------
_update(11, 'Load analysis ...')
from spectrochempy.core.analysis.api import *
from spectrochempy.core.analysis import api

__all__ += api.__all__

# fitting
# ----------------------------------------------------------------------------------------------------------------------
_update(12, 'Load fitting ...')
from spectrochempy.core.fitting.api import *
from spectrochempy.core.fitting import api

__all__ += api.__all__

# project
# ----------------------------------------------------------------------------------------------------------------------
_update(13, 'Load project ...')
from spectrochempy.core.project.api import *
from spectrochempy.core.project import api

__all__ += api.__all__

# script
# ----------------------------------------------------------------------------------------------------------------------
_update(14, 'Load scripts ...')
from spectrochempy.core.scripts.api import *
from spectrochempy.core.scripts import api

__all__ += api.__all__

# script
# ----------------------------------------------------------------------------------------------------------------------
_update(15, 'Load widgets ...')
from spectrochempy.widgets.api import *
from spectrochempy.widgets import api

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
_update(18, 'Start API ...')
_started = app.start()

_update(19, "API activated "
if _started else "API was not started!")

_update(20, '')  # make the last message visible (empirical solution...
# don't know why this work, at least on Mac OSX.


# ======================================================================================================================
if __name__ == '__main__':
    pass
