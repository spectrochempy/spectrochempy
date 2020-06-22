# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""
Package defining the *core* methods of the |scpy| API such as plotting,
processing, analysis, etc...

"""

# ----------------------------------------------------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------------------------------------------------

# warnings.simplefilter('ignore', (DeprecationWarning,
#                                 FutureWarning, UserWarning))

import os
import sys
import time
import warnings

# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------

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
    # Useful librairies alias for the end user avoiding to load them
    # --------------------------------------------------------------

    'np', 'plt', 'scipy', 'os', 'sys', 'mpl',

    # methods and objects from other packages will be added
    # later on this module (see below)

]

# ======================================================================================================================
# Progress bar
# ======================================================================================================================
if os.environ.get('USE_TQDM', 'Yes') == 'Yes' and "/bin/scpy" not in sys.argv[0]:      # deactivate for console scripts
    pbar = tqdm(total=1211)
    pbar.set_description('Loading SpectroChemPy API')
    val_tqdm = [1, 39, 52, 83, 83, 89, 92, 93, 94, 95, 96, 97, 98, 99, 100]

    def _pbar_update(i):
        if i == 'CLOSE':
            pbar.clear()
            pbar.close()
        else:
            time.sleep(.1)
            pbar.update(val_tqdm[i])
else:

    def _pbar_update(i):
        pass

# ======================================================================================================================
# loading module libraries
# here we also construct the __all__ list automatically
# ======================================================================================================================

_pbar_update(0)
from spectrochempy.application import SpectroChemPy   # noqa: E402

app = SpectroChemPy()
__all__ += ['app']

# ======================================================================================================================
# logging functions
# ======================================================================================================================

from spectrochempy.utils import pstr   # noqa: E402


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
    app.logs.debug(s)


# ----------------------------------------------------------------------------------------------------------------------
def error_(*args, **kwargs):
    s = ""
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

from spectrochempy.application import (          # noqa: E402
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
    # Helpers
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

# load the default style
# print("mpl_config_dir", mpl.get_configdir(), plt.style.available)
try:
    plt.style.use(project_preferences.style)
except Exception:
    pass  # if the scp styles are not yet installed, ignore it

# IPython methods
# ----------------------------------------------------------------------------------------------------------------------
# we put them before so that we can eventually overwrite them

_pbar_update(1)
#from IPython.core.display import *          # noqa: E402,F403,F401
#from IPython.core import display            # noqa: E402

#__all__.extend(display.__all__)

#from IPython.lib.display import *           # noqa: E402,F401,F403
#from IPython.lib import display             # noqa: E402

#__all__.extend(display.__all__)

"""
This packages contains most of the core methods expose in the spectrochempy API.

"""
# constants
# ----------------------------------------------------------------------------------------------------------------------
from spectrochempy.utils import show, MASKED, NOMASK, EPSILON, INPLACE  # noqa: E402

__all__ += ['show', 'MASKED', 'NOMASK', 'EPSILON', 'INPLACE']


# dataset
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update(2)
from spectrochempy.core.dataset.api import *     # noqa: E402,F403,F401
from spectrochempy.core.dataset import api       # noqa: E402

__all__ += api.__all__

# plotters
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update(3)
from spectrochempy.core.plotters.api import *      # noqa: E402,F403,F401
from spectrochempy.core.plotters import api        # noqa: E402

__all__ += api.__all__

# processors
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update(4)
from spectrochempy.core.processors.api import *     # noqa: E402,F403,F401
from spectrochempy.core.processors import api       # noqa: E402

__all__ += api.__all__

# readers
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update(5)
from spectrochempy.core.readers.api import *        # noqa: E402,F403,F401
from spectrochempy.core.readers import api          # noqa: E402

__all__ += api.__all__

# writers
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update(6)
from spectrochempy.core.writers.api import *         # noqa: E402,F403,F401
from spectrochempy.core.writers import api           # noqa: E402

__all__ += api.__all__

# units
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update(7)
from spectrochempy.units.units import *               # noqa: E402,F403,F401
from spectrochempy.units import units                 # noqa: E402

__all__ += units.__all__

# databases
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update(8)
from spectrochempy.databases.api import *              # noqa: E402,F403,F401
from spectrochempy.databases import api                # noqa: E402

__all__ += api.__all__

# analysis
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update(9)
from spectrochempy.core.analysis.api import *           # noqa: E402,F403,F401
from spectrochempy.core.analysis import api             # noqa: E402

__all__ += api.__all__

# fitting
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update(10)
from spectrochempy.core.fitting.api import *             # noqa: E402,F403,F401
from spectrochempy.core.fitting import api               # noqa: E402

__all__ += api.__all__

# project
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update(11)
from spectrochempy.core.project.api import *              # noqa: E402,F403,F401
from spectrochempy.core.project import api                # noqa: E402

__all__ += api.__all__

# script
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update(12)
from spectrochempy.core.scripts.api import *              # noqa: E402,F403,F401
from spectrochempy.core.scripts import api                # noqa: E402

__all__ += api.__all__

# widgets
# ----------------------------------------------------------------------------------------------------------------------
_pbar_update(13)
from spectrochempy.widgets.api import *                    # noqa: E402,F403,F401
from spectrochempy.widgets import api                      # noqa: E402

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

_pbar_update(14)

_pbar_update('CLOSE')

_started = app.start()

warnings.filterwarnings(action='ignore', module='matplotlib', category=UserWarning)

# ======================================================================================================================
if __name__ == '__main__':
    pass
