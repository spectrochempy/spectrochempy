# -*- coding: utf-8 -*-
#
# ============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ============================================================================

"""
Package defining the *core* methods of the |scpy| API such as plotting,
processing, analysis, etc...

"""


# ----------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------

import warnings

warnings.simplefilter('ignore', (DeprecationWarning,
                                 FutureWarning, UserWarning))

import os
import sys
import time

# check for sys.gui_splash flag which exist only if application is lauched
# in the GUI mode.
# (See comment on `sys.gui_splash` in launch_gui)
def _update(i, text):
    if hasattr(sys, 'gui_splash'):
        sys.gui_splash(i, text)
        time.sleep(.15) # intentionally limit the speed.

# ----------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy


# ============================================================================
# Tells here the methods or object we allow to import from this library
# ============================================================================

__all__ = [
           # Useful librairies alias for the end user avoiding to load them
           # --------------------------------------------------------------

           'np', 'plt', 'scipy', 'os', 'sys', 'mpl',

            ### methods and objects from other packages will be added
            ### later on this module (see below)

           ]



# ============================================================================
# loading module libraries
# ============================================================================

# here we also construct the __all__ list automatically
_update(1, 'Start API ...')
from spectrochempy.application import (log,
                                       __version__ as version,
                                       __release__ as release,
                                       __copyright__ as copyright,
                                       __license__ as license,
                                       __release_date__ as release_date,
                                       __author__ as authors ,
                                       __contributor__ as contributors,
                                       __url__ as url,
                                       preferences,
                                       project_preferences,
                                       DEBUG,
                                       WARNING,
                                       ERROR,
                                       CRITICAL,
                                       INFO,
                                       ####
                                       app as APPLICATION,
                                       datadir,
                                       )

__all__ += [
    ### Helpers
    'log',
    'DEBUG',
    'WARNING',
    'ERROR',
    'CRITICAL',
    'INFO',
    'project_preferences',
    'preferences',
    'datadir',

    ### Info
    'copyright',
    'version',
    'release',
    'license',
    'url',
    'release_date',
    'authors',
    'contributors'
]

# START THE APPLICATION ======================================================
_started = APPLICATION.start()

# load the default style
# print("mpl_config_dir", mpl.get_configdir(), plt.style.available)
try:
    plt.style.use(APPLICATION.project_preferences.style)
except:
    pass # if the scp styles are not yet installed, ignore it

log.info("API activated "
         if _started else "API was not started!")
_update(2, 'API activated.')

# IPython methods
# ----------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------
from spectrochempy.utils import  show, masked, nomask, EPSILON, INPLACE

__all__ +=  'show', 'masked', 'nomask', 'EPSILON', 'INPLACE'

# dataset
# --------
_update(4, 'Load NDDataset ...')
from spectrochempy.dataset.api import *
from spectrochempy.dataset import api

__all__ += api.__all__

# plotters
# --------
_update(5, 'Load plotters ...')
from spectrochempy.core.plotters.api import *
from spectrochempy.core.plotters import api

__all__ += api.__all__

# processors
# ----------
_update(6, 'Load processors ...')
from spectrochempy.core.processors.api import *
from spectrochempy.core.processors import api

__all__ += api.__all__

# readers
# -------
_update(7, 'Load readers ...')
from spectrochempy.core.readers.api import *
from spectrochempy.core.readers import api

__all__ += api.__all__

# writers
# -------
_update(8, 'Load writers ...')
from spectrochempy.core.writers.api import *
from spectrochempy.core.writers import api

__all__ += api.__all__


# units
# -----
_update(9, 'Load units ...')
from spectrochempy.units.units import *
from spectrochempy.units import units

__all__ += units.__all__


# databases
# ---------
_update(10, 'Load database ...')
from spectrochempy.databases.api import *
from spectrochempy.databases import api

__all__ += api.__all__

# analysis
# --------
_update(11, 'Load analysis ...')
from .analysis.api import *
from .analysis import api

__all__ += api.__all__

# fitting
# -------
_update(12, 'Load fitting ...')
from spectrochempy.core.fitting.api import *
from spectrochempy.core.fitting import api

__all__ += api.__all__


# project
# -------
_update(13, 'Load projects ...')
from spectrochempy.core.projects.api import *
from spectrochempy.core.projects import api

__all__ += api.__all__

# script
# -------
_update(14, 'Load scripts ...')
from spectrochempy.core.scripts.api import *
from spectrochempy.core.scripts import api

__all__ += api.__all__


# optional libraries
# ------------------

try:
    import sympy as sym
    __HAS_SYMPY__ = True
    __all__.append('sym')
except ImportError:
    __HAS_SYMPY__ = True
__all__.append('__HAS_SYMPY__')

try:
    import sklearn as skl
    __HAS_SCIKITLEARN__ = True
    __all__.append('skl')
except ImportError:
    __HAS_SCIKITLEARN__ = False
__all__.append('__HAS_SCIKITLEARN__')


# Helpers
# -------

def APIref():
    """
    Helper to display public objects and methods from the API

    """
    a = __all__[:]
    a = sorted(a)
    return a

APIref = APIref()

__all__.append('APIref')

_update(15, 'API loaded and activated.')

_update(16, '')    # make the last message visible (empirical solution...
# don't know why this work, at least on Mac OSX.


# ============================================================================
if __name__ == '__main__':
    pass
