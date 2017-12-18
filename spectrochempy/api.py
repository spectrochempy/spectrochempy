# -*- coding: utf-8 -*-
#
# ============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ============================================================================
# ============================================================================
# standard library import
# ============================================================================

import os
import sys
import logging
import warnings

warnings.simplefilter('ignore', (DeprecationWarning,
                                 FutureWarning, UserWarning))

# ============================================================================
# third party import
# ============================================================================

import numpy as np
from pytest import raises
import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt

# ============================================================================
# Tells here the methods or object we allow to import from this library
# ============================================================================

__all__ = [
           # Useful librairies alias for the end user avoiding to load them
           # --------------------------------------------------------------

           'np', 'plt', 'scipy', 'os', 'sys', 'mpl',

           # useful methods from external library
           # ------------------------------------
           'raises',

            ### methods and objects from other packages will be added
            ### later on this module (see below)

           ]

# ============================================================================
# loading module libraries
# ============================================================================

# here we also construct the __all__ list automatically

from spectrochempy.application import (app,
                                       __version__,
                                       __release__,
                                       __copyright__,
                                       __license__,
                                       __release_date__,
                                       __author__,
                                       __contributor__,
                                       __url__
                                       )
from spectrochempy import application

running = app.running

version = __version__
release = __release__
copyright =  __copyright__
license = __license__
release_date = __release_date__
authors = __author__
contributors = __contributor__
url = __url__

log =  app.log
log_level =  app.log_level

# give a user friendly name to the objects containing configurables preferences
preferences = app
scpdata =  app.general_preferences.data
list_scpdata =  app.general_preferences.list_scpdata
plotter_preferences =  app.plotter_preferences
project_preferences =  app.project_preferences


# Log levels
# -----------------------------------------------------------------------------
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

__all__ += application.__all__
__all__ += [

    ### Helpers
    'app',
    'log', 'log_level', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL', 'INFO',
    'scpdata', 'list_scpdata',
    'plotter_preferences', 'project_preferences', 'preferences',
    #'pcl',

    ### Info
    'copyright', 'version', 'release', 'license', 'url', 'release_date',
    'authors', 'contributors'
]


# dataset
# --------
from spectrochempy.dataset.api import *
from spectrochempy.dataset import api

__all__ += api.__all__

# plotters
# --------
from spectrochempy.plotters.api import *
from spectrochempy.plotters import api

__all__ += api.__all__

# processors
# ----------
from spectrochempy.processors.api import *
from spectrochempy.processors import api

__all__ += api.__all__

# readers
# -------
from spectrochempy.readers.api import *
from spectrochempy.readers import api

__all__ += api.__all__

# writers
# -------
from spectrochempy.writers.api import *
from spectrochempy.writers import api

__all__ += api.__all__

# units
# -----
from spectrochempy.units.units import *
from spectrochempy.units import units

__all__ += units.__all__


# databases
# ---------
from spectrochempy.databases.api import *
from spectrochempy.databases import api

__all__ += api.__all__

# analysis
# --------
from spectrochempy.analysis.api import *
from spectrochempy.analysis import api

__all__ += api.__all__

# fitting
# -------
from spectrochempy.fitting.api import *
from spectrochempy.fitting import api

__all__ += api.__all__

# project
# -------
from spectrochempy.projects.api import *
from spectrochempy.projects import api

__all__ += api.__all__

# script
# -------
from spectrochempy.scripts.api import *
from spectrochempy.scripts import api

__all__ += api.__all__

# optional libraries
# ------------------

try:
    import sympy as sym
    HAS_SYMPY = True
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

# IPython methods
from IPython.core.display import *
from IPython.core import display
__all__.extend(display.__all__)
from IPython.lib.display import *
from IPython.lib import display
__all__.extend(display.__all__)

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

# START THE APPLICATION ======================================================

_debug = False
_reset_config = False

_started = application.app.start(debug=_debug, reset_config=_reset_config)

# load the default style
plt.style.use(application.app.plotter_preferences.style)

log.info("API activated "
         if _started else "API was not started!")


# ============================================================================
if __name__ == '__main__':
    pass
