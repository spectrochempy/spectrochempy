# -*- coding: utf-8 -*-
#
# ============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL FREE SOFTWARE LICENSE AGREEMENT (Version B)
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

from spectrochempy.application import app
from spectrochempy import application

running = app.running
version, dev_version, release = (app.version,
                                 app.dev_version,
                                 app.release)
copyright =  app.copyright
log =  app.log
log_level =  app.log_level

# give a user friendly name to the objects containing configurables options
options = app
scpdata =  app.scpdata
list_scpdata =  app.list_scpdata
plotoptions =  app.plotoptions
projectsoptions =  app.projectsoptions


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
    'plotoptions', 'options',
    #'pcl',

    ### Info
    'copyright', 'version', 'dev_version', 'release'
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
plt.style.use(application.app.plotoptions.style)

log.info("API activated "
         if _started else "API was not started!")


# ============================================================================
if __name__ == '__main__':
    pass
