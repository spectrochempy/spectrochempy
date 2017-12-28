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

import warnings

warnings.simplefilter('ignore', (DeprecationWarning,
                                 FutureWarning, UserWarning))

# ----------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------
import os
import sys

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
                                       plotter_preferences,
                                       processor_preferences,
                                       project_preferences,
                                       reader_preferences,
                                       writer_preferences,
                                       DEBUG,
                                       WARNING,
                                       ERROR,
                                       CRITICAL,
                                       INFO,
                                       ####
                                       app as APPLICATION,
                                       )

__all__ += [
    ### Helpers
    'log',
    'DEBUG',
    'WARNING',
    'ERROR',
    'CRITICAL',
    'INFO',
    'plotter_preferences',
    'project_preferences',
    'reader_preferences',
    'writer_preferences',
    'processor_preferences',
    'preferences',

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

# IPython methods
# ----------------------------------------------------------------------------
# we put them before so that we can eventually overwrite them

from IPython.core.display import *
from IPython.core import display
__all__.extend(display.__all__)

from IPython.lib.display import *
from IPython.lib import display
__all__.extend(display.__all__)

# constants
# ----------------------------------------------------------------------------
from spectrochempy.utils import  masked, nomask, EPSILON, INPLACE

__all__ +=  'masked', 'nomask', 'EPSILON', 'INPLACE'

# dataset
# --------
from spectrochempy.dataset.api import *
from spectrochempy.dataset import api

__all__ += api.__all__

# plotters
# --------
from spectrochempy.scp.plotters.api import *
from spectrochempy.scp.plotters import api

__all__ += api.__all__

# processors
# ----------
from spectrochempy.scp.processors.api import *
from spectrochempy.scp.processors import api

__all__ += api.__all__

# readers
# -------
from spectrochempy.scp.readers.api import *
from spectrochempy.scp.readers import api

__all__ += api.__all__

# writers
# -------
from spectrochempy.scp.writers.api import *
from spectrochempy.scp.writers import api

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
from spectrochempy.scp.analysis.api import *
from spectrochempy.scp.analysis import api

__all__ += api.__all__

# fitting
# -------
from spectrochempy.scp.fitting.api import *
from spectrochempy.scp.fitting import api

__all__ += api.__all__

# project
# -------
from spectrochempy.scp.projects.api import *
from spectrochempy.scp.projects import api

__all__ += api.__all__

# script
# -------
from spectrochempy.scp.scripts.api import *
from spectrochempy.scp.scripts import api

__all__ += api.__all__

# Core
# ----
from spectrochempy.scp.core import *
from spectrochempy.scp import core
__all__ += core.__all__


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

# START THE APPLICATION ======================================================

_debug = False
_reset_config = False


_started = APPLICATION.start(debug=_debug, reset_config=_reset_config)

# load the default style
plt.style.use(APPLICATION.plotter_preferences.style)

log.info("API activated "
         if _started else "API was not started!")


# ============================================================================
if __name__ == '__main__':
    pass
