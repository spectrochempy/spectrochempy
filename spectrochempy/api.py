# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================

# ==============================================================================
# standard library import
# ==============================================================================

import os
import sys
import logging
import warnings

warnings.simplefilter('ignore', (DeprecationWarning,
                                 FutureWarning, UserWarning))

# ==============================================================================
# third party import
# ==============================================================================

import numpy as np
from pytest import raises
import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt

# ==============================================================================
# Tells here the methods or object we allow to import from this library
# ==============================================================================

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

# ==============================================================================
# loading module libraries
# ==============================================================================

# here we also construct the __all__ list automatically

from spectrochempy import application

running = application.app.running
version, dev_version, release = (application.app.version,
                                 application.app.dev_version,
                                 application.app.release)
copyright = application.app.copyright
log = application.app.log
log_level = application.app.log_level

# give a user friendly name to the objects containing configurables options
plotoptions = application.app.plotoptions
guioptions = application.app.guioptions
options = application.app
scpdata = application.app.scpdata
list_scpdata = application.app.list_scpdata

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
    'log', 'log_level', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL', 'INFO',
    'scpdata', 'list_scpdata',
    'plotoptions', 'options',
    #'pcl',

    ### Info
    'copyright', 'version', 'dev_version', 'release'
]


# core
# ------
from spectrochempy.core.api import *
from spectrochempy.core import api

__all__ += api.__all__

# databases
# ----------
from spectrochempy.databases.api import *
from spectrochempy.databases import api

__all__ += api.__all__

# analysis
# ---------
from spectrochempy.analysis.api import *
from spectrochempy.analysis import api

__all__ += api.__all__

# fitting
# --------
from spectrochempy.fitting.api import *
from spectrochempy.fitting import api

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

# START THE APPLICATION ========================================================
_started = application.app.start(debug=True, reset_config=True)

# load the default style
plt.style.use(application.app.plotoptions.style)

log.info("API activated "
         if _started else "API was not started!")


# ==============================================================================

if __name__ == '__main__':
    pass
