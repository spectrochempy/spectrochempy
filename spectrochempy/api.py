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

"""
Pseudo-package for all of the core symbols from SpectroChemPy.

Use this module for importing Spectrochempy names into your namespace.

Examples
---------

    >>> from spectrochempy.api import NDDataset
    >>> mynewdataset = NDDataset()   # create an empty dataset

"""
# python imports

import os
import sys
import logging
import warnings

warnings.simplefilter('ignore', (DeprecationWarning,
                                 FutureWarning, UserWarning))
from pytest import raises

# ==============================================================================
# Tells here the methods or object we allow to import from this library
# ==============================================================================
__all__ = ['raises',
           'APIref',
           ### imported library
           'np', 'plt', 'scipy'

           ### methods and objects from other packages will be added
           ### later on this module (see below)

           ]

# loading module libraries
# ------------------------
# here we also construct the __all__ list automatically

from spectrochempy.application import *

__all__ += [

    ### Helpers
    'log', 'log_level', 'DEBUG', 'WARN', 'ERROR', 'CRITICAL', 'INFO',
    'data', 'list_data',
    'plotoptions',
    'options',

    ### Info
    'copyright', 'version',
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

# Useful librairies alias for the end user avoiding to load them
# --------------------------------------------------------------
import numpy as np
import scipy

import matplotlib.pyplot as plt

# optional libraries
try:
    import sympy as sym

    __all__.append('sym')
except ImportError:
    pass

try:
    import sklearn as skl

    __all__.append('skl')
except ImportError:
    pass


def APIref():
    """
    Helper to display public objects and methods from the API

    """
    a = __all__[:]
    a = sorted(a)

    return a


APIref = APIref()

# START THE APPLICATION ========================================================
from spectrochempy.application import app

_started = app.start(debug=False, reset_config=True)

log.info("API activated "
         if _started else "API was not started!")

# ==============================================================================

if __name__ == '__main__':
    pass
