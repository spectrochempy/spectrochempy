# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
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
__all__ = [ ### Helpers
             'log', 'DEBUG','WARN', 'ERROR', 'CRITICAL', 'INFO', 'raises',
             'APIref', 'data_dir', 'list_data_dir',
             'options', 'plotoptions',

            ### Data
             'NDDataset', 'Axes', 'Axis', 'AxisRange', 'Meta',
             'ur', 'Quantity', 'Measurement', 'set_nmr_context',

            ### Info
             'copyright', 'release', 'version',

            ### Database
            'Isotopes',

            ### Analysis and Fit
             'Efa', 'Fit', 'Pca', 'Svd', 'align', 'autosub', 'basecor',
             'concatenate', 'interpolate', 'lsqnonneg', 'lstsq', 'mcrals', 'nnmf',

            ### View
             'plot_1D', 'plot_2D', 'contour',

            ### import/export
             'read_dso', 'read_jdx', 'read_omnic', 'read_bruker_nmr',
             'write_jdx',

            ### imported library
             'np', 'plt', 'scipy'

          ]



def APIref():
    """
    Helper to display public objects and methods from the API

    """
    return __all__


# ==============================================================================
# matplotlib use directive to set before calling matplotlib backends
# ==============================================================================
#from spectrochempy.application import scp
from spectrochempy.application import SpectroChemPy
scp = SpectroChemPy()
scp.initialize()

# ==============================================================================
# API namespace
# ==============================================================================

version = scp.version
release = scp.release
copyright = scp.copyright
log = scp.log

# give a user friendly name to the objects containing configurables options
plotoptions = scp.plotoptions
options = scp

_do_not_block = plotoptions.do_not_block

data_dir = scp.data_dir
list_data_dir = scp.list_data_dir

# log levels
# ----------
DEBUG = logging.DEBUG
INFO = logging.INFO
WARN = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# loading module libraries
# ------------------------
from spectrochempy.core.api import *
from spectrochempy.databases.api import *
from spectrochempy.analysis.api import *
from spectrochempy.fitting.api import *

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

try:
    import nmrglue as ng
    __all__.append('ng')
except ImportError:
    pass


# START THE APPLICATION ========================================================
_started = scp.start(debug=False)

scp.log.info("API activated "
             if _started else "API was not started!")

# ==============================================================================

if __name__ == '__main__':

   pass