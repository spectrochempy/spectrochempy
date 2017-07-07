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


"""

import os
import sys

import spectrochempy

# ==============================================================================
# Preferences
# ==============================================================================
#from spectrochempy.preferences import view_preferences, preferences

from spectrochempy.preferences.preference_manager_view import view_preferences
from spectrochempy.preferences.preference_manager import preference_manager as preferences

# ==============================================================================
# Test detection
# ==============================================================================
preferences.general._DO_NOT_BLOCK = False
for app in ['make.py','pytest', 'py.test', 'docrunner.py',]:
    if  app in sys.argv[0]:
        # this is necessary to buid doc with sphinx-gallery and doctests
        preferences.general._DO_NOT_BLOCK = True


# ==============================================================================
# Logger
# ==============================================================================
from spectrochempy.logger import *

import warnings
warnings.simplefilter('ignore', (DeprecationWarning,
                                 FutureWarning, UserWarning))

# ==============================================================================
# Graphics backend
# ==============================================================================
os.environ['ETS_TOOLKIT'] = 'qt5'

import matplotlib as mpl
if not 'sphinx-build' in sys.argv[0]:
    mpl.use('Qt5Agg')
else:
    # this is necessary to buid doc with sphinx-gallery
    log.info('Building docs')
    mpl.use('agg')
    preferences.general._DO_NOT_BLOCK = True

mpl.rcParams['backend.qt5'] = 'PyQt5'

from IPython.core.magic import UsageError
from IPython import get_ipython
from spectrochempy.utils import is_kernel

ip = get_ipython()
if ip is not None:

    if is_kernel():
        # if we are running this under ipython and jupyter notebooks
        try:
            ip.magic('matplotlib nbagg')
        except UsageError:
            try:
                ip.magic('matplotlib osx')
            except:
                ip.magic('matplotlib qt')
    else:
        try:
            ip.magic('matplotlib osx')
        except:
            ip.magic('matplotlib qt')


# ==============================================================================
# Matplotlib preamble for latex
# ==============================================================================
usetex = preferences.plot.use_latex

if usetex:
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble'] = [
            r'\usepackage{siunitx}',
            r'\sisetup{detect-all}',
            r'\usepackage{times}',     # set the normal font here
            r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
            r'\sansmath'
        ]
else:
    mpl.rc('text', usetex=False)
    mpl.rcParams['text.latex.preamble'] = []

# ==============================================================================
# PYTHONPATH
# ==============================================================================
# in case spectrochempy was not yet installed using setup
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
log.debug(sys.path)

# =============================================================================
# Load Spectrochempy API
# =============================================================================
log.debug('loading spectrochempy API.')

from spectrochempy.core import *

from spectrochempy.databases import *

from spectrochempy.analysis import *

from spectrochempy.fitting import *

# =============================================================================
# version
# =============================================================================

from spectrochempy.version import get_version

__version__, __release__ = get_version()
__copyright__ = u'2014-2017, LCS - ' \
                u'Laboratory for Catalysis and Spectrochempy'

info_string = u"""
SpectroChemPy's API
    Version   : {}
    Copyright : {}
""".format(__version__, __copyright__)

if preferences.general.print_info_on_loading and \
    not preferences.general._DO_NOT_BLOCK:
    print(info_string)
    log.debug("argv0 : ", sys.argv[0])


# =============================================================================
# additional useful librairies
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# optional libraries

try:
    import scipy
except:
    pass

try:
    import sympy
except:
    pass



# =============================================================================
if __name__ == '__main__':

    print(get_version())
