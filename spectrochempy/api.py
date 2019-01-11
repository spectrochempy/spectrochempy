# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

# We intentionnaly DO NOT SET DOCSTRING, HERE... due to the way the documentation
# is builded.

# """
# Main package
#
# During the initialization of this package, a `matplotlib` backend is set
# and some `IPython` configurations are made.
#
#
# """

import sys
import os
import logging

from IPython.core.magic import UsageError
from IPython import get_ipython
import matplotlib as mpl

# ----------------------------------------------------------------------------
# Backend
# ----------------------------------------------------------------------------

# .........................................................................
def _setup_backend_and_ipython(backend=None):
    """Backend and IPython matplotlib environ setup

    Here, we performs this setup before any call to `matplotlib.pyplot`
    that are performed later in this application

    ..Note:: This method is called automatically at the initialization step
        if the application is called from the command line

    Parameters
    ----------
    backend : str, optional
        backend to use, default = ``Qt5Agg``.

    """

    # change backend here before the project module is imported
    #if backend == 'spectrochempy_gui':
    #    # this happen when the GUI is used
    #    backend = 'module://spectrochempy_gui.backend'

    # the current backend
    backend = mpl.get_backend()
    if backend == 'module://ipykernel.pylab.backend_inline'  or backend == \
          'MacOSX':
        # Force QT5
        backend = 'Qt5Agg'
        # mpl.rcParams['backend.qt5'] = 'PyQt5'  # <-- This is deprecated in
        # version 2.2 of matplotlib, but it seems, this doesn't make
        # difference for our notebooks

    # if we are building the docs, in principle it should be done using
    # the builddocs.py located in the scripts folder
    #print('sys.argv', sys.argv, backend)
    if not 'builddocs.py' in sys.argv[0] :
        mpl.use(backend, warn = False, force = True)
    else:
        # 'agg' backend is necessary to build docs with sphinx-gallery
        mpl.use('agg', warn = False, force = True)

    # use of IPython (console or notebook)
    ip = get_ipython()
    if ip is not None:
        if getattr(get_ipython(), 'kernel', None) is not None:
            # set the ipython matplotlib environments
            try:
               import ipympl
               ip.magic('matplotlib widget')
            except UsageError as e:
                try:
                    ip.magic('matplotlib qt5')
                except:
                    pass
        else:
            try:
                ip.magic('matplotlib qt5')
            except:
                 pass

    #print('backend', backend)
    return (ip, backend)

if not 'pytest' in sys.argv[0] or (len(sys.argv) > 1 and not sys.argv[1].endswith("spectrochempy")): # individual test):
    # and os.environ.get('PWD', None)!='/spectrochempy':
    _setup_backend_and_ipython()
else:
    mpl.use('agg', warn=False, force=True)

# import the core api
from .core import *
from spectrochempy import core

__all__ = core.__all__

# ==============================================================================
# For documentation
# ==============================================================================

if __name__ == '__main__':
    pass


