# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to provide a general
# API for displaying, processing and analysing spectrochemical data.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
# =============================================================================



"""During the initialization of this package, a `matplotlib` backend is set
and some `IPython` configurations are made.


"""
import sys
import warnings

from IPython.core.magic import UsageError
from IPython import get_ipython
import matplotlib as mpl

from pkg_resources import get_distribution, DistributionNotFound
from setuptools_scm import get_version
import  subprocess
import datetime

__all__ = ['__version__', '__release__', '__release_date__', '__copyright__']

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
    if backend == 'spectrochempy_gui':
        # this happen when the GUI is used
        backend = 'module://spectrochempy_gui.backend'
    # the current backend
    backend = mpl.get_backend()
    if backend == 'module://ipykernel.pylab.backend_inline'  or backend == \
            'MacOSX':
        # Force QT5
        backend = 'Qt5Agg'
        mpl.rcParams['backend.qt5'] = 'PyQt5'

    # if we are building the docs, in principle it should be done using
    # the builddocs.py located in the scripts folder
    if not 'builddocs.py' in sys.argv[0]:
        mpl.use(backend, warn = False, force = True)
    else:
        # 'agg' backend is necessary to build docs with sphinx-gallery
        mpl.use('agg', warn = False, force = True)

    ip = get_ipython()
    if ip is not None:
        if getattr(get_ipython(), 'kernel', None) is not None:
            # set the ipython matplotlib environments
            try:
                import ipympl
                ip.magic('matplotlib notebook')
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

    return (ip, backend)

_setup_backend_and_ipython()

# ----------------------------------------------------------------------------
# Version
# ----------------------------------------------------------------------------

try:
    __release__ = get_distribution('spectrochempy').version.split('+')[0]


except DistributionNotFound:
    # package is not installed
    __release__ = '0.1'

__version__ = get_version(root='..', relative_to=__file__)

# ............................................................................
def _get_copyright():
    current_year = datetime.date.today().year
    copyright = '2014-{}'.format(current_year)
    copyright += ' - A.Travert and C.Fernandez @ LCS'
    return copyright

__copyright__ = _get_copyright()

# .............................................................................
def _get_release_date():
    try:
        return subprocess.getoutput(
            "git log -1 --tags --date='short' --format='%ad'")
    except:
        pass

__release_date__ = _get_release_date()

# ==============================================================================
# For documentation
# ==============================================================================

if __name__ == '__main__':
    pass





