# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

# """
# Main package
#
# During the initialization of this package, a `matplotlib` backend is set
# and some `IPython` configurations are made.
#
#
# """

import sys

import matplotlib as mpl

from IPython.core.interactiveshell import InteractiveShell
from IPython import get_ipython

__all__ = ['IN_IPYTHON', 'NO_DISPLAY', 'ip', 'kernel']

# ----------------------------------------------------------------------------------------------------------------------
# Check the environment for plotting
# ----------------------------------------------------------------------------------------------------------------------

# Do we run in IPython ?
IN_IPYTHON = False
kernel = None
ip = None
if InteractiveShell.initialized():
    IN_IPYTHON = True
    ip = get_ipython()
    kernel = getattr(ip, "kernel", None)

NO_DISPLAY = False
NO_DIALOG = False

# Are we buidings the docs ?
if 'make.py' in sys.argv[0]:
    # if we are building the documentation, in principle it should be done
    # using the make.py located at the root of the spectrchempy package.
    NO_DISPLAY = True
    NO_DIALOG = True
    mpl.use('agg', force=True)

# is there a --nodisplay flag
if '--nodisplay' in sys.argv:
    NO_DISPLAY = True
    NO_DIALOG = True
    mpl.use('agg', force=True)

# Are we running pytest?
if 'pytest' in sys.argv[0] or 'py.test' in sys.argv[0]:
    # if we are testing we also like a silent work with no figure popup!
    NO_DISPLAY = True
    NO_DIALOG = True

    # OK, but if we are doing individual function testing in PyCharm
    # it is interesting to see the plots and the file dialogs (except if we set explicitely --nodisplay argument!
    # if len(sys.argv) > 1 and not any([arg.endswith(".py") for arg in sys.argv[1:]]) and '--nodisplay' not in sys.argv:
    if len(sys.argv) > 1 and any(
            [arg.split('::')[0].endswith(".py") for arg in sys.argv[1:]]) and '--nodisplay' not in sys.argv:
        # individual module testing
        NO_DISPLAY = False
        NO_DIALOG = False

    if NO_DISPLAY:
        mpl.use('agg', force=True)

# Are we running in PyCharm scientific mode?
if mpl.get_backend() == 'module://backend_interagg':
    IN_PYCHARM_SCIMODE = True
else:
    IN_PYCHARM_SCIMODE = False

if not (IN_IPYTHON and kernel) and not IN_PYCHARM_SCIMODE and not NO_DISPLAY:
    backend = mpl.rcParams['backend']  # 'Qt5Agg'
    mpl.use(backend, force=True)

# ----------------------------------------------------------------------------------------------------------------------
# Now we can start loading the API
# ----------------------------------------------------------------------------------------------------------------------
# import the core api
from spectrochempy.core import *  # noqa: F403, F401, E402
from spectrochempy import core  # noqa: E402

__all__ += core.__all__

if not IN_IPYTHON:
    # needed in windows terminal - but must not be inited in Jupyter notebook
    from colorama import init as initcolor

    initcolor()

# def set_backend():

# workaround this problem https://github.com/jupyter/notebook/issues/3385
# ip.magic('matplotlib notebook')

if IN_IPYTHON and kernel and not NO_DISPLAY:
    try:
        if 'ipykernel_launcher' in sys.argv[0] and \
                "--InlineBackend.rc={'figure.dpi': 96}" in sys.argv:
            # We are running from NBSphinx - the plot must be inline to show up.
            ip.magic('matplotlib inline')
        else:
            # Do not set the widget backend.... do not work most of the time after upbgrade of the various
            # library and
            # jupyter!!! ...
            ip.magic('matplotlib inline')  # widget
    except Exception:
        ip.magic('matplotlib qt')


# set_backend()

# a usefull utilities for dealing with path
from spectrochempy.utils import pathclean

DATADIR = pathclean(preferences.datadir)

__all__ += ['pathclean', 'DATADIR']

import warnings

warnings.filterwarnings(action='ignore', module='matplotlib')  # , category=UserWarning)
# warnings.filterwarnings(action="error", category=DeprecationWarning)

# ==============================================================================
if __name__ == '__main__':
    pass
