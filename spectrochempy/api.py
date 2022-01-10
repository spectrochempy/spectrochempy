# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
Application Programming Interface.
"""

# During the initialization of this package, a `matplotlib` backend is set
# and some `IPython` configurations are made.


import sys

import matplotlib as mpl

from IPython.core.interactiveshell import InteractiveShell
from IPython import get_ipython

from pathlib import Path
from os import environ

# ------------------------------------------------------------------
# Check the environment for plotting
# ------------------------------------------------------------------

# Do we run in IPython ?
IN_IPYTHON = False
KERNEL = None
IP = None
if InteractiveShell.initialized():  # pragma: no cover
    IN_IPYTHON = True
    IP = get_ipython()
    KERNEL = getattr(IP, "kernel", None)

NO_DISPLAY = False
NO_DIALOG = False

# Are we buidings the docs ?
if Path(sys.argv[0]).name in ["make.py", "validate_docstrings.py"]:  # pragma: no cover
    # if we are building the documentation, in principle it should be done
    # using the make.py located at the root of the spectrochempy package.
    NO_DISPLAY = True
    NO_DIALOG = True
    mpl.use("agg", force=True)

# is there a --nodisplay flag
if "--nodisplay" in sys.argv:  # pragma: no cover
    NO_DISPLAY = True
    NO_DIALOG = True

# Are we running pytest?

if "pytest" in sys.argv[0] or "py.test" in sys.argv[0]:
    # if we are testing we also like a silent work with no figure popup!
    NO_DISPLAY = True
    NO_DIALOG = True

    # OK, but if we are doing individual function testing in PyCharm
    # it is interesting to see the plots and the file dialogs (except if we set explicitely --nodisplay argument!
    # if len(sys.argv) > 1 and not any([arg.endswith(".py") for arg in sys.argv[1:]]) and '--nodisplay' not in sys.argv:
    if (
        len(sys.argv) > 1
        and any((arg.split("::")[0].endswith(".py") for arg in sys.argv[1:]))
        and "--nodisplay" not in sys.argv
    ):  # pragma: no cover
        # individual module testing
        NO_DISPLAY = False
        NO_DIALOG = False

# Are we running in PyCharm scientific mode?
IN_PYCHARM_SCIMODE = mpl.get_backend() == "module://backend_interagg"

if (
    not (IN_IPYTHON and KERNEL) and not IN_PYCHARM_SCIMODE and not NO_DISPLAY
):  # pragma: no cover
    backend = mpl.rcParams["backend"]  # 'Qt5Agg'
    mpl.use(backend, force=True)

ALL = ["NO_DISPLAY", "NO_DIALOG"]

# ------------------------------------------------------------------
# Now we can start loading the API
# ------------------------------------------------------------------
# import the core api
from . import core
from .core import *  # noqa: F403, F401, E402

ALL += core.__all__

if not IN_IPYTHON:
    # needed in windows terminal - but must not be inited in Jupyter notebook
    from colorama import init as initcolor

    initcolor()

# def set_backend():

# workaround this problem https://github.com/jupyter/notebook/issues/3385
# ip.magic('matplotlib notebook')

if IN_IPYTHON and KERNEL and not NO_DISPLAY:  # pragma: no cover
    try:
        if (
            "ipykernel_launcher" in sys.argv[0]
            and "--InlineBackend.rc={'figure.dpi': 96}" in sys.argv
        ):
            # We are running from NBSphinx - the plot must be inline to show up.
            IP.magic("matplotlib inline")
        else:
            # Do not set the widget backend.... do not work most of the time after upbgrade of the various
            # library and
            # jupyter!!! ...
            IP.magic("matplotlib inline")  # widget
    except Exception:
        IP.magic("matplotlib qt")


# set_backend()

# a usefull utilities for dealing with path
from .utils import pathclean

DATADIR = pathclean(preferences.datadir)

__all__ = ["pathclean", "DATADIR"] + ALL

import warnings

warnings.filterwarnings(action="ignore", module="matplotlib")  # , category=UserWarning)
# warnings.filterwarnings(action="error", category=DeprecationWarning)

if NO_DISPLAY:
    mpl.use("agg", force=True)

    # set test file and folder in environment
    # set a test file in environment

    environ["TEST_FILE"] = str(DATADIR / "irdata" / "nh4y-activation.spg")
    environ["TEST_FOLDER"] = str(DATADIR / "irdata" / "subdir")
    environ["TEST_NMR_FOLDER"] = str(
        DATADIR / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_2d"
    )

# ==============================================================================
if __name__ == "__main__":
    pass
