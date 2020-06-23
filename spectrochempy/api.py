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
import warnings

import matplotlib as mpl
from IPython.core.interactiveshell import InteractiveShell
from IPython import get_ipython
from PyQt5 import QtWidgets

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

# Which backend to choose for matplotlib ?

# That is from the matplotlib doc:
# What is a backend?
# A lot of documentation on the website and in the mailing lists refers to
# the "backend" and many new users are confused by this term. matplotlib
# targets many different use cases and output formats. Some people use
# matplotlib interactively from the python shell and have plotting windows
# pop up when they type commands. Some people run Jupyter notebooks and draw
# inline plots for quick data analysis. Others embed matplotlib into graphical
# user interfaces like wxpython or pygtk to build rich applications. Some people
# use matplotlib in batch scripts to generate postscript images from numerical
# simulations, and still others run web application servers to dynamically
# serve up graphs.
#
# To support all of these use cases, matplotlib can target different outputs,
# and each of these capabilities is called a backend; the "frontend" is the user
# facing code, i.e., the plotting code, whereas the "backend" does all the hard
# work behind-the-scenes to make the figure. There are two types of backends:
# user interface backends (for use in pygtk, wxpython, tkinter, qt4, or macosx;
# also referred to as "interactive backends") and hardcopy backends to make
# image files (PNG, SVG, PDF, PS; also referred to as "non-interactive
# backends").

# There is different way to setup the backend externally: using environment
# variable or setting up in the matplotlibrc file. But when a script depends on
# a specific backend it is advised to use the use() function:

# import matplotlib
# matplotlib.use('PS')   # generate postscript output by default

# So what we will do in spectrochempy ?:

# For non interactive processs -> backend agg
# else TkAgg (which has no dependency - conversely to QT) except if PyQT5
# is already imported.
#
# if we are not running in a jupyter notebook or lab
#
# if we are in a notebook, we will encounter two situation (real interactive job)
# or execution o notebook in the background using ``nbsphinx``.
#
# if we are in the Scientific mode of PyCharm (module://backend_interagg)
#   -> keep it

NO_DISPLAY = False

# Are we buidings the docs ?
if 'make.py' in sys.argv[0]:
    # if we are building the documentation, in principle it should be done
    # using the make.py located at the root of the spectrchempy package.
    NO_DISPLAY = True
    mpl.use('agg', force=True)

# Are we running pytest ?
if 'pytest' in sys.argv[0] or 'py.test' in sys.argv[0]:
    # if we are testing we also like a silent work with no figure popup!
    NO_DISPLAY = True

    # OK, but if we are doing individual module or function testing in PyCharm
    # it is interesting to see the plots!
    if len(sys.argv) > 1 \
            and not any([argv.endswith("tests") for argv in sys.argv[1:]]) \
            and '--nodisplay' not in sys.argv:
        # individual module testing
        NO_DISPLAY = False

    if NO_DISPLAY:
        mpl.use('agg', force=True)

# Are we running in PyCharm scientific mode?
if mpl.get_backend() == 'module://backend_interagg':
    IN_PYCHARM_SCIMODE = True
else:
    IN_PYCHARM_SCIMODE = False

if not (IN_IPYTHON and kernel and not NO_DISPLAY) and not IN_PYCHARM_SCIMODE:
    backend = 'Qt5Agg'
    mpl.use('Qt5Agg', force=True)

# ----------------------------------------------------------------------------------------------------------------------
# Now we can start loading the API
# ----------------------------------------------------------------------------------------------------------------------

# import the core api
from spectrochempy.core import *         # noqa: F403, F401, E402
from spectrochempy import core           # noqa: E402

__all__ = core.__all__
__all__ += ['IN_IPYTHON', 'NO_DISPLAY', 'ip', 'kernel']

if not IN_IPYTHON:
    # needed in windows terminal - but must not be inited in Jupyter notebook
    from colorama import init as initcolor

    initcolor()

# GUI application needed for dialogs
GUI = QtWidgets.QApplication(sys.argv)

def set_backend():

    if IN_IPYTHON and kernel and not NO_DISPLAY:
        try:
            if 'ipykernel_launcher' in sys.argv[0] and \
                    "--InlineBackend.rc={'figure.dpi': 96}" in sys.argv:
                # We are running from NBSphinx - the plot must be inline to show up.
                ip.magic('matplotlib inline')
            else:
              pass #  ip.magic('matplotlib widget')
              # TODO: Why widget does not work!!!!
        except Exception:
            ip.magic('matplotlib qt')


set_backend()

__all__ += ['set_backend']


warnings.filterwarnings(action='ignore', module='matplotlib', category=UserWarning)
# warnings.filterwarnings(action="error", category=DeprecationWarning)

# ==============================================================================
if __name__ == '__main__':
    pass
