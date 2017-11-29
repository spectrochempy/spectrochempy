# -*- coding: utf-8 -*-
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

"""During the initialization of this package, two operations are performed:

#. setup a gui `PyQt5.QApplication` needed for dialogs such as for
   opening/saving files.

#. setup a `matplotlib` backend and some `IPython` configurations.


"""
import sys
import warnings

from PyQt5.QtWidgets import QApplication

#: Handler to the GUI underlying application
guiApp = QApplication(sys.argv)

from IPython.core.magic import UsageError
from IPython import get_ipython
import matplotlib as mpl
from spectrochempy.utils import is_kernel

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
    else:
        # the current backend
        backend = mpl.get_backend()
        if backend == 'module://ipykernel.pylab.backend_inline' \
                or backend == 'MacOSX':
            # Force QT5
            backend = 'Qt5Agg'
            mpl.rcParams['backend.qt5'] = 'PyQt5'

    # if we are building the docs, in principle it should be done using
    # the builddocs.py located in the scripts folder
    if not 'builddocs.py' in sys.argv[0]:
        mpl.use(backend)
    else:
        # 'agg' backend is necessary to build docs with sphinx-gallery
        mpl.use('agg')

    ip = get_ipython()
    if ip is not None:
        if is_kernel():
            # set the ipython matplotlib environments
            try:
                import ipympl
                ip.magic('matplotlib notebook')
            except UsageError as e:
                try:
                    ip.magic('matplotlib osx')
                except:
                    try:
                        ip.magic('matplotlib qt5')
                    except:
                        pass
        else:
            try:
                ip.magic('matplotlib osx')  # on mac
            except:
                try:
                    ip.magic('matplotlib qt5')
                except:
                    pass

    return (ip, backend)

#: handler to the IPython instance
ip = None

#: Current backend
backend = "Qt5Agg"

# ==============================================================================
# For documentation
# ==============================================================================

if __name__ == '__main__':

    ip, backend = _setup_backend_and_ipython()

