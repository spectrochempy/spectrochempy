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
The Spectrochempy package.

The only things made here is to setup a gui PyQt5.QApplication

Attributes
==========
guiApp : :class:`~PyQt5.QtWidgets.QApplication`
    The main gui.

"""
import sys
import os

from PyQt5.QtWidgets import QApplication
guiApp = QApplication(sys.argv)

# ==============================================================================
# PYTHONPATH
# ==============================================================================
# in case spectrochempy was not yet installed using setup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# from spectrochempy.api import *


if __name__ == "__main__":

    from spectrochempy.api import scp
    import logging
    scp.start(
            reset_config=True,
            log_level = logging.INFO,
    )

    # ==============================================================================
    # Logger
    # ==============================================================================
    log = scp.log

    log.info('Name : %s ' % scp.name)

    scp.plotoptions.use_latex = True

    log.info(scp.plotoptions.latex_preamble)

