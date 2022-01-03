# -*- coding: utf-8 -*-

# ======================================================================================================================
# Copyright (©) 2015-2022 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# Authors:
# christian.fernandez@ensicaen.fr
# arnaud.travert@ensicaen.fr
#
# This software is a computer program whose purpose is to provide a framework
# for processing, analysing and modelling *Spectro*scopic
# data for *Chem*istry with *Py*thon (SpectroChemPy). It is is a cross
# platform software, running on Linux, Windows or OS X.
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL-B
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.
# ======================================================================================================================
"""
SpectroChemPy API.

SpectroChemPy is a framework for processing, analyzing and modeling Spectroscopic data for Chemistry with Python.
It is a cross platform software, running on Linux, Windows or OS X.
"""

# check for update
# import sys

# from time import time
# from pathlib import Path

__all__ = ["api"]

# import the main api

from spectrochempy import api
from spectrochempy.api import *  # noqa: F401

__all__ += api.__all__


# ==============================================================================
if __name__ == "__main__":
    pass
