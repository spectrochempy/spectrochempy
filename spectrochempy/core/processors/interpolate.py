# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""
This module defines functions related to interpolations

"""

__all__ = ['interpolate']

__dataset_methods__ = __all__

# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------

import scipy.interpolate
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# localimports
# ----------------------------------------------------------------------------------------------------------------------

from spectrochempy.utils import NOMASK, MASKED, warning_, error_, UnitsCompatibilityError
from spectrochempy.extern.orderedset import OrderedSet

# ............................................................................
def interpolate(dataset, axis=0, size=None):
    #TODO: a simple interpolator of the data (either to reduce
    #      or increase number of points in every dimension)
    raise NotImplementedError('Not yet implemented')

