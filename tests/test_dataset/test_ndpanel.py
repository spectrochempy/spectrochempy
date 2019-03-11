# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================
"""


"""
import numpy as np
from copy import copy, deepcopy
from datetime import datetime
import pytest

from pint.errors import DimensionalityError

from spectrochempy.core.dataset.ndarray import NDArray
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.core.dataset.ndcoordset import CoordSet
from spectrochempy.core.dataset.ndpanel import NDPanel
from spectrochempy.core.dataset.nddataset import NDDataset

from spectrochempy.core import log
from spectrochempy.units import ur, Quantity
from spectrochempy.utils import (SpectroChemPyWarning, info_, debug_,
                                 INPLACE, MASKED,
                                 TYPE_INTEGER, TYPE_FLOAT)
from spectrochempy.utils.testing import (assert_equal, assert_array_equal,
                                         raises, catch_warnings,
                                         assert_approx_equal)



def test_ndpanel_init():
    
    panel = NDPanel()
    assert panel.datasets == {}
    assert panel.coords is None
    assert panel.meta is None
    assert panel.name == panel.id
    





# ============================================================================
if __name__ == '__main__':
    pass

# end of module