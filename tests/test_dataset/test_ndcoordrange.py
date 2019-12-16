# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT 
# See full LICENSE agreement in the root directory
# ======================================================================================================================


"""

"""

from copy import copy

import numpy as np
import pytest

from traitlets import TraitError, HasTraits
from spectrochempy.core.dataset.ndcoordrange import CoordRange, Range
from spectrochempy.units import ur, Quantity
from spectrochempy.core.dataset.ndarray import NDArray

from spectrochempy.utils.testing import (assert_array_equal,
                                         assert_equal_units, raises)
from spectrochempy.core import info_, print_


# ======================================================================================================================
# CoordRange
# ======================================================================================================================

def test_coordrange():
    r = CoordRange()
    assert r == []

    r = CoordRange(3, 2)
    assert r[0] == [2, 3]

    r = CoordRange((3, 2), (4.4, 10), (4, 5))
    assert r[-1] == [4, 10]
    assert r == [[2, 3], [4, 10]]

    r = CoordRange((3, 2), (4.4, 10), (4, 5),
                   reversed=True)
    assert r == [[10, 4], [3, 2]]


# ======================================================================================================================
# Range
# ======================================================================================================================

def test_range():

    class MyClass(HasTraits):
        r = Range()  # Initialized with some default values

    c = MyClass()
    c.r = [10, 5]
    assert c.r == [5,10]
    with raises(TraitError):
        c.r = [10, 5, 1]
