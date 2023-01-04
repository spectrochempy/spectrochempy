# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa

from traitlets import HasTraits, TraitError

from spectrochempy.core.dataset.coordrange import trim_ranges
from spectrochempy.utils import Range
from spectrochempy.utils.testing import raises

# ======================================================================================================================
# trim_ranges
# ======================================================================================================================


def test_trim_ranges():
    r = trim_ranges()
    assert r == []

    r = trim_ranges(3, 2)
    assert r[0] == [2, 3]

    r = trim_ranges((3, 2), (4.4, 10), (4, 5))
    assert r[-1] == [4, 10]
    assert r == [[2, 3], [4, 10]]

    r = trim_ranges((3, 2), (4.4, 10), (4, 5), reversed=True)
    assert r == [[10, 4], [3, 2]]


# ======================================================================================================================
# Range
# ======================================================================================================================


def test_range():
    class MyClass(HasTraits):
        r = Range()  # Initialized with some default values

    c = MyClass()
    c.r = [10, 5]
    assert c.r == [5, 10]
    with raises(TraitError):
        c.r = [10, 5, 1]
