# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
from spectrochempy.utils.orderedset import OrderedSet


def test_orderedset():
    s = OrderedSet("abracadaba")
    t = OrderedSet("simsalabim")
    assert s | t == OrderedSet(["a", "b", "r", "c", "d", "s", "i", "m", "l"])
    assert s & t == OrderedSet(["a", "b"])
    assert s - t == OrderedSet(["r", "c", "d"])
