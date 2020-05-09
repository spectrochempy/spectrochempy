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

from spectrochempy.utils import *
import os

def test_dict_compare():
    x = dict(a=1, b=2)
    y = dict(a=2, b=2)
    added, removed, modified, same = dict_compare(x, y, check_equal_only=False)
    assert modified == set('a')
    assert not dict_compare(x, y)

