# -*- coding: utf-8 -*-
#

from spectrochempy.utils import dict_compare


def test_dict_compare():
    x = dict(a=1, b=2)
    y = dict(a=2, b=2)
    added, removed, modified, same = dict_compare(x, y, check_equal_only=False)
    assert modified == set("a")
    assert not dict_compare(x, y)
