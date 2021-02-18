#  -*- coding: utf-8 -*-
#
#  =====================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
#  =====================================================================================================================
#
import pytest
from spectrochempy.utils import get_user, get_user_and_node, get_node, is_kernel, sh


def test_get_user():
    res = get_user()
    assert res is not None


def test_get_node():
    res = get_node()
    assert res is not None


def test_get_user_and_node():
    res = get_user_and_node()
    assert res is not None


def test_is_kernel():
    res = is_kernel()
    assert not res


@pytest.mark.skip('problem with one of the commit - look at this later')
def test_sh():
    res = sh.git('show', 'HEAD')
    assert res is not None
