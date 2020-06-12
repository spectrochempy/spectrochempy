# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

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


def test_sh():
    res = sh.git('show', 'HEAD')
    assert res is not None
