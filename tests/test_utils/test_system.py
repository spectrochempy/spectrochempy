# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import sys

import pytest

from spectrochempy.utils.system import (
    get_node,
    get_user,
    get_user_and_node,
    is_kernel,
    sh,
)


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


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="does not run well on windows (seems to be linked to some commit message)",
)
def test_sh():
    res = sh.git("show", "HEAD")
    assert res is not None
