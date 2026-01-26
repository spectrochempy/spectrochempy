# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import pathlib

import pytest

import spectrochempy as scp
from spectrochempy.utils.decorators import deprecated
from spectrochempy.utils.exceptions import ProtocolError, ignored


def test_protocolerror():
    # wrong protocol
    with pytest.raises(ProtocolError):
        _ = scp.read("wodger", protocol="xxx")


def test_deprecated():
    @deprecated(replace="something")
    def deprecated_function():
        pass

    with pytest.warns(DeprecationWarning):
        deprecated_function()


def test_ignored():
    with ignored(FileNotFoundError):
        pathlib.Path("file-that-does-not-exist").unlink()
