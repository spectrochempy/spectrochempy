# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa

import pathlib

import pytest

import spectrochempy as scp
from spectrochempy.utils.exceptions import ProtocolError, deprecated, ignored


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
