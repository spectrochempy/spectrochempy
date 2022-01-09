# -*- coding: utf-8 -*-
# flake8: noqa

import pathlib

import pytest

import spectrochempy as scp
from spectrochempy.utils import deprecated, ProtocolError, ignored


def test_protocolerror():

    # wrong protocol
    with pytest.raises(ProtocolError):
        _ = scp.read("wodger", protocol="xxx")


def test_deprecated():
    @deprecated("This func is deprecated")
    def deprecated_function():
        pass

    with pytest.warns(DeprecationWarning):
        deprecated_function()


def test_ignored():

    with ignored(FileNotFoundError):
        pathlib.Path("file-that-does-not-exist").unlink()
