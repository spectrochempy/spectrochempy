# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa

import pytest

import spectrochempy as scp
from spectrochempy.core import preferences as prefs

DATADIR = prefs.datadir
IRDATA = DATADIR / "irdata"


def test_read_remote():

    filename = IRDATA / "nh4y-activation.spg"

    # read normally
    nd1 = scp.read_omnic(filename)
    assert str(nd1) == "NDDataset: [float64] a.u. (shape: (y:55, x:5549))"

    # delete file to simulate its absence:
    filename.unlink()

    # now try to download from github s not found locally (use _read_remote)
    nd2 = scp.read_omnic("irdata/nh4y-activation.spg")
    assert str(nd2) == "NDDataset: [float64] a.u. (shape: (y:55, x:5549))"

    # delete file to simulate its absence:
    filename.unlink()

    # now try a direct call to read_remote
    assert not filename.exists()
    nd2 = scp.read_remote("irdata/nh4y-activation.spg")
    assert str(nd2) == "NDDataset: [float64] a.u. (shape: (y:55, x:5549))"
    assert filename.exists()

    # now try to download from github s not found locally (use _read_remote)
    # but file doesn't exist on github
    with pytest.raises(FileNotFoundError):
        scp.read_omnic("irdata/nh4y-active.spg")

    # now try a direct call to read_remote
    with pytest.raises(FileNotFoundError):
        scp.read_remote("irdata/nh4y-active.spg")

    # delete file to simulate its absence:
    filename.unlink()

    # now try a using generic read
    assert not filename.exists()
    nd2 = scp.read("irdata/nh4y-activation.spg")
    assert str(nd2) == "NDDataset: [float64] a.u. (shape: (y:55, x:5549))"
    assert filename.exists()

    # delete file to simulate its absence:
    filename.unlink()

    # now try a using generic read
    with pytest.raises(FileNotFoundError):
        scp.read("irdata/nh4y-acti.spg")
