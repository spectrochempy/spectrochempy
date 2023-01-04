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
from spectrochempy.core.dataset.nddataset import NDDataset

DATADIR = prefs.datadir
IRDATA = DATADIR / "irdata"


def test_read_remote():

    filename = IRDATA / "nh4y-activation.spg"

    # read normally
    nd1 = NDDataset.read_omnic(filename)
    assert str(nd1) == "NDDataset: [float64] a.u. (shape: (y:55, x:5549))"

    # move the files to simulate their absence:
    filesaved = filename.rename("~irdata_save")

    # now try to read
    nd2 = NDDataset.read_omnic("irdata/nh4y-activation.spg")
    assert str(nd2) == "NDDataset: [float64] a.u. (shape: (y:55, x:5549))"
