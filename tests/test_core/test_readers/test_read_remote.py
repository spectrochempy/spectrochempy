# -*- coding: utf-8 -*-
# flake8: noqa


from pathlib import Path
import pytest

import spectrochempy as scp
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core import preferences as prefs

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
    nd2 = NDDataset.read_omnic(IRDATA / "nh4y-activation.spg")
    assert str(nd2) == "NDDataset: [float64] a.u. (shape: (y:55, x:5549))"
