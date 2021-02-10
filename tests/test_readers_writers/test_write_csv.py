# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

from pathlib import Path

import spectrochempy as scp
from spectrochempy.core import preferences as prefs
from spectrochempy.core.dataset.nddataset import NDDataset


def test_write_csv():
    datadir = prefs.datadir
    ds = scp.NDDataset([1,2,3])
    f = ds.write_csv('myfile')

    assert f.name == 'myfile.csv'

    f.unlink()