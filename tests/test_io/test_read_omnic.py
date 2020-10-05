# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import os

import spectrochempy as scp
from spectrochempy.core.dataset.nddataset import NDDataset

def test_read_omnic():
    A = NDDataset.read_omnic()  # should open a dialog
    assert A is None

    A = scp.read_omnic(os.path.join('irdata', 'nh4y-activation.spg'))
    assert A.filename == 'nh4y-activation'
    assert str(A)=='NDDataset: [float32] a.u. (shape: (y:55, x:5549))'

