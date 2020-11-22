# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import os
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.processors.align import align


def test_read_write_jdx(IR_dataset_2D):
    X = IR_dataset_2D[:10]
    X.write_jdx('nh4y-activation.jdx')
    Y = NDDataset.read_jdx('nh4y-activation.jdx')
    os.remove('nh4y-activation.jdx')
    X1, Y1 = align(X, Y)
    maxdiff = (X1[:, 1:-1] - Y1[:, 1:-1]).abs().max()
    assert maxdiff.data < 1e-8
