# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import os
from spectrochempy.core.dataset.nddataset import NDDataset
import pytest


def test_read_with_filename():
    A = NDDataset.read_opus(os.path.join('irdata', 'OPUS', 'test.0000'))
    assert A[0, 2303.8694].data == pytest.approx(2.72740, 0.00001)
    B = NDDataset.read_opus(['test.0000', 'test.0001', 'test.0002'],
                            directory=os.path.join('irdata', 'OPUS', ))
    assert len(B) == 3
