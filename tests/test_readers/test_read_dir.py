# -*- coding: utf-8 -*-
# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import os

from spectrochempy.core.dataset.nddataset import NDDataset


# comment the next line to test it manually
# @pytest.mark.skip('interactive so cannot be used with full testing')
def test_read_dir():
    A = NDDataset.read_dir(os.path.join('irdata', 'subdir'), recursive=True)
    assert len(A) == 9

    # in case we do not specify a directory:
    #  - open a dialog but handle the case we clik cancel
    NDDataset.read_dir()

    C = NDDataset.read_dir(os.path.join('matlabdata'))
    print('Matrixes in .mat files:')
    for x in C:
        print(C)
    assert len(C) == 6
    assert C[3].shape == (204, 96)
