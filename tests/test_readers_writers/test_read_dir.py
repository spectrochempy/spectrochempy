# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import spectrochempy as scp


def test_read_dir():
    A = scp.read_dir('irdata/subdir')
    assert A.shape == (4, 5549)

    C = scp.NDDataset.read_dir('matlabdata')
    print('Matrixes in .mat files:')
    for x in C:
        print(C)
    assert len(C) == 6
    assert C[3].shape == (204, 96)

    A = scp.read_dir(directory='irdata/subdir')  # should open a dialog
    assert A.shape == (4, 5549)

    B = scp.read_dir('irdata/subdir', recursive=True)
    assert len(B) == 7
    assert B[0].shape == (8, 5549)

    B = scp.read_dir(directory='irdata/subdir', recursive=True)
    assert len(B) == 7
    assert B[0].shape == (8, 5549)

    C = scp.read_dir()

    return
