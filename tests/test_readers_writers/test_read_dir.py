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
    assert isinstance(C, list)
    assert len(C) == 6  # six matrices
    assert C[3].shape == (204, 96)

    A = scp.read_dir(directory='irdata/subdir')  # open a dialog to eventually select
    # directory inside the specified one
    assert A.shape == (4, 5549)

    B = scp.read_dir('irdata/subdir', recursive=True)
    assert len(B) == 8
    assert B.shape == (8, 5549)

    # no merging
    B = scp.read_dir(directory='irdata/subdir', recursive=True, merge=False)
    assert len(B) == 8
    assert isinstance(B, list)

    C = scp.read_dir()
    assert C == A


def test_read_dir_nmr():
    D = scp.read_dir('nmrdata/bruker/tests/nmr', protocol='topspin')
    assert isinstance(D, list)
    Dl = [item.name for item in D]
    assert 'topspin_2d expno:1 procno:1 (SER)' in Dl
    assert 'topspin_1d expno:1 procno:1 (FID)' in Dl


def test_read_dir_glob():
    pass

# EOF
