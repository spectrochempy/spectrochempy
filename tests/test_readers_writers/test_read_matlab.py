# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import os

import pytest

from spectrochempy.core.dataset.nddataset import NDDataset


# comment the next line to test it manually
@pytest.mark.skip('interactive so cannot be used with full testing')
def test_read_without_filename():
    NDDataset.read_matlab()


def test_read_with_filename():
    A = NDDataset.read_matlab(os.path.join('matlabdata', 'als2004dataset.MAT'))
    assert len(A) == 6
    assert A[3].shape == (204, 96)


def test_read_DSO():
    A = NDDataset.read_matlab(os.path.join('matlabdata', 'dso.mat'))
    assert A.name == "Group sust_base line withoutEQU.SPG"
    assert A.shape == (20, 426)
