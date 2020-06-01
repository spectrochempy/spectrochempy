# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

import os
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core import general_preferences as prefs
from spectrochempy.core import info_
from spectrochempy.utils.testing import assert_approx_equal
import pytest


# comment the next line to test it manually
# @pytest.mark.skip('interactive so cannot be used with full testing')
def test_read_without_filename():
    A = NDDataset.read_matlab()
    info_(A)


def test_read_with_filename():
    A = NDDataset.read_matlab(os.path.join('matlabdata', 'als2004dataset.MAT'))
    info_('Matrices in .mat file:')
    for x in A:
        info_('  ' + x.name + ': ' + str(x.shape))
    assert len(A) == 6
    assert A[3].shape == (204, 96)
