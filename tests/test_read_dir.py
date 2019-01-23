# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

import os
from spectrochempy.dataset.nddataset import NDDataset
from spectrochempy.application import general_preferences as prefs
from spectrochempy.utils import show
from spectrochempy.utils.testing import assert_approx_equal
import pytest

# comment the next line to test it manually
#@pytest.mark.skip('interactive so cannot be used with full testing')
def test_read_dir():
    A = NDDataset.read_dir(os.path.join('irdata','subdir'))
    assert len(A) == 4

    # in case we do not specify a directory:
    #  - open a dialog but handle the case we clik cancel
    B = NDDataset.read_dir()




