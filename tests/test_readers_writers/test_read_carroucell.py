# This is a sample Python script.

# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import os

from spectrochempy.core import info_
from spectrochempy.core.dataset.nddataset import NDDataset


# comment the next line to test it manually
# @pytest.mark.skip('interactive so cannot be used with full testing')
def test_read_carroucell_without_dirname():
    A = NDDataset.read_carroucell()
    info_(A)


def test_read_carroucell_with_dirname():
    A = NDDataset.read_carroucell(os.path.join('irdata', 'carroucell_samp'))
    info_(''
          'Datasets:')
    for x in A:
        info_('  ' + x.name + ': ' + str(x.shape))
    assert len(A) == 11
    assert A[3].shape == (6, 11098)

