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
from spectrochempy.core.processors.align import align


# comment the next line to test it manually
#@pytest.mark.skip('interactive so cannot be used with full testing')

def test_readomnic_writejdx_readjdx():
    X = NDDataset.read_omnic(os.path.join('irdata','nh4y-activation.spg'))
    X.write_jdx('nh4y-activation.jdx')
    Y = NDDataset.read_jdx('nh4y-activation.jdx')
    os.remove('nh4y-activation.jdx')
    X, Y = align(X, Y)
    maxdiff = (X[:,1:-1] - Y[:,1:-1]).abs().max()
    assert maxdiff < 1e-8





