# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

import os
import numpy as np

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.analysis.pca import PCA
from spectrochempy.core.analysis.mcrals import MCRALS
from spectrochempy.utils import show, info_

def test_MCRALS():

    data = NDDataset.read_matlab(os.path.join('matlabdata','als2004dataset.MAT'), transposed=True)
    info_('\nDataset (Jaumot et al., Chemometr. Intell. Lab. 76 (2005) 101-110)):\n')

    for i, mat in enumerate(data):
        info_('    ' + mat.name, str(mat.shape))

    info_('\n test on single experiment (m1) with estimate of pure species (spure)...\n')

    X = data[0]  # m1
    X.coords = [np.arange(51), np.arange(96)]
    X.title = 'intensity'
    X.coords.titles = ['concentration', 'retention time']
    info_(X)
    X.plot()

    guess = data[1] # cpure
    guess.coords = [np.arange(4), np.arange(96)]
    guess.title = 'intensity'
    guess.coords.titles = ['retention time', 'spectral component']
    guess.plot()

    mcr  = MCRALS(X, guess)
    [C, St] = mcr.transform()

    info_(C)
    C.T.plot()
    info_(St)
    St.plot()

    mcr.plot()

# ======================================================================================================================
if __name__ == '__main__':
    pass
