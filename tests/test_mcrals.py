# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

import os

from spectrochempy import NDDataset, MCRALS


def test_MCRALS():
    print('')
    data = NDDataset.read_matlab(os.path.join('matlabdata','als2004dataset.MAT'))
    print('Dataset (Jaumot et al., Chemometr. Intell. Lab. 76 (2005) 101-110)):')
    print('')
    for mat in data:
        print('    ' + mat.name, str(mat.shape))

    print('\n test on single experiment (m1) with estimate of pure species (spure)...')

    X = data[0]
    guess = data[1]

    mcr  = MCRALS(X, guess)
    [C, St] = mcr.transform()

    print(C)
    C.T.plot()
    print(St)
    St.plot()

    mcr.plot()

# =============================================================================
if __name__ == '__main__':
    pass
