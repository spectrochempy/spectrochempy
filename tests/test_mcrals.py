# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

import os

from spectrochempy import NDDataset, MCRALS, EFA


def test_MCRALS():
    print('')
    data = NDDataset.read_matlab(os.path.join('matlabdata','als2004dataset.MAT'))
    print('Dataset (Jaumot et al., Chemometr. Intell. Lab. 76 (2005) 101-110)):')
    print('')
    for mat in data:
        print('    ' + mat.name, str(mat.shape))

    print('\n test on single experiment (m1) with given estimate of pure species (spure)...\n')

    X = data[0]
    guess = data[1]
    mcr  = MCRALS(X, guess, verbose=True)
    [C, St] = mcr.transform()


    C.T.plot()
    St.plot()
    mcr.plot()

    print('\n test on single experiment (m1) with EFA estimate of pure species (verbose off)...\n')
    guess = EFA(X).get_conc(4, plot=False)

    param = {'normSpec':'euclid', 'maxit':100}
    mcr2 = MCRALS(X, guess, param=param, verbose=False)
    mcr2.plot()

    assert 'converged !' in mcr2._log[-15:]



# =============================================================================
if __name__ == '__main__':
    pass
