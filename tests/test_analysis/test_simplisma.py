# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

import os

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.analysis.simplisma import SIMPLISMA


def test_SIMPLISMA():
    print('')
    data = NDDataset.read_matlab(os.path.join('matlabdata','als2004dataset.MAT'))
    print('Dataset (Jaumot et al., Chemometr. Intell. Lab. 76 (2005) 101-110)):')
    print('')
    for mat in data:
        print('    ' + mat.name, str(mat.shape))

    X = data[0]
    print('\n test simplisma on {}\n'.format(X.name))
    pure = SIMPLISMA(X, n_pc=20, tol=0.2,  noise=3, verbose=True)

    pure.C.T.plot()
    pure.St.plot()
    pure.plotmerit()
    assert '3     29      29.0     0.0072     0.9981' in pure.log