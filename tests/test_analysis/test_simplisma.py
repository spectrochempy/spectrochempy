# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa

import os

from spectrochempy.analysis.simplisma import SIMPLISMA
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import show


def test_simplisma():
    print("")
    data = NDDataset.read_matlab(
        os.path.join("matlabdata", "als2004dataset.MAT"), merge=False
    )
    print("Dataset (Jaumot et al., Chemometr. Intell. Lab. 76 (2005) 101-110)):")
    print("")
    for mat in data:
        print("    " + mat.name, str(mat.shape))

    ds = data[-1]
    assert ds.name == "m1"
    print("\n test simplisma on {}\n".format(ds.name))
    pure = SIMPLISMA(ds, n_pc=20, tol=0.2, noise=3, verbose=True)

    pure.C.T.plot()
    pure.St.plot()
    pure.plotmerit()
    assert "3     29      29.0     0.0072     0.9981" in pure.logs

    show()
