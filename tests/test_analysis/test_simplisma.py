# -*- coding: utf-8 -*-
# flake8: noqa


import os

from spectrochempy.utils import show
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.analysis.simplisma import SIMPLISMA


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
