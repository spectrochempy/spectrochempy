# -*- coding: utf-8 -*-
# flake8: noqa

import os

import numpy as np

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.analysis.efa import EFA
from spectrochempy.core.analysis.mcrals import MCRALS
from spectrochempy.utils import show


def test_MCRALS_no_coords():
    print("")
    data = NDDataset.read_matlab(os.path.join("matlabdata", "als2004dataset.MAT"))
    print("Dataset (Jaumot et al., Chemometr. Intell. Lab. 76 (2005) 101-110)):")
    print("")
    for mat in data:
        print("    " + mat.name, str(mat.shape))

    print(
        "\n test on single experiment (m1) with given estimate of pure species (spure)...\n"
    )

    X = data[-1]
    assert X.name == "m1"
    guess = data[3]
    assert guess.name == "spure"
    mcr = MCRALS(X, guess, verbose=True)

    mcr.C.T.plot()
    mcr.St.plot()
    mcr.plotmerit()

    print(
        "\n test on single experiment (m1) with EFA estimate of pure species (verbose off)...\n"
    )
    guess = EFA(X).get_conc(4)

    param = {"normSpec": "euclid", "maxit": 100}
    mcr2 = MCRALS(X, guess, param=param, verbose=False)
    mcr2.plotmerit()

    assert "converged !" in mcr2.logs[-15:]

    show()


def test_MCRALS():
    data = NDDataset.read_matlab(
        os.path.join("matlabdata", "als2004dataset.MAT"), transposed=True
    )

    X = data[-1]
    assert X.name == "m1"
    X.set_coordset(y=np.arange(51), x=np.arange(96))
    X.title = "concentration"
    X.coordset.set_titles(y="spec coord.", x="elution time")
    X.plot(title="M1")

    guess = data[3]
    assert guess.name == "spure"  # spure
    guess.set_coordset(y=np.arange(4), x=np.arange(96))
    guess.title = "concentration"
    guess.coordset.set_titles(y="#components", x="elution time")
    guess.plot(title="spure")

    mcr = MCRALS(X, guess, verbose=True)

    mcr.C.T.plot(title="Concentration")

    mcr.St.plot(title="spectra")

    mcr.plotmerit()

    guess = EFA(X).get_conc(4)
    guess.plot(title="EFA guess")

    param = {"normSpec": "euclid", "maxit": 100}
    mcr2 = MCRALS(X, guess, param=param, verbose=False)
    mcr.plotmerit()

    assert "converged !" in mcr2.logs[-15:]

    show()


# =============================================================================
if __name__ == "__main__":
    pass
