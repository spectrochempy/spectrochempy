# -*- coding: utf-8 -*-
# flake8: noqa

import os

import numpy as np

from spectrochempy.core.dataset.nddataset import NDDataset, Coord
from spectrochempy.analysis.efa import EFA
from spectrochempy.analysis.mcrals import MCRALS
from spectrochempy.analysis.models import (
    lorentzianmodel,
    gaussianmodel,
    asymmetricvoigtmodel,
)
from spectrochempy.utils import show


def test_MCRALS_Jaumot():
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

    # test without coordinates
    mcr = MCRALS(X, guess, verbose=True)

    mcr.C.T.plot()
    mcr.St.plot()
    mcr.plotmerit()

    print(
        "\n test on single experiment (m1) with EFA estimate of pure species (verbose off)...\n"
    )
    guess = EFA(X).get_conc(4)

    mcr2 = MCRALS(X, guess, normSpec="max")
    mcr2.plotmerit()

    assert "converged !" in mcr2.logs[-15:]

    # the same with coordinates
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
    guess.T.plot(title="EFA guess")

    # few variations on options to improve coverage...
    mcr2 = MCRALS(X, guess, normSpec="euclid")
    assert "converged !" in mcr2.logs[-15:]

    mcr3 = MCRALS(X, guess, unimodMod="smooth", normSpec="max")
    assert "converged !" in mcr3.logs[-15:]

    mcr4 = MCRALS(X, guess, maxit=1)
    assert "Convergence criterion ('tol') not reached" in mcr4.logs[-100:]

    show()


def test_MCRALS_synth():
    """Test with synthetic data"""

    n_PS = 5  # number of pure species
    time_c = Coord(np.arange(0, 100, 1), title="time", units="s")
    pressure_c = Coord(np.arange(0, 1, 0.01), title="relative pressure")

    def langmuir(K, P):
        return K * P / (1 + K * P)

    y = Coord(np.arange(0, 100, 1), title="time", units="s")
    x = Coord(np.arange(0, 1000, 1), title="wavelength", units="nm")
    s = Coord(
        range(0, n_PS), title="PS#", labels=["PS#" + str(i) for i in range(0, n_PS)]
    )


# =============================================================================
if __name__ == "__main__":
    pass
