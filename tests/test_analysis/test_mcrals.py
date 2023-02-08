# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa


import numpy as np

from spectrochempy import dot, set_loglevel
from spectrochempy.analysis.mcrals import MCRALS
from spectrochempy.core.dataset.nddataset import Coord, NDDataset
from spectrochempy.utils import show


def test_MCRALS():
    """Test MCRALS with synthetic data"""

    def gaussian(x, h, c, w, noise):
        return h * (np.exp(-1 / 2 * ((x.data - c) / w) ** 2)) + noise * np.random.randn(
            len(x)
        )  # a gaussian with added noise

    def expon(t, c0, l, noise):
        return c0 * (np.exp(l * t.data)) + noise * np.random.randn(len(t.data))

    def get_C(C):
        return C

    n_PS = 2  # number of pure species
    n_t = 10
    n_wl = 10
    h = [1, 1]
    c = [250, 750]
    w = [100, 100]
    noise_spec = [0.0, 0.0]
    noise_conc = [0.0, 0.0]

    c0 = [10, 1]
    l = np.array([-2, 2]) * 1e-2

    t_c = Coord(np.arange(0, 100, 100 / n_t), title="time", units="s")
    wl_c = Coord(np.arange(0, 1000, 1000 / n_wl), title="wavelength", units="nm")
    PS_c = Coord(
        range(n_PS),
        title="species",
        units=None,
        labels=["PS#" + str(i) for i in range(n_PS)],
    )

    St = NDDataset.zeros((n_PS, len(wl_c)), coordset=(PS_c, wl_c))
    C = NDDataset.zeros((len(t_c), n_PS), coordset=(t_c, PS_c))

    St0 = NDDataset.zeros((n_PS, len(wl_c)), coordset=(PS_c, wl_c))
    C0 = NDDataset.zeros((len(t_c), n_PS), coordset=(t_c, PS_c))

    for i, id in enumerate((0, 1)):
        C.data[:, i] = expon(t_c, c0[id], l[id], noise_conc[id])
        St.data[i, :] = gaussian(wl_c, h[id], c[id], w[id], noise_spec[id])

        C0.data[:, i] = expon(t_c, c0[id], l[id], 0)
        St0.data[i, :] = gaussian(wl_c, h[id], c[id], w[id], 0)

    D = dot(C, St)
    D.title = "intensity"

    #############################
    # Test normal functioning

    # guess = C0
    mcr = MCRALS(D, C0, tol=30.0)

    assert "converged !" in mcr.log[-15:]

    # test attributes
    for attr in [
        "log",
        "logs",
        "params",
        "Chard",
        "Stsoft",
        "St",
        "C",
        "extOutput",
        "X",
    ]:
        assert hasattr(mcr, attr)

    # test plot
    mcr.plotmerit()
    show()

    # test diverging
    mcr = MCRALS(
        D,
        C0,
        monoIncConc=[0, 1],
        monoIncTol=1.0,
        unimodSpec=[0, 1],
        normSpec="euclid",
        closureConc=[0, 1],
        closureMethod="constantSum",
        maxdiv=1,
    )
    assert "Stop ALS optimization" in mcr.log[-40:]

    # guess = C0, hard modeling
    mcr = MCRALS(D, C0, hardConc=[0, 1], getConc=get_C, argsGetConc=None, tol=30.0)
    assert "converged !" in mcr.log[-15:]

    # guess = C, test with deprecated parameters
    # and few other parameters set to non-default values to improve coverage
    mcr = MCRALS(
        D,
        C0,
        # deprecated:
        unimodMod="strict",
        unimodTol=1.1,
        verbose=True,
        # other parameters set to non-default values
        monoIncConc=[0],
        monoDecConc=[1],
        closureConc=[0, 1],
        nonnegConc=None,
        unimodConc=None,
        unimodSpec="all",
        nonnegSpec=None,
        normSpec="max",
        maxit=1,
    )
    set_loglevel("WARN")

    # guess = C0.data, test with other parameters
    mcr = MCRALS(
        D,
        C0.data,
        normSpec="euclid",
        closureConc=[0, 1],
        closureMethod="constantSum",
        maxit=1,
    )
    assert "Convergence criterion ('tol')" in mcr.log[-100:]

    # guess = St as ndarray
    mcr = MCRALS(D, St0.data, tol=15.0)
    assert "converged !" in mcr.log[-15:]

    #########################
    # Test exceptions

    # guess with wrong number of dimensions
    try:
        mcr = MCRALS(
            D,
            np.random.rand(n_t - 1, n_PS),
        )
    except ValueError as e:
        assert e.args[0] == "the dimensions of guess do not match the data"

    # guess with wrong nonnegConc parameter
    try:
        mcr = MCRALS(D, C, nonnegConc=[2])
    except ValueError as e:
        assert "please check nonnegConc" in e.args[0]
    try:
        mcr = MCRALS(D, C, nonnegConc=[0, 1, 1])
    except ValueError as e:
        assert "please check nonnegConc" in e.args[0]

    # guess with wrong unimodConc parameter
    try:
        mcr = MCRALS(D, C, unimodConc=[2])
    except ValueError as e:
        assert "please check unimodConc" in e.args[0]
    try:
        mcr = MCRALS(D, C, unimodConc=[0, 1, 1])
    except ValueError as e:
        assert "please check unimodConc" in e.args[0]

    # wrong closureTarget
    try:
        mcr = MCRALS(D, C, closureTarget=[0, 1, 1])
    except ValueError as e:
        assert "please check closureTarget" in e.args[0]

    # wrong hardC_to_C_idx
    try:
        mcr = MCRALS(D, C, hardC_to_C_idx=[2])
    except ValueError as e:
        assert "please check hardC_to_C_idx" in e.args[0]
    try:
        mcr = MCRALS(D, C, hardC_to_C_idx=[0, 1, 1])
    except ValueError as e:
        assert "please check hardC_to_C_idx" in e.args[0]

    # wrong unimodSpec
    try:
        mcr = MCRALS(D, C, unimodSpec=[2])
    except ValueError as e:
        assert "please check unimodSpec" in e.args[0]
    try:
        mcr = MCRALS(D, C, unimodSpec=[0, 1, 1])
    except ValueError as e:
        assert "please check unimodSpec" in e.args[0]

    # wrong nonnegSpec
    try:
        mcr = MCRALS(D, C, nonnegSpec=[2])
    except ValueError as e:
        assert "please check nonnegSpec" in e.args[0]
    try:
        mcr = MCRALS(D, C, nonnegSpec=[0, 1, 1])
    except ValueError as e:
        assert "please check nonnegSpec" in e.args[0]
