# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa

import numpy as np
import pytest
import traitlets
from traitlets.config import Config

from spectrochempy.analysis.mcrals import MCRALS
from spectrochempy.core import set_loglevel
from spectrochempy.core.dataset.arraymixins.npy import dot
from spectrochempy.core.dataset.nddataset import Coord, NDDataset
from spectrochempy.utils import show, testing


def gaussian(x, h, c, w, noise):
    with testing.RandomSeedContext(12345):
        # seed the rndom generator to ensure the output is always the same for comparison
        return h * (np.exp(-1 / 2 * ((x.data - c) / w) ** 2)) + noise * np.random.randn(
            len(x)
        )  # a gaussian with added noise


def expon(t, c0, l, noise):
    with testing.RandomSeedContext(1234589):
        return c0 * (np.exp(l * t.data)) + noise * np.random.randn(len(t.data))


def get_C(x):
    return x


def get_C_a(C, a):
    return C / a


def get_C_kb(C, b=1):
    return C * b


def get_C_akb(C, a, b=1):
    return C * b / a


@pytest.fixture()
def model():
    class Model(object):
        def __init__(self):
            n_t = 10
            n_wl = 10
            h = [1, 1]
            c = [250, 750]
            w = [100, 100]
            noise_spec = [0.0, 0.0]
            noise_conc = [0.0, 0.0]

            c0 = [10, 1]
            l = np.array([-2, 2]) * 1e-2

            n_PS = 2  # number of pure species
            t_c = Coord(np.arange(0, 100, 100 / n_t), title="time", units="s")
            wl_c = Coord(
                np.arange(0, 1000, 1000 / n_wl), title="wavelength", units="nm"
            )
            PS_c = Coord(
                range(n_PS),
                title="species",
                units=None,
                labels=["PS#" + str(i) for i in range(n_PS)],
            )

            self.St0 = St0 = NDDataset.zeros((n_PS, len(wl_c)), coordset=(PS_c, wl_c))
            self.C0 = C0 = NDDataset.zeros((len(t_c), n_PS), coordset=(t_c, PS_c))

            self.St = St = NDDataset.zeros((n_PS, len(wl_c)), coordset=(PS_c, wl_c))
            self.C = C = NDDataset.zeros((len(t_c), n_PS), coordset=(t_c, PS_c))

            for i, id in enumerate((0, 1)):
                C.data[:, i] = expon(t_c, c0[id], l[id], noise_conc[id])
                St.data[i, :] = gaussian(wl_c, h[id], c[id], w[id], noise_spec[id])

                C0.data[:, i] = expon(t_c, c0[id], l[id], 0)
                St0.data[i, :] = gaussian(wl_c, h[id], c[id], w[id], 0)

    return Model()


@pytest.fixture
def data(model):
    D = dot(model.C, model.St)
    D.title = "intensity"
    return D


def test_MCRALS(model, data):
    # Test normal workflow
    D = data
    St0 = model.St0
    C0 = model.C0

    # Instanciate a MCRALS object, with some log_level
    # Note that the console log will never show debug
    # ( For this, look to attribute log or in the spectrochempy log file)

    mcr = MCRALS(log_level="INFO")

    # # set data (dataset X)
    # mcr.X = D
    #
    # # set or guess a profile (here concentrations C0)
    # mcr.set_profile(C0)

    # Now set or modify some configuration parameters
    mcr.tol = 30.0

    # if necessary get help
    print(mcr.help())

    # execute the main process
    mcr.fit(D, C0)

    # assert result
    # =============
    assert mcr.log.endswith("converged !")

    # test attributes
    for attr in [
        "log",
        "Chard",
        "Stsoft",
        "St",
        "C",
        "extOutput",
        "X",
    ]:
        assert hasattr(mcr, attr)

    # test current aprameters
    params = mcr.parameters()
    assert len(params) == 24
    assert np.all(params.closureTarget == [1] * 10)
    assert params.tol == 30.0

    # test display of default
    params = mcr.parameters(default=True)
    assert params.tol == 0.1

    # test plot
    mcr.plotmerit()
    show()

    # reset to default
    mcr.reset()
    assert mcr.tol == 0.1

    # test diverging
    mcr.monoIncConc = [0, 1]
    mcr.monoIncTol = 1.0
    mcr.unimodSpec = [0, 1]
    mcr.normSpec = "euclid"
    mcr.closureConc = [0, 1]
    mcr.closureMethod = "constantSum"
    mcr.maxdiv = 1

    mcr.fit()  # by default take the previously used X and C0 data

    assert mcr.log.endswith("Stop ALS optimization.")

    # guess = C0, hard modeling
    mcr.reset()  # we reset everything to default

    mcr.hardConc = [0, 1]
    mcr.getConc = get_C
    mcr.argsGetConc = ()
    mcr.kwargsGetConc = {}
    mcr.tol = 30.0
    mcr.fit()
    assert "converged !" in mcr.log[-15:]

    # using the full MCRALS constructor
    mcr = MCRALS(
        D,
        C0,
        # other parameters set to non-default values
        monoIncConc=[0],
        monoDecConc=[1],
        closureConc=[0, 1],
        normSpec="max",
        maxit=1,
    )
    set_loglevel("WARNING")
    mcr.fit()

    # guess = C0.data, test with other parameters
    mcr = MCRALS(
        D,
        C0.data,
        normSpec="euclid",
        closureConc=[0, 1],
        closureMethod="constantSum",
        maxit=1,
    )
    mcr.fit()
    assert "Convergence criterion ('tol')" in mcr.log[-100:]

    # guess = St as ndarray
    mcr = MCRALS(D, St0.data, tol=15.0)
    mcr.fit()
    assert "converged !" in mcr.log[-15:]


def test_MCRALS_errors(model, data):
    # Text exceptions
    D = data
    C = model.C
    mcr = MCRALS()

    # inexistant keyword parameters
    try:
        _ = MCRALS(maxit=25, inexistant=0, log_level="DEBUG")
    except KeyError as exc:
        assert "'inexistant' is not a valid" in exc.args[0]

    # guess with wrong size of dimensions
    try:
        _ = MCRALS(
            D,
            np.random.rand(11, 2),
        )
    except ValueError as e:
        assert "None of the dimensions of the given profile" in e.args[0]

    # guess with wrong nonnegConc parameter
    try:
        _ = MCRALS(D, C, nonnegConc=[2])
    except ValueError as e:
        assert "please check the" in e.args[0]
    try:
        _ = MCRALS(D, C, nonnegConc=[0, 1, 1])
    except ValueError as e:
        assert "please check the" in e.args[0]

    # guess = C, test with deprecated parameters
    # and few other parameters set to non-default values to improve coverage
    with testing.catch_warnings() as w:
        _ = MCRALS(unimodMod="strict")
    assert w[0].category == DeprecationWarning

    with testing.catch_warnings() as w:
        _ = MCRALS(unimodTol=1.0)
    assert w[0].category == DeprecationWarning

    with testing.catch_warnings() as w:
        _ = MCRALS(verbose=True)
    assert w[0].category == DeprecationWarning

    try:
        mcr.unimodSpec = "alls"
        raise ValueError  # should not arrive here
    except traitlets.TraitError:
        # expect a str = 'all' or a list
        pass

    try:
        mcr.nonnegConc = None
        raise ValueError  # should not arrive here
    except traitlets.TraitError:
        # expect a list or a str not None
        pass

    try:
        mcr.unimodConc = None
        raise ValueError  # should not arrive here
    except traitlets.TraitError:
        # expect a list or a str not None
        pass

    try:
        mcr.nonnegSpec = None
        raise ValueError  # should not arrive here
    except traitlets.TraitError:
        # expect a list or a str not None
        pass

    try:
        mcr.nonnegSpec = None
        raise ValueError  # should not arrive here
    except traitlets.TraitError:
        # expect a list or a str not None
        pass

    # guess with wrong unimodConc parameter
    try:
        _ = MCRALS(D, C, unimodConc=[2])
    except ValueError as e:
        assert "please check the" in e.args[0]
    try:
        _ = MCRALS(D, C, unimodConc=[0, 1, 1])
    except ValueError as e:
        assert "please check the" in e.args[0]

    # wrong closureTarget
    try:
        _ = MCRALS(D, C, closureTarget=[0, 1, 1])
    except ValueError as e:
        assert "please check the" in e.args[0]

    # wrong hardC_to_C_idx
    try:
        _ = MCRALS(D, C, hardC_to_C_idx=[2])
    except ValueError as e:
        assert "please check the" in e.args[0]
    try:
        _ = MCRALS(D, C, hardC_to_C_idx=[0, 1, 1])
    except ValueError as e:
        assert "please check the" in e.args[0]

    # wrong unimodSpec
    try:
        _ = MCRALS(D, C, unimodSpec=[2])
    except ValueError as e:
        assert "please check the" in e.args[0]
    try:
        _ = MCRALS(D, C, unimodSpec=[0, 1, 1])
    except ValueError as e:
        assert "please check the" in e.args[0]

    # wrong nonnegSpec
    try:
        _ = MCRALS(D, C, nonnegSpec=[2])
    except ValueError as e:
        assert "please check the" in e.args[0]
    try:
        _ = MCRALS(D, C, nonnegSpec=[0, 1, 1])
    except ValueError as e:
        assert "please check the" in e.args[0]

    # check same things but when setting the attributes
    mcr = MCRALS(D, C)

    try:
        mcr.nonnegSpec = [0, 1, 1]
    except ValueError as e:
        assert "please check the" in e.args[0]

    try:
        mcr.closureTarget = [0, 1, 1]
    except ValueError as e:
        assert "please check the" in e.args[0]
