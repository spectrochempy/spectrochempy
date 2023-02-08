# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa


import numpy as np
import traitlets

from spectrochempy import dot, set_loglevel
from spectrochempy.analysis.mcrals import MCRALS
from spectrochempy.core.dataset.nddataset import Coord, NDDataset
from spectrochempy.utils import show, testing


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
    # Test normal workflow

    # Instanciate a MCRALS object, with some log_level
    # Note that the console log will never show debug
    # ( For this look to attribute log or in the spectrochempy log file)
    try:
        mcr = MCRALS(maxit=25, inexistant=0, log_level="DEBUG")
    except KeyError as exc:
        assert "'inexistant' is not a valid" in exc.args[0]

    mcr = MCRALS(log_level="INFO")

    # set data (dataset X)
    mcr.X = D

    # set or guess a profile (here concentrations C0)
    mcr.set_profile(C0)

    # Now set or modify some configuration parameters
    mcr.tol = 30.0

    # Check the current configuration parameters using

    # execute the main process
    mcr.run()

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

    params = mcr.parameters()
    assert len(params) == 24
    assert np.all(params.closureTarget == [1] * 10)

    # test plot
    mcr.plotmerit()
    show()

    # test diverging
    # we reuse the same initialised MCRALS object as only some parameters changes
    mcr.tol = 0.1
    mcr.monoIncConc = [0, 1]
    mcr.monoIncTol = 1.0
    mcr.unimodSpec = [0, 1]
    mcr.normSpec = "euclid"
    mcr.closureConc = [0, 1]
    mcr.closureMethod = "constantSum"
    mcr.maxdiv = 1

    mcr.run()

    assert mcr.log.endswith("Stop ALS optimization.")

    # guess = C0, hard modeling
    mcr = MCRALS(
        D, C0
    )  # we reinit with a new MCRALS object as we want to come back to the default configuration
    # TODO: use config to reset and uplaod previous configuration
    mcr.hardConc = [0, 1]
    mcr.getConc = get_C
    mcr.argsGetConc = ()
    mcr.kwargsGetConc = {}
    mcr.tol = 30.0
    mcr.run()
    assert "converged !" in mcr.log[-15:]

    # guess = C, test with deprecated parameters
    # and few other parameters set to non-default values to improve coverage
    with testing.catch_warnings() as w:
        _ = MCRALS(unimodMod="strict")
    assert w[-1].category == DeprecationWarning

    with testing.catch_warnings() as w:
        _ = MCRALS(unimodTol=1.0)
    assert w[-1].category == DeprecationWarning

    with testing.catch_warnings() as w:
        _ = MCRALS(verbose=True)
    assert w[-1].category == DeprecationWarning

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
    mcr.run()
    assert "Convergence criterion ('tol')" in mcr.log[-100:]

    # guess = St as ndarray
    mcr = MCRALS(D, St0.data, tol=15.0)
    mcr.run()
    assert "converged !" in mcr.log[-15:]

    #########################
    # Text exceptions

    # guess with wrong number of dimensions
    try:
        _ = MCRALS(
            D,
            np.random.rand(n_t - 1, n_PS),
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
