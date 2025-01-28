# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from os import environ

import numpy as np
import pytest
import traitlets as tr

import spectrochempy as scp
from spectrochempy.analysis.decomposition.mcrals import MCRALS
from spectrochempy.core import set_loglevel
from spectrochempy.core.dataset.nddataset import Coord, NDDataset
from spectrochempy.processing.transformation.npy import dot
from spectrochempy.utils import docstrings as chd
from spectrochempy.utils import testing
from spectrochempy.utils.plots import show


# test docstring
# but this is not intended to work with the debugger - use run instead of debug!
@pytest.mark.skipif(
    environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
    reason="debug mode cause error when checking docstrings",
)
def test_MCRALS_docstrings():
    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.analysis.decomposition.mcrals"
    chd.check_docstrings(
        module,
        obj=scp.MCRALS,
        # exclude some errors - remove whatever you want to check
        exclude=["SA01", "EX01", "ES01", "GL11", "GL08", "PR01", "PR06"],
    )


def gaussian(x, h, c, w, noise):
    with testing.RandomSeedContext(12345):
        # seed the rndom generator to ensure the output is always the same for comparison
        return h * (np.exp(-1 / 2 * ((x.data - c) / w) ** 2)) + noise * np.random.randn(
            len(x)
        )  # a gaussian with added noise


def expon(t, c0, l, noise):
    with testing.RandomSeedContext(1234589):
        return c0 * (np.exp(l * t.data)) + noise * np.random.randn(len(t.data))


def get_C(C):
    return C


def get_C_a(C, a):
    return C / a


def get_C_kb(C, b=1):
    return C * b


def get_C_akb(C, a, b=1):
    return C * b / a


def get_St(St):
    return St


@pytest.fixture()
def model():
    class Model(object):
        def __init__(self):
            n_t = 10
            n_wl = 100
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
                self.C.data[:, i] = expon(t_c, c0[id], l[id], noise_conc[id])
                self.St.data[i, :] = gaussian(wl_c, h[id], c[id], w[id], noise_spec[id])

                self.C0.data[:, i] = expon(t_c, c0[id], l[id], 0)
                self.St0.data[i, :] = gaussian(wl_c, h[id], c[id], w[id], 0)

    return Model()


@pytest.fixture
def data(model):
    D = dot(model.C, model.St)
    D.title = "intensity"
    D.units = "absorbance"
    D.set_coordset(None, None)
    D.y.title = "elution time"
    D.x.title = "wavelength"
    D.y.units = "hours"
    D.x.units = "cm^-1"
    return D


def test_MCRALS(model, data):
    # Test normal workflow
    D = data
    St0 = model.St0
    C0 = model.C0

    # Instantiate a MCRALS object, with some log_level
    # Note that the console log will never show debug
    # ( For this, look to attribute log or in the spectrochempy log file)

    mcr = MCRALS(log_level="INFO")

    # Now set or modify some configuration parameters
    mcr.tol = 30.0

    # execute the main process
    mcr.fit(D, C0)

    # assert result
    assert mcr.log.endswith("converged !")

    # transform
    assert mcr.transform(D) == mcr.C

    # test current parameters
    params = mcr.params()
    assert len(params) == 32
    assert np.all(params.closureTarget == [1] * 10)
    assert params.tol == 30.0

    # test display of default
    params = mcr.params(default=True)
    assert params.tol == 0.1

    # full process
    mcr.fit(D, C0)
    X_hat = mcr.inverse_transform()

    # assert iterations not stored (default)
    assert mcr.C_ls_list == []

    # test plot
    mcr.C.T.plot(title="Concentration")
    mcr.St.plot(title="Components")
    mcr.plotmerit(offset=0, nb_traces=5)
    show()

    # reset to default
    mcr.reset()
    assert mcr.tol == 0.1

    # test chaining fit runs
    mcr = MCRALS()
    mcr.fit(D, C0)
    mcr.storeIterations = True
    mcr.fit(D, (mcr.C, mcr.St))

    # assert iterations are stored
    assert mcr.C_ls_list != []

    mcr1 = MCRALS()
    mcr1.tol == 0.01
    mcr1.fit(D, C0)

    assert np.max(np.abs(mcr.C - mcr1.C)) < 1.0e-12
    assert np.max(np.abs(mcr.St - mcr1.St)) < 1.0e-12

    # test diverging
    mcr.monoIncConc = [0, 1]
    mcr.monoIncTol = 1.0
    mcr.unimodSpec = [0, 1]
    mcr.normSpec = "euclid"
    mcr.closureConc = [0, 1]
    mcr.closureMethod = "constantSum"
    mcr.maxdiv = 1

    mcr.fit(D, C0)

    assert mcr.log.endswith("Stop ALS optimization.")

    # guess = C0, hard modeling
    mcr.reset()  # we reset everything to default

    mcr.hardConc = [0, 1]
    mcr.getConc = get_C
    mcr.argsGetConc = ()
    mcr.kwargsGetConc = {}
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert "converged !" in mcr.log[-15:]

    # using the full MCRALS constructor
    mcr = MCRALS(
        monoIncConc=[0],
        monoDecConc=[1],
        closureConc=[0, 1],
        normSpec="max",
        max_iter=1,
    )
    set_loglevel("WARNING")
    mcr.fit(D, C0)

    # guess = C0, hard modeling of spectra
    mcr.reset()  # we reset everything to default
    mcr.hardSpec = [0, 1]
    mcr.getSpec = get_St
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert "converged !" in mcr.log[-15:]

    # guess = C0.data, test with other parameters
    mcr = MCRALS(
        normSpec="euclid",
        closureConc=[0, 1],
        closureMethod="constantSum",
        max_iter=1,
    )
    mcr.fit(D, C0.data)
    assert "Convergence criterion ('tol')" in mcr.log[-100:]

    # guess = St as ndarray
    mcr = MCRALS(tol=15.0)
    mcr.fit(D, St0.data)
    assert "converged !" in mcr.log[-15:]

    # solvers nnls
    mcr = MCRALS(tol=15.0, nonnegConc=[], solverConc="nnls", solverSpec="nnls")
    mcr.fit(D, St0.data)
    assert "converged !" in mcr.log[-15:]

    # solverConc pnnls
    mcr = MCRALS(
        tol=15.0, nonnegConc=[0], solverConc="pnnls", nonnegSpec=[0], solverSpec="pnnls"
    )
    mcr.fit(D, St0.data)
    assert "converged !" in mcr.log[-15:]

    # solverConc pnnls
    mcr = MCRALS(tol=15.0, solverConc="pnnls")
    mcr.fit(D, St0.data)
    assert "converged !" in mcr.log[-15:]

    # reconstruct
    Dh = mcr.inverse_transform()
    assert (Dh - D).abs().max() < 1.0e-12


def test_MCRALS_errors(model, data):
    # Text exceptions
    D = data
    C0 = model.C0
    mcr = MCRALS()

    # inexistant keyword parameters
    try:
        _ = MCRALS(max_iter=25, inexistant=0, log_level="DEBUG")
    except KeyError as exc:
        assert "'inexistant' is not a valid" in exc.args[0]

    # guess with wrong size of dimensions
    try:
        mcr.fit(
            D,
            np.random.rand(11, 2),
        )
    except ValueError as e:
        assert "None of the dimensions of the given profile" in e.args[0]

    # guess with wrong nonnegConc parameter
    mcr = MCRALS()
    mcr.fit(D, C0)

    with pytest.raises(ValueError) as e:
        mcr.nonnegConc = [2]
    assert "please check the" in e.value.args[0]

    with pytest.raises(ValueError) as e:
        mcr.nonnegConc = [0, 1, 1]
    assert "please check the" in e.value.args[0]

    # guess = C, test with deprecated parameters
    # and few other parameters set to non-default values to improve coverage
    with testing.catch_warnings(DeprecationWarning) as w:
        MCRALS(unimodMod="strict")
    print(w)
    assert w[0].category == DeprecationWarning

    with testing.catch_warnings(DeprecationWarning) as w:
        MCRALS(unimodTol=1.0)
    assert w[0].category == DeprecationWarning

    with testing.catch_warnings(DeprecationWarning) as w:
        MCRALS(verbose=True)
    assert w[0].category == DeprecationWarning

    with pytest.raises(tr.TraitError):
        mcr.unimodSpec = "alls"

    with pytest.raises(tr.TraitError):
        mcr.unimodConc = None

    with pytest.raises(tr.TraitError):
        mcr.nonnegSpec = None

    with pytest.raises(tr.TraitError):
        mcr.nonnegSpec = None

    # guess with wrong unimodConc parameter
    with pytest.raises(ValueError) as e:
        mcr.unimodConc = [2]
    assert "please check the" in e.value.args[0]

    with pytest.raises(ValueError) as e:
        mcr.unimodConc = [0, 1, 1]
    assert "please check the" in e.value.args[0]

    # wrong closureTarget
    with pytest.raises(ValueError) as e:
        mcr.closureTarget = [0, 1, 1]
    assert "please check the" in e.value.args[0]

    with pytest.raises(ValueError) as e:
        mcr.getC_to_C_idx = [0, 1, 1]
    assert "please check the" in e.value.args[0]

    # wrong unimodSpec
    with pytest.raises(ValueError) as e:
        mcr.unimodSpec = [2]
    assert "please check the" in e.value.args[0]

    with pytest.raises(ValueError) as e:
        mcr.unimodSpec = [0, 1, 1]
    assert "please check the" in e.value.args[0]

    # wrong nonnegSpec
    with pytest.raises(ValueError) as e:
        mcr.nonnegSpec = [2]
    assert "please check the" in e.value.args[0]

    with pytest.raises(ValueError) as e:
        mcr.nonnegSpec = [0, 1, 1]
    assert "please check the" in e.value.args[0]
