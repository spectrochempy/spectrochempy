# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
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
from spectrochempy.application.application import set_loglevel
from spectrochempy.core.dataset.nddataset import Coord, NDDataset
from spectrochempy.processing.transformation.npy import dot
from spectrochempy.utils import docutils as chd
from spectrochempy.utils import testing


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


def _assert_mcrals_shapes(mcr, data, n_components=2):
    assert mcr.C.shape == (data.shape[0], n_components)
    assert mcr.St.shape == (n_components, data.shape[1])


def _assert_reconstructs(mcr, data, tol=1.0e-12):
    reconstructed = mcr.inverse_transform()
    assert reconstructed.shape == data.shape
    assert (reconstructed - data).abs().max() < tol


def test_mcrals_basic_fit_converges(model, data):
    D = data
    C0 = model.C0

    mcr = MCRALS(log_level="INFO")
    mcr.tol = 30.0
    result = mcr.fit(D, C0)

    assert result is mcr
    assert mcr.log.endswith("converged !")
    _assert_mcrals_shapes(mcr, D)
    assert mcr.transform(D) == mcr.C

    params = mcr.params()
    assert len(params) == 32
    assert np.all(params.closureTarget == [1] * 10)
    assert params.tol == 30.0

    params = mcr.params(default=True)
    assert params.tol == 0.1
    assert mcr.C_ls_list == []
    _assert_reconstructs(mcr, D)


def test_mcrals_iteration_storage(model, data):
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.fit(D, C0)
    mcr.storeIterations = True
    mcr.fit(D, (mcr.C, mcr.St))
    assert mcr.C_ls_list != []

    mcr1 = MCRALS()
    mcr1.fit(D, C0)
    assert np.max(np.abs(mcr.C - mcr1.C)) < 1.0e-12
    assert np.max(np.abs(mcr.St - mcr1.St)) < 1.0e-12


def test_mcrals_diverging_path_stops_cleanly(model, data):
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.fit(D, C0)
    mcr.monoIncConc = [0, 1]
    mcr.monoIncTol = 1.0
    mcr.unimodSpec = [0, 1]
    mcr.normSpec = "euclid"
    mcr.closureConc = [0, 1]
    mcr.closureMethod = "constantSum"
    mcr.maxdiv = 1

    mcr.fit(D, C0)
    assert mcr.log.endswith("Stop ALS optimization.")


def test_mcrals_closure_all_regression_issue_911(model, data):
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.closureConc = "all"
    mcr.closureMethod = "constantSum"
    mcr.maxdiv = 1

    mcr.fit(D, C0)
    assert mcr.log.endswith("Stop ALS optimization.")


def test_mcrals_hard_concentration_model_smoke(model, data):
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardConc = [0, 1]
    mcr.getConc = get_C
    mcr.argsGetConc = ()
    mcr.kwargsGetConc = {}
    mcr.tol = 30.0

    mcr.fit(D, C0)
    assert "converged !" in mcr.log[-15:]
    _assert_mcrals_shapes(mcr, D)


def test_mcrals_constructor_constraints_smoke(model, data):
    D = data
    C0 = model.C0

    mcr = MCRALS(
        monoIncConc=[0],
        monoDecConc=[1],
        closureConc=[0, 1],
        normSpec="max",
        max_iter=1,
    )
    set_loglevel("WARNING")
    mcr.fit(D, C0)
    _assert_mcrals_shapes(mcr, D)


def test_mcrals_hard_spectral_model_smoke(model, data):
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardSpec = [0, 1]
    mcr.getSpec = get_St
    mcr.tol = 30.0

    mcr.fit(D, C0)
    assert "converged !" in mcr.log[-15:]
    _assert_mcrals_shapes(mcr, D)


def test_mcrals_closure_with_array_guess_smoke(model, data):
    D = data
    C0 = model.C0

    mcr = MCRALS(
        normSpec="euclid",
        closureConc=[0, 1],
        closureMethod="constantSum",
        max_iter=1,
    )
    mcr.fit(D, C0.data)
    assert "Convergence criterion ('tol')" in mcr.log[-100:]
    _assert_mcrals_shapes(mcr, D)


def test_mcrals_spectral_guess_converges(model, data):
    D = data
    St0 = model.St0

    mcr = MCRALS(tol=15.0)
    mcr.fit(D, St0.data)
    assert "converged !" in mcr.log[-15:]
    _assert_mcrals_shapes(mcr, D)


def test_mcrals_nnls_solver_smoke(model, data):
    D = data
    St0 = model.St0

    mcr = MCRALS(tol=15.0, nonnegConc=[], solverConc="nnls", solverSpec="nnls")
    mcr.fit(D, St0.data)

    assert "converged !" in mcr.log[-15:]
    _assert_mcrals_shapes(mcr, D)
    assert mcr.C.min() >= -1.0e-12
    assert mcr.St.min() >= -1.0e-12


def test_mcrals_pnnls_solver_smoke(model, data):
    D = data
    St0 = model.St0

    mcr = MCRALS(
        tol=15.0, nonnegConc=[0], solverConc="pnnls", nonnegSpec=[0], solverSpec="pnnls"
    )
    mcr.fit(D, St0.data)

    assert "converged !" in mcr.log[-15:]
    _assert_mcrals_shapes(mcr, D)
    assert mcr.C[:, 0].min() >= -1.0e-12
    assert mcr.St[0].min() >= -1.0e-12


def test_mcrals_pnnls_concentration_solver_reconstructs(model, data):
    D = data
    St0 = model.St0

    mcr = MCRALS(tol=15.0, solverConc="pnnls")
    mcr.fit(D, St0.data)

    assert "converged !" in mcr.log[-15:]
    _assert_mcrals_shapes(mcr, D)
    _assert_reconstructs(mcr, D)


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
            np.ones((11, 2)),
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
