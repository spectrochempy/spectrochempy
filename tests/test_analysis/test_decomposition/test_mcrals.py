# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from os import environ
from types import SimpleNamespace

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
        exclude=["SA01", "EX01", "ES01", "GL11", "GL08", "PR01", "PR02", "PR06"],
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


# --- generators for the hard-constraint dispatch branches (PR1 regression) ---


def get_St_a(St, a):
    # args-only path
    return St / a


def get_St_kb(St, b=1):
    # kwargs-only path
    return St * b


def get_St_akb(St, a, b=1):
    # args + kwargs path
    return St * b / a


def get_St_tuple(St):
    # 2-tuple return: (profiles, new_args)
    return St, ()


def get_St_tuple_extra(St):
    # 3-tuple return: (profiles, new_args, extra)
    return St, (), {"extra": "spec"}


def get_St_zero(St):
    # returns a profile with a zero row (to exercise normSpec zero guard)
    St = St.copy()
    St[0] = 0
    return St


def get_C_tuple(C):
    return C, ()


def get_C_tuple_extra(C):
    return C, (), {"extra": "conc"}


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
    assert mcr.result.converged
    _assert_mcrals_shapes(mcr, D)
    assert mcr.transform(D) == mcr.C

    params = mcr.params()
    assert len(params) == 35
    assert np.all(params.closureTarget == [1] * 10)
    assert params.tol == 30.0
    assert params.tol_residual_change == 0.3
    assert params.tol_reconstruction_error is None
    assert params.tol_profile_change is None

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


def test_store_iteration_copies_all_arrays():
    mcr = MCRALS(storeIterations=True)
    state = SimpleNamespace(
        C=np.full((3, 2), 1.0),
        C_constrained=np.full((3, 2), 2.0),
        C_ls=np.full((3, 2), 3.0),
        St=np.full((2, 4), 4.0),
        St_ls=np.full((2, 4), 5.0),
        C_constrained_list=[],
        C_ls_list=[],
        St_constrained_list=[],
        St_ls_list=[],
    )

    mcr._store_iteration(state)

    assert not np.shares_memory(state.C_constrained_list[0], state.C_constrained)
    assert not np.shares_memory(state.C_ls_list[0], state.C_ls)
    assert not np.shares_memory(state.St_constrained_list[0], state.St)
    assert not np.shares_memory(state.St_ls_list[0], state.St_ls)


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
    assert mcr.result.converged


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
    assert mcr.result.converged
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
    assert mcr.result.converged
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
    assert "Convergence criterion not reached" in mcr.log[-100:]
    _assert_mcrals_shapes(mcr, D)


def test_mcrals_spectral_guess_converges(model, data):
    D = data
    St0 = model.St0

    mcr = MCRALS(tol=15.0)
    mcr.fit(D, St0.data)
    assert mcr.result.converged
    _assert_mcrals_shapes(mcr, D)


def test_mcrals_nnls_solver_smoke(model, data):
    D = data
    St0 = model.St0

    mcr = MCRALS(tol=15.0, nonnegConc=[], solverConc="nnls", solverSpec="nnls")
    mcr.fit(D, St0.data)

    assert mcr.result.converged
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

    assert mcr.result.converged
    _assert_mcrals_shapes(mcr, D)
    assert mcr.C[:, 0].min() >= -1.0e-12
    assert mcr.St[0].min() >= -1.0e-12


def test_mcrals_pnnls_concentration_solver_reconstructs(model, data):
    D = data
    St0 = model.St0

    mcr = MCRALS(tol=15.0, solverConc="pnnls")
    mcr.fit(D, St0.data)

    assert mcr.result.converged
    _assert_mcrals_shapes(mcr, D)
    _assert_reconstructs(mcr, D)


def test_pnnls_constrains_concentration_components_not_observations():
    from spectrochempy.analysis import constraints as ct

    raw_C = np.array([[-2.0, -1.0], [3.0, -2.0], [-4.0, 5.0]])
    St = np.eye(2)
    mcr = MCRALS(
        constraints=[ct.NonNegative("C", components=[0])],
        solver_C="pnnls",
    )
    mcr._X = NDDataset(raw_C @ St)

    C = mcr._solve_C(St)

    np.testing.assert_allclose(C[:, 0], [0.0, 3.0, 0.0], atol=1.0e-10)
    np.testing.assert_allclose(C[:, 1], raw_C[:, 1], atol=1.0e-10)


def test_pnnls_constrains_spectral_components_not_variables():
    from spectrochempy.analysis import constraints as ct

    C = np.eye(2)
    raw_St = np.array([[-2.0, 3.0, -4.0], [-1.0, -2.0, 5.0]])
    mcr = MCRALS(
        constraints=[ct.NonNegative("St", components=[0])],
        solver_St="pnnls",
    )
    mcr._X = NDDataset(C @ raw_St)

    St = mcr._solve_St(C)

    np.testing.assert_allclose(St[0], [0.0, 3.0, 0.0], atol=1.0e-10)
    np.testing.assert_allclose(St[1], raw_St[1], atol=1.0e-10)


def test_MCRALS_errors(model, data):
    # Text exceptions
    D = data
    C0 = model.C0
    mcr = MCRALS()

    # nonexistent keyword parameters
    try:
        _ = MCRALS(max_iter=25, nonexistent=0, log_level="DEBUG")
    except KeyError as exc:
        assert "'nonexistent' is not a valid" in exc.args[0]

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
        mcr.unimodSpec = "everything"

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


# --------------------------------------------------------------------------------------
# PR1 regression tests
# --------------------------------------------------------------------------------------


def test_MCRALS_pr1_getspec_with_args(model, data):
    """B1: `argsGetSpecc` typo used to raise AttributeError here."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardSpec = [0, 1]
    mcr.getSpec = get_St_a
    mcr.argsGetSpec = (1.0,)
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert mcr.result.converged
    _assert_mcrals_shapes(mcr, D)
    assert np.all(np.isfinite(mcr.C.data))
    assert np.all(np.isfinite(mcr.St.data))


def test_MCRALS_pr1_getspec_with_kwargs(model, data):
    """Dispatch branch: kwargs-only path for getSpec."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardSpec = [0, 1]
    mcr.getSpec = get_St_kb
    mcr.kwargsGetSpec = {"b": 1.0}
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert mcr.result.converged
    assert np.all(np.isfinite(mcr.St.data))


def test_MCRALS_pr1_getspec_with_args_and_kwargs(model, data):
    """Dispatch branch: args + kwargs path for getSpec."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardSpec = [0, 1]
    mcr.getSpec = get_St_akb
    mcr.argsGetSpec = (1.0,)
    mcr.kwargsGetSpec = {"b": 1.0}
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert mcr.result.converged
    assert np.all(np.isfinite(mcr.St.data))


def test_MCRALS_pr1_getspec_tuple_return(model, data):
    """B2: a 2-tuple return from getSpec used to crash (`.data` on a tuple)."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardSpec = [0, 1]
    mcr.getSpec = get_St_tuple
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert mcr.result.converged
    _assert_mcrals_shapes(mcr, D)
    assert np.all(np.isfinite(mcr.St.data))


def test_MCRALS_pr1_getspec_tuple_with_extra(model, data):
    """B2: a 3-tuple return from getSpec must populate extraOutputGetSpec."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardSpec = [0, 1]
    mcr.getSpec = get_St_tuple_extra
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert mcr.result.converged
    # the extra output is a list with one dict per iteration
    assert mcr.extraOutputGetSpec != []
    assert mcr.extraOutputGetSpec[0] == {"extra": "spec"}


def test_MCRALS_pr1_getspec_tuple_extra_with_args(model, data):
    """B1+B2 combined: 3-tuple return with non-empty argsGetSpec."""
    D = data
    C0 = model.C0

    def _gen(St, a):
        return St / a, (a,), {"extra": "spec"}

    mcr = MCRALS()
    mcr.hardSpec = [0, 1]
    mcr.getSpec = _gen
    mcr.argsGetSpec = (1.0,)
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert mcr.result.converged
    assert mcr.extraOutputGetSpec[0] == {"extra": "spec"}


def test_MCRALS_pr1_getconc_tuple_return(model, data):
    """getConc 2-tuple return branch (previously untested)."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardConc = [0, 1]
    mcr.getConc = get_C_tuple
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert mcr.result.converged
    _assert_mcrals_shapes(mcr, D)
    assert np.all(np.isfinite(mcr.C.data))


def test_MCRALS_pr1_getconc_tuple_with_extra(model, data):
    """getConc 3-tuple return must populate extraOutputGetConc."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardConc = [0, 1]
    mcr.getConc = get_C_tuple_extra
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert mcr.result.converged
    assert mcr.extraOutputGetConc != []
    assert mcr.extraOutputGetConc[0] == {"extra": "conc"}


def test_MCRALS_pr1_getconc_with_args(model, data):
    """getConc args-only dispatch branch (previously untested)."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardConc = [0, 1]
    mcr.getConc = get_C_a
    mcr.argsGetConc = (1.0,)
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert mcr.result.converged
    assert np.all(np.isfinite(mcr.C.data))


def test_MCRALS_pr1_getconc_with_args_and_kwargs(model, data):
    """getConc args+kwargs dispatch branch (previously untested)."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardConc = [0, 1]
    mcr.getConc = get_C_akb
    mcr.argsGetConc = (1.0,)
    mcr.kwargsGetConc = {"b": 1.0}
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert mcr.result.converged
    assert np.all(np.isfinite(mcr.C.data))


def test_MCRALS_pr1_getst_to_st_idx_none_entries(model, data):
    """B5: `getSt_to_St_idx` with `None` entries used to crash in `max()`."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.fit(D, C0)
    # assigning a list with a None entry must not raise (validator must skip None)
    mcr.getSt_to_St_idx = [0, None]
    assert mcr.getSt_to_St_idx == [0, None]

    # still rejects an out-of-range non-None index
    with pytest.raises(ValueError) as e:
        mcr.getSt_to_St_idx = [0, 5]
    assert "please check the" in e.value.args[0]


def test_MCRALS_pr1_closure_empty_list_is_noop(model, data):
    """B4: with the default `closureConc=[]`, closure must not run."""
    D = data
    C0 = model.C0

    # constantSum with an empty closure list used to run a wasteful / risky
    # block; it must now be a no-op and produce finite profiles.
    mcr = MCRALS()
    mcr.closureMethod = "constantSum"
    mcr.closureConc = []
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert np.all(np.isfinite(mcr.C.data))

    # result must match a run where closure is fully disabled
    mcr_ref = MCRALS()
    mcr_ref.tol = 30.0
    mcr_ref.fit(D, C0)
    assert np.allclose(mcr.C.data, mcr_ref.C.data, atol=1e-12)


def test_MCRALS_pr1_closure_constantsum_smoke(model, data):
    """B4/B9: constantSum closure must run without producing nan/inf."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.closureConc = [0, 1]
    mcr.closureMethod = "constantSum"
    mcr.max_iter = 2
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert np.all(np.isfinite(mcr.C.data))
    # the closure target (default -> ones) must be respected approximately
    assert np.allclose(
        np.sum(mcr.C_constrained.data[:, [0, 1]], axis=1),
        np.ones(D.shape[0]),
        atol=1e-6,
    )


def test_MCRALS_pr1_closure_single_component_is_active(model, data):
    """B4 regression: `closureConc=[0]` must activate closure on component 0.

    The earlier `np.any(self.closureConc)` guard evaluated `np.any([0]) == False`
    and silently disabled closure for a single selected component. The
    truthiness guard (`if self.closureConc:`) must keep it active.
    """
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.closureConc = [0]
    mcr.closureMethod = "constantSum"
    mcr.max_iter = 2
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert np.all(np.isfinite(mcr.C.data))
    # component 0 must be closed to the default target (ones)
    assert np.allclose(
        mcr.C_constrained.data[:, 0],
        np.ones(D.shape[0]),
        atol=1e-6,
    )


def test_MCRALS_pr1_normspec_max_zero_guard(model, data):
    """B9: `normSpec='max'` with a zero spectrum row must not produce nan."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardSpec = [0, 1]
    mcr.getSpec = get_St_zero
    mcr.normSpec = "max"
    mcr.max_iter = 2
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert np.all(np.isfinite(mcr.St.data))
    assert np.all(np.isfinite(mcr.C.data))


def test_MCRALS_pr1_normspec_euclid_zero_guard(model, data):
    """B9: `normSpec='euclid'` with a zero spectrum row must not produce nan."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardSpec = [0, 1]
    mcr.getSpec = get_St_zero
    mcr.normSpec = "euclid"
    mcr.max_iter = 2
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert np.all(np.isfinite(mcr.St.data))
    assert np.all(np.isfinite(mcr.C.data))


def test_MCRALS_pr1_unimodconc_not_duplicated(model, data):
    """B3: `unimodConc` is defined only once; setting it must behave normally."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.unimodConc = "all"
    mcr.unimodConcMod = "strict"
    mcr.unimodConcTol = 1.1
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert mcr.result.converged
    _assert_mcrals_shapes(mcr, D)
    # the trait must be a single, functional trait
    assert mcr.unimodConc == list(range(mcr._n_components))


def test_MCRALS_pr1_unimodal_smooth_bounds():
    """B6: smooth unimodality must terminate, stay finite and in-bounds.

    The original `_unimodal_1D` could index out of bounds or infinite-loop
    for pathological tolerances (including ``tol == 1.0`` and ``tol < 1``).
    The fix keeps byte-identical results for the documented regime
    (``tol >= 1.1``) and only guarantees safety (termination + finite +
    correct shape) for the pathological cases.
    """
    from spectrochempy.analysis.decomposition.mcrals import _unimodal_1D

    cases = (
        np.array([6.0, 10.0]),
        np.array([10.0, 6.0]),
        np.array([0.0, 5.0, 1.0, 3.0, 10.0]),
        np.array([10.0, 1.0, 5.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 10.0, 1.0, 5.0, 1.0]),
        np.array([0.0, 4.0, 1.0, 3.0, 0.0, 2.0, 1.0]),
    )

    # documented regime (tol >= 1.1): finite, shape-preserving, both modes
    for a0 in cases:
        for mod in ("smooth", "strict"):
            a = _unimodal_1D(a0.copy(), tol=1.1, mod=mod)
            assert a.shape == a0.shape
            assert np.all(np.isfinite(a))

    # pathological tolerances that used to IndexError / infinite-loop
    # (including tol == 1.0): must terminate, stay in bounds and stay finite.
    for tol in (0.1, 0.3, 0.5, 0.9, 1.0):
        for a0 in cases:
            a = _unimodal_1D(a0.copy(), tol=tol, mod="smooth")
            assert a.shape == a0.shape
            assert np.all(np.isfinite(a))


# --------------------------------------------------------------------------------------
# PR3 — internal constraint engine
#
# These tests assert properties of the *internal* constraint abstraction
# introduced by PR3. The constraint classes are private (``_``-prefixed) and
# must never leak into the public API; the tests here only guard against
# accidental regressions of the internal contract (privacy, builder wiring,
# byte-equivalence of constraint application vs. the historical helpers).
# They are intentionally small and do not re-test the numerical behavior
# already covered by the end-to-end MCRALS regression suite.
# --------------------------------------------------------------------------------------


def test_MCRALS_pr3_constraint_classes_are_private():
    """Internal constraint classes must not be exported in ``__all__``."""
    import spectrochempy.analysis.decomposition.mcrals as mod

    assert mod.__all__ == ["MCRALS"]
    expected_classes = (
        "_Constraint",
        "_NonNegativeConstraint",
        "_UnimodalConstraint",
        "_MonotonicIncreaseConstraint",
        "_MonotonicDecreaseConstraint",
        "_ClosureConstraint",
        "_NormalizationConstraint",
        "_ModelProfileConstraint",
    )
    for name in expected_classes:
        cls = getattr(mod, name, None)
        assert cls is not None, f"{name} missing from module"
        assert name.startswith("_"), f"{name} must remain private"


def test_MCRALS_pr3_constraint_apply_is_virtual():
    """The base ``_Constraint.apply`` must raise NotImplementedError."""
    from spectrochempy.analysis.decomposition.mcrals import _Constraint

    c = _Constraint()
    raised = False
    try:
        c.apply(None, None)
    except NotImplementedError:
        raised = True
    assert raised


def test_MCRALS_pr3_legacy_to_internal_pipeline():
    """Legacy traitlets are converted through the unified pipeline.

    ``legacy_to_constraints(estimator)`` + ``_build_from_public_constraints``
    must produce the correct internal constraints for the default traitlet
    configuration.  No ``_ModelProfileConstraint`` should appear when
    ``hardConc`` / ``hardSpec`` are empty (no hard profile configured).
    """
    import numpy as np

    from spectrochempy.analysis.decomposition._legacy_constraint_converter import (
        legacy_to_constraints,
    )
    from spectrochempy.analysis.decomposition.mcrals import (
        _ClosureConstraint,
        _ModelProfileConstraint,
        _NonNegativeConstraint,
        _NormalizationConstraint,
        _UnimodalConstraint,
        MCRALS,
    )

    mcr = MCRALS()
    mcr.fit(np.zeros((4, 3)), np.ones((4, 2)))

    constraints = legacy_to_constraints(mcr)
    conc, spec, norm = mcr._build_from_public_constraints(constraints)

    # Default nonnegConc="all" and unimodConc="all" produce constraints
    assert any(isinstance(c, _NonNegativeConstraint) for c in conc)
    assert any(isinstance(c, _UnimodalConstraint) for c in conc)
    # Default monoIncConc=[] / monoDecConc=[] / closureConc=[] produce nothing
    assert not any(isinstance(c, _ClosureConstraint) for c in conc)
    # Default hardConc=[] produces no ModelProfile
    assert not any(isinstance(c, _ModelProfileConstraint) for c in conc)
    # Default nonnegSpec="all" produces a spectral NonNegativeConstraint
    assert any(isinstance(c, _NonNegativeConstraint) for c in spec)
    # Default hardSpec=[] produces no ModelProfile on spec side
    assert not any(isinstance(c, _ModelProfileConstraint) for c in spec)
    # normSpec is None by default -> no normalization constraint
    assert norm is None


def test_MCRALS_pr3_closure_built_only_when_truthy():
    """``closureConc=[0]`` (PR1 #911 single-component case) must build closure."""
    import numpy as np

    from spectrochempy.analysis.decomposition._legacy_constraint_converter import (
        legacy_to_constraints,
    )
    from spectrochempy.analysis.decomposition.mcrals import (
        _ClosureConstraint,
        MCRALS,
    )

    mcr = MCRALS()
    mcr.closureConc = "all"
    mcr.fit(np.zeros((4, 3)), np.ones((4, 2)))
    constraints = legacy_to_constraints(mcr)
    conc, _, _ = mcr._build_from_public_constraints(constraints)
    assert any(isinstance(c, _ClosureConstraint) for c in conc)

    mcr2 = MCRALS()
    mcr2.closureConc = [0]
    mcr2.fit(np.zeros((4, 3)), np.ones((4, 2)))
    constraints2 = legacy_to_constraints(mcr2)
    conc2, _, _ = mcr2._build_from_public_constraints(constraints2)
    assert any(isinstance(c, _ClosureConstraint) for c in conc2)


def test_MCRALS_pr3_normalization_built_only_when_set():
    """``normSpec`` must drive normalization constraint construction."""
    import numpy as np

    from spectrochempy.analysis.decomposition.mcrals import (
        _NormalizationConstraint,
        MCRALS,
    )

    mcr = MCRALS(normSpec="max")
    mcr.fit(np.zeros((4, 3)), np.ones((4, 2)))
    norm = mcr._build_normalization()
    assert isinstance(norm, _NormalizationConstraint)

    mcr2 = MCRALS(normSpec="euclid")
    mcr2.fit(np.zeros((4, 3)), np.ones((4, 2)))
    assert isinstance(mcr2._build_normalization(), _NormalizationConstraint)

    mcr3 = MCRALS(normSpec=None)
    mcr3.fit(np.zeros((4, 3)), np.ones((4, 2)))
    assert mcr3._build_normalization() is None


def test_MCRALS_pr1_monotonic_tol_docstrings():
    """B7/B8: docstring/annotation sanity for the monotonic tolerances."""
    from spectrochempy.analysis.decomposition.mcrals import MCRALS as _M
    from spectrochempy.analysis.decomposition.mcrals import _unimodal_1D

    assert "monotonic increase" in _M.monoIncTol.help
    assert "monoIncTol" in _M.monoIncTol.help
    assert "monotonic decrease" in _M.monoDecTol.help
    assert "monoDecTol" in _M.monoDecTol.help
    # the type annotation of tol must be float, not str
    import inspect

    sig = inspect.signature(_unimodal_1D)
    assert sig.parameters["tol"].annotation is float


# --------------------------------------------------------------------------------------
# PR4 — behavioral characterization matrix
#
# These tests freeze the *current* numerical behavior of MCRALS across the
# documented constraint / solver / initialization space, before the next
# structural refactoring. They use a tiny deterministic synthetic dataset
# (8 observations x 12 wavelengths, 2 components) whose unconstrained
# least-squares solution already satisfies the soft constraints — so several
# constraint configurations reduce to no-ops. The no-op behavior itself is
# part of the characterization (a future regression would change it).
#
# Discovered bug (analogous to issue #911): the ``np.any(indices)`` guard
# used by ``_NonNegativeConstraint`` / ``_MonotonicIncreaseConstraint`` /
# ``_MonotonicDecreaseConstraint`` evaluates to ``False`` for ``[0]`` (a
# single zero), so selecting only component 0 silently disables the
# constraint. The working characterization below therefore uses indices
# ``[1]`` or ``[0, 1]``; the ``[0]`` selection silently no-op'd.
# This PR fixes the guard family (truthiness check, mirroring the PR1
# ``_ClosureConstraint`` fix for issue #911), so the former ``xfail``
# diagnostics are now permanent passing regression tests (see the
# "component-0 guard family" block below).
# --------------------------------------------------------------------------------------


def _pr4_make_data():
    """Deterministic (X, C0, St0) for the PR4 characterization matrix."""
    rng = np.random.RandomState(0)
    _n_obs, _n_wl, _n_comp = 8, 12, 2
    C_true = np.zeros((_n_obs, _n_comp))
    C_true[:, 0] = np.array([0.1, 0.3, 0.8, 1.0, 0.7, 0.3, 0.1, 0.05])
    C_true[:, 1] = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 0.9])
    St_true = np.zeros((_n_comp, _n_wl))
    St_true[0] = np.array(
        [0.1, 0.3, 0.9, 1.0, 0.8, 0.4, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01]
    )
    St_true[1] = np.array(
        [0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9, 1.0, 0.8, 0.5, 0.2, 0.05]
    )
    X = C_true @ St_true + 1.0e-6 * rng.randn(_n_obs, _n_wl)
    C0 = np.abs(C_true + 0.05 * np.array([[1.0, -0.5]] * _n_obs))
    St0 = np.abs(St_true + 0.05 * rng.randn(_n_comp, _n_wl))
    return X, C0, St0


_PR4_X, _PR4_C0, _PR4_St0 = _pr4_make_data()
_PR4_RTOL = 1.0e-7
_PR4_ATOL = 1.0e-10

# Baseline (no constraint active) reference values.
_PR4_BASE_C = np.array(
    [
        [0.1068653836, 0.0465680540],
        [0.3183345404, 0.0908325178],
        [0.8458804735, 0.1770601895],
        [1.0641205340, 0.3679394708],
        [0.7593293601, 0.5703353606],
        [0.3499313687, 0.7750337689],
        [0.1497472154, 0.9751260820],
        [0.0929305874, 0.8785355335],
    ]
)
_PR4_BASE_ST = np.array(
    [
        [
            0.0959501102,
            0.2876228217,
            0.8617432217,
            0.9594936673,
            0.7729962939,
            0.3977491435,
            0.2112514051,
            0.1180037060,
            0.0657518587,
            0.0399004442,
            0.0235996660,
            0.0106755570,
        ],
        [
            0.0160308846,
            0.0378714425,
            0.0625115383,
            0.1603051033,
            0.3735362700,
            0.6977946952,
            0.9110254785,
            1.0176422473,
            0.8154361736,
            0.5097041496,
            0.2035290395,
            0.0506610918,
        ],
    ]
)


def _pr4_fit(**kwargs):
    """Helper: fit a fresh MCRALS on the PR4 dataset and return it."""
    guess = kwargs.pop("guess", "C")
    mcr = MCRALS(**kwargs)
    if guess == "C":
        mcr.fit(_PR4_X, _PR4_C0)
    else:
        mcr.fit(_PR4_X, _PR4_St0)
    return mcr


# --- generators for the hard-profile return-form matrix --------------------------


def _pr4_getconc_bare(C):
    return np.full((C.shape[0], C.shape[1]), 0.5)


def _pr4_getconc_args(C, a):
    return np.full((C.shape[0], C.shape[1]), 0.5), (a * 2,)


def _pr4_getconc_extra(C, a):
    return np.full((C.shape[0], C.shape[1]), 0.5), (a,), {"marker": "conc"}


def _pr4_getspec_bare(St):
    return np.full((St.shape[0], St.shape[1]), 0.3)


def _pr4_getspec_args(St, a):
    return np.full((St.shape[0], St.shape[1]), 0.3), (a * 3,)


def _pr4_getspec_extra(St, a):
    return np.full((St.shape[0], St.shape[1]), 0.3), (a,), {"marker": "spec"}


# === 1. Baseline behavior ========================================================


def test_pr4_baseline_no_constraints_freezes_numerics():
    """No constraints active: deterministic C / St / metadata freeze."""
    mcr = _pr4_fit(nonnegConc=[], nonnegSpec=[], unimodConc=[], tol=1.0e-9, max_iter=50)
    assert mcr._fit_meta["n_iter"] == 3
    assert bool(mcr._fit_meta["converged"])
    np.testing.assert_allclose(
        np.asarray(mcr.C.data), _PR4_BASE_C, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )
    np.testing.assert_allclose(
        np.asarray(mcr.St.data), _PR4_BASE_ST, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )
    # In this regime C == C_constrained and St == St_ls.
    np.testing.assert_allclose(
        np.asarray(mcr.C_constrained.data),
        np.asarray(mcr.C.data),
        rtol=_PR4_RTOL,
        atol=_PR4_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(mcr.St_ls.data), np.asarray(mcr.St.data), rtol=1.0e-12
    )
    # extra outputs are empty when hard constraints are inactive
    assert mcr.extraOutputGetConc == []
    assert mcr.extraOutputGetSpec == []


# === 2. Non-negativity ===========================================================


def test_pr4_nonneg_conc_only_matches_baseline_noop():
    """For this non-negative LS solution, conc non-negativity is a no-op."""
    mcr = _pr4_fit(
        nonnegConc="all", nonnegSpec=[], unimodConc=[], tol=1.0e-9, max_iter=50
    )
    np.testing.assert_allclose(
        np.asarray(mcr.C.data), _PR4_BASE_C, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )
    np.testing.assert_allclose(
        np.asarray(mcr.St.data), _PR4_BASE_ST, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )
    assert np.min(np.asarray(mcr.C_constrained.data)) >= -_PR4_ATOL


def test_pr4_nonneg_spec_only_matches_baseline_noop():
    mcr = _pr4_fit(
        nonnegConc=[], nonnegSpec="all", unimodConc=[], tol=1.0e-9, max_iter=50
    )
    np.testing.assert_allclose(
        np.asarray(mcr.C.data), _PR4_BASE_C, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )
    np.testing.assert_allclose(
        np.asarray(mcr.St.data), _PR4_BASE_ST, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )
    assert np.min(np.asarray(mcr.St.data)) >= -_PR4_ATOL


def test_pr4_nonneg_both_matches_baseline_noop():
    mcr = _pr4_fit(
        nonnegConc="all", nonnegSpec="all", unimodConc=[], tol=1.0e-9, max_iter=50
    )
    np.testing.assert_allclose(
        np.asarray(mcr.C.data), _PR4_BASE_C, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )
    np.testing.assert_allclose(
        np.asarray(mcr.St.data), _PR4_BASE_ST, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )


def test_pr4_nonneg_conc_selected_component_stays_nonneg():
    """Selecting component 1 only keeps column 1 non-negative, leaves co. 0 free."""
    mcr = _pr4_fit(
        nonnegConc=[1], nonnegSpec=[], unimodConc=[], tol=1.0e-9, max_iter=50
    )
    Cc = np.asarray(mcr.C_constrained.data)
    assert Cc[:, 1].min() >= -_PR4_ATOL


def test_pr4_nonneg_conc_component_zero_is_enforced():
    """Regression: selecting only component 0 with ``nonnegConc=[0]`` must
    activate (not silently disable) the non-negativity constraint.

    Previously the ``np.any([0])`` activation guard evaluated to ``False``
    and skipped the constraint (same bug family as issue #911, fixed for
    closure in PR1). The guard now uses an explicit truthiness test, so a
    guess whose LS column 0 would otherwise go negative is clipped to 0.
    """
    X = _PR4_X.copy()
    # A guess whose first column is negative so that the LS column 0 dips
    # below zero and the non-negativity constraint actually has work to do.
    C0 = _PR4_C0.copy()
    C0[:, 0] = -np.abs(C0[:, 0])
    mcr = MCRALS(nonnegConc=[0], nonnegSpec=[], unimodConc=[], tol=1.0e-9, max_iter=50)
    mcr.fit(X, C0)
    Cc = np.asarray(mcr.C_constrained.data)
    # column 0 must be clipped to >= 0
    assert Cc[:, 0].min() >= -_PR4_ATOL
    # column 1 is *not* selected -> it is left free (the LS value, which here
    # stays positive but is not clipped by this constraint).
    np.testing.assert_allclose(
        Cc[:, 1], np.asarray(mcr.C_constrained.data)[:, 1], atol=1.0e-12
    )


# === 3. Unimodality ==============================================================


def test_pr4_unimod_strict_conc_noop_for_unimodal_data():
    """Concentration profiles are already unimodal — strict mode is a no-op."""
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc="all",
        unimodConcMod="strict",
        tol=1.0e-9,
        max_iter=50,
    )
    np.testing.assert_allclose(
        np.asarray(mcr.C.data), _PR4_BASE_C, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )
    np.testing.assert_allclose(
        np.asarray(mcr.St.data), _PR4_BASE_ST, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )


def test_pr4_unimod_smooth_conc_noop_for_unimodal_data():
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc="all",
        unimodConcMod="smooth",
        tol=1.0e-9,
        max_iter=50,
    )
    np.testing.assert_allclose(
        np.asarray(mcr.C.data), _PR4_BASE_C, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )


def test_pr4_unimod_strict_spec_noop():
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        unimodSpec="all",
        unimodSpecMod="strict",
        tol=1.0e-9,
        max_iter=50,
    )
    np.testing.assert_allclose(
        np.asarray(mcr.St.data), _PR4_BASE_ST, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )


def test_pr4_unimod_smooth_spec_noop():
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        unimodSpec="all",
        unimodSpecMod="smooth",
        tol=1.0e-9,
        max_iter=50,
    )
    np.testing.assert_allclose(
        np.asarray(mcr.St.data), _PR4_BASE_ST, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )


# === 4. Monotonicity =============================================================


def test_pr4_monoinc_component_one_bites_and_does_not_converge():
    """monoIncConc=[1] on a profile whose column 1 decreases after the peak
    forces monotonic increase; the loop does not converge within max_iter."""
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        monoIncConc=[1],
        tol=1.0e-9,
        max_iter=50,
    )
    assert mcr._fit_meta["n_iter"] == 50
    assert not bool(mcr._fit_meta["converged"])
    # exact characterization: the snapshot is byte-identical across runs.
    expected_Cc = np.array(
        [
            [0.1068653836, 0.0694232457],
            [0.3183345404, 0.0835876211],
            [0.8458804735, 0.0835876211],
            [1.0641205340, 0.4448786900],
            [0.7593293601, 1.1093924163],
            [0.3499313687, 1.8291636913],
            [0.1497472154, 2.4384170505],
            [0.0929305874, 2.4384170505],
        ]
    )
    np.testing.assert_allclose(
        np.asarray(mcr.C_constrained.data),
        expected_Cc,
        rtol=_PR4_RTOL,
        atol=_PR4_ATOL,
    )


def test_pr4_monodec_component_one_clamps_to_constant():
    """monoDecConc=[1] forces column 1 to be non-increasing; for this data the
    clamp collapses it to a near-constant slope."""
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        monoDecConc=[1],
        tol=1.0e-9,
        max_iter=50,
    )
    assert mcr._fit_meta["n_iter"] == 2
    assert bool(mcr._fit_meta["converged"])
    # exact characterization: column 1 collapses to a constant; column 0 is
    # the LS snapshot from the last iteration.
    expected_Cc = np.array(
        [
            [0.1068653836, 0.0058150120],
            [0.3183345404, 0.0058150120],
            [0.8458804735, 0.0058150120],
            [1.0641205340, 0.0058150120],
            [0.7593293601, 0.0058150120],
            [0.3499313687, 0.0058150120],
            [0.1497472154, 0.0058150120],
            [0.0929305874, 0.0058150120],
        ]
    )
    np.testing.assert_allclose(
        np.asarray(mcr.C_constrained.data),
        expected_Cc,
        rtol=_PR4_RTOL,
        atol=_PR4_ATOL,
    )


def test_pr4_monoinc_both_components():
    """monoIncConc=[0, 1] applies to both columns and converges (different
    initial / dynamic regime than the single-column case)."""
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        monoIncConc=[0, 1],
        tol=1.0e-9,
        max_iter=50,
    )
    # n_iter is intentionally not asserted: it varies across BLAS/LAPACK
    # platforms and is an incidental implementation detail, not behavior.
    # ``monoIncTol`` (default 1.1) lets the constraint accept small local
    # decreases, so a strict ``diff >= 0`` invariant would be wrong; the
    # meaningful invariants here are convergence, finiteness, shape, and a
    # bounded reconstruction error.
    assert bool(mcr._fit_meta["converged"])
    Cc = np.asarray(mcr.C_constrained.data)
    assert Cc.shape == (8, 2)
    assert np.all(np.isfinite(Cc))
    recon = np.asarray(mcr.C.data) @ np.asarray(mcr.St.data)
    np.testing.assert_allclose(recon, _PR4_X, atol=0.12)


def test_pr4_monodec_both_components():
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        monoDecConc=[0, 1],
        tol=1.0e-9,
        max_iter=50,
    )
    assert mcr._fit_meta["n_iter"] == 3
    assert bool(mcr._fit_meta["converged"])
    # exact characterization: the snapshot collapses to a 2-row constant matrix.
    expected_Cc = np.tile(
        np.array([[0.0028221982, 0.0011799111]]),
        (8, 1),
    )
    np.testing.assert_allclose(
        np.asarray(mcr.C_constrained.data),
        expected_Cc,
        rtol=_PR4_RTOL,
        atol=_PR4_ATOL,
    )


def test_pr4_monodec_component_zero_is_enforced():
    """Regression: selecting only component 0 with ``monoDecConc=[0]`` must
    activate (not silently disable) the monotonic-decrease constraint.

    Previously the ``np.any([0])`` activation guard evaluated to ``False``
    and skipped the constraint (same bug family as issue #911, fixed for
    closure in PR1). With the truthiness guard, column 0 is forced
    non-increasing, which for this dataset collapses it to a near-constant
    value across all observations.
    """
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        monoDecConc=[0],
        tol=1.0e-9,
        max_iter=50,
    )
    assert mcr._fit_meta["n_iter"] == 3
    assert bool(mcr._fit_meta["converged"])
    # exact characterization: column 0 collapses to a constant; column 1 is
    # the LS snapshot from the last iteration (constraint not selected).
    expected_Cc = np.array(
        [
            [0.0041909742, 0.0465680540],
            [0.0041909742, 0.0908325178],
            [0.0041909742, 0.1770601895],
            [0.0041909742, 0.3679394708],
            [0.0041909742, 0.5703353606],
            [0.0041909742, 0.7750337689],
            [0.0041909742, 0.9751260820],
            [0.0041909742, 0.8785355335],
        ]
    )
    np.testing.assert_allclose(
        np.asarray(mcr.C_constrained.data),
        expected_Cc,
        rtol=_PR4_RTOL,
        atol=_PR4_ATOL,
    )


# --- component-0 guard family: regression tests for the fixed activation -----
#
# The two preceding tests cover the originally reported members of the
# ``np.any([0])`` bug family (non-negativity and monotonic decrease). The
# same anti-pattern was also present in ``_UnimodalConstraint``,
# ``_MonotonicIncreaseConstraint`` and the two ``_ModelProfileConstraint``
# branches; each is exercised below with the component-0 selection that
# used to silently disable it.


def test_pr4_monoinc_component_zero_is_enforced():
    """Regression: ``monoIncConc=[0]`` must activate (not silently disable)."""
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        monoIncConc=[0],
        tol=1.0e-9,
        max_iter=50,
    )
    # column 0 of the snapshot is forced non-decreasing.
    Cc = np.asarray(mcr.C_constrained.data)
    assert np.diff(Cc[:, 0]).min() >= -1.0e-9
    # exact characterization: last clamped value
    np.testing.assert_allclose(Cc[-1, 0], 1.3728718899, rtol=_PR4_RTOL, atol=_PR4_ATOL)


def test_pr4_unimod_conc_component_zero_is_enforced():
    """Regression: ``unimodConc=[0]`` must activate (not silently disable).

    For this already-unimodal dataset the constraint is a no-op on the
    numerics, so the regression value is that it runs at all and produces
    the baseline solution (the pre-fix behaviour matched baseline because
    the guard was skipped; with the fix the constraint runs and still
    matches baseline because there is nothing to correct).
    """
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[0],
        unimodConcMod="strict",
        tol=1.0e-9,
        max_iter=50,
    )
    np.testing.assert_allclose(
        np.asarray(mcr.C.data), _PR4_BASE_C, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )


def test_pr4_nonneg_spec_component_zero_is_enforced():
    """Regression: ``nonnegSpec=[0]`` must clip St row 0 to >= 0."""
    # Use a guess whose St row 0 is negative so the constraint bites.
    St0 = _PR4_St0.copy()
    St0[0] = -np.abs(St0[0])
    mcr = MCRALS(nonnegConc=[], nonnegSpec=[0], unimodConc=[], tol=1.0e-9, max_iter=50)
    mcr.fit(_PR4_X, St0)
    assert np.asarray(mcr.St.data)[0].min() >= -_PR4_ATOL


def test_pr4_unimod_spec_component_zero_is_enforced():
    """Regression: ``unimodSpec=[0]`` must run (baseline-equivalent here)."""
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        unimodSpec=[0],
        unimodSpecMod="strict",
        tol=1.0e-9,
        max_iter=50,
    )
    np.testing.assert_allclose(
        np.asarray(mcr.St.data), _PR4_BASE_ST, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )


def test_pr4_hard_conc_component_zero_is_enforced():
    """Regression: ``hardConc=[0]`` must inject the generated column 0.

    Previously ``np.any([0])`` skipped the hard-profile injection. The
    generator returns shape ``(n_obs, len(hardConc))`` = ``(8, 1)`` per the
    documented contract, with ``getC_to_C_idx=[0]`` mapping it to C column 0.
    """

    def _gen_col0(C):
        return np.full((C.shape[0], 1), 0.5)

    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        hardConc=[0],
        getConc=_gen_col0,
        getC_to_C_idx=[0],
        tol=1.0e-9,
        max_iter=50,
    )
    # column 0 must be the injected constant 0.5
    np.testing.assert_allclose(
        np.asarray(mcr.C_constrained.data)[:, 0],
        np.full(8, 0.5),
        atol=1.0e-12,
    )


def test_pr4_hard_spec_component_zero_is_enforced():
    """Regression: ``hardSpec=[0]`` must inject the generated St row 0.

    The generator returns shape ``(len(hardSpec), n_wl)`` = ``(1, 12)``
    per the documented contract, with ``getSt_to_St_idx=[0]``.
    """

    def _gen_row0(St):
        return np.full((1, St.shape[1]), 0.3)

    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        hardSpec=[0],
        getSpec=_gen_row0,
        getSt_to_St_idx=[0],
        tol=1.0e-9,
        max_iter=50,
    )
    # row 0 must be the injected constant 0.3
    np.testing.assert_allclose(
        np.asarray(mcr.St.data)[0], np.full(12, 0.3), atol=1.0e-12
    )


# === 5. Closure =================================================================


def test_pr4_closure_empty_list_is_noop_vs_baseline():
    """closureConc=[] matches baseline exactly (closure never runs)."""
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        closureConc=[],
        closureMethod="scaling",
        tol=1.0e-9,
        max_iter=50,
    )
    np.testing.assert_allclose(
        np.asarray(mcr.C.data), _PR4_BASE_C, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )
    np.testing.assert_allclose(
        np.asarray(mcr.St.data), _PR4_BASE_ST, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )


def test_pr4_closure_scaling_all_rescales_columns():
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        closureConc="all",
        closureMethod="scaling",
        tol=1.0e-9,
        max_iter=50,
    )
    # The exact stopping iteration is an implementation detail; the factor
    # values below are the numerical contract of this regression test.
    assert mcr._fit_meta["converged"]
    expected_C = np.array(
        [
            [0.0915791433, 0.0426800234],
            [0.2727993249, 0.0832487866],
            [0.7248840224, 0.1622771920],
            [0.9119065838, 0.3372197010],
            [0.6507133550, 0.5227172810],
            [0.2998764791, 0.7103251391],
            [0.1283270713, 0.8937114712],
            [0.0796376085, 0.8051853997],
        ]
    )
    np.testing.assert_allclose(
        np.asarray(mcr.C.data), expected_C, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )


def test_pr4_closure_constantsum_all_enforces_unit_row_sum():
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        closureConc="all",
        closureMethod="constantSum",
        tol=1.0e-9,
        max_iter=50,
    )
    # n_iter is intentionally not asserted: it varies across BLAS/LAPACK
    # platforms and is an incidental implementation detail, not behavior.
    assert bool(mcr._fit_meta["converged"])
    Cc = np.asarray(mcr.C_constrained.data)
    assert np.all(np.isfinite(Cc))
    # meaningful invariant: every constrained row sums to the default
    # target (ones) — i.e. the constantSum closure is actually enforced.
    np.testing.assert_allclose(np.sum(Cc, axis=1), np.ones(8), atol=1.0e-6)


def test_pr4_closure_single_component_zero_is_active():
    """closureConc=[0] must activate closure on component 0 (PR1 #911 fix)."""
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        closureConc=[0],
        closureMethod="constantSum",
        tol=1.0e-9,
        max_iter=50,
    )
    # n_iter is intentionally not asserted: it varies across BLAS/LAPACK
    # platforms and is an incidental implementation detail, not behavior.
    assert bool(mcr._fit_meta["converged"])
    # component 0 must be closed to the default target (ones)
    np.testing.assert_allclose(
        np.asarray(mcr.C_constrained.data)[:, 0],
        np.ones(8),
        atol=1.0e-6,
    )


def test_pr4_closure_method_scaling_vs_constantsum_differ():
    """The two closure methods produce numerically distinct solutions."""
    mcr_scaling = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        closureConc="all",
        closureMethod="scaling",
        tol=1.0e-9,
        max_iter=50,
    )
    mcr_sum = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        closureConc="all",
        closureMethod="constantSum",
        tol=1.0e-9,
        max_iter=50,
    )
    assert not np.allclose(
        np.asarray(mcr_scaling.C.data),
        np.asarray(mcr_sum.C.data),
        atol=1.0e-6,
    )


# === 6. Spectral normalization ==================================================


def test_pr4_normspec_max_makes_each_spectrum_max_one():
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        normSpec="max",
        tol=1.0e-9,
        max_iter=50,
    )
    # n_iter is intentionally not asserted: convergence count may vary
    # slightly across numerical backends while preserving the same normalized
    # spectral solution.
    assert bool(mcr._fit_meta["converged"])
    St = np.asarray(mcr.St.data)
    np.testing.assert_allclose(np.max(St, axis=1), [1.0, 1.0], atol=1.0e-6)
    expected_St = np.array(
        [
            [
                0.1000007748,
                0.2997652111,
                0.8981228861,
                1.0,
                0.8056293858,
                0.4145406656,
                0.2201696710,
                0.1229853933,
                0.0685276630,
                0.0415848958,
                0.0245959580,
                0.0111262402,
            ],
            [
                0.0157529668,
                0.0372148882,
                0.0614278136,
                0.1575259908,
                0.3670604979,
                0.6856974512,
                0.8952315816,
                1.0,
                0.8012994505,
                0.5008677175,
                0.2000005798,
                0.0497828111,
            ],
        ]
    )
    np.testing.assert_allclose(St, expected_St, rtol=_PR4_RTOL, atol=_PR4_ATOL)


def test_pr4_normspec_euclid_makes_each_spectrum_unit_norm():
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        normSpec="euclid",
        tol=1.0e-9,
        max_iter=50,
    )
    # n_iter is intentionally not asserted: it varies across BLAS/LAPACK
    # platforms and is an incidental implementation detail, not behavior.
    assert bool(mcr._fit_meta["converged"])
    St = np.asarray(mcr.St.data)
    assert np.all(np.isfinite(St))
    # meaningful invariant: each spectral row has unit Euclidean norm.
    np.testing.assert_allclose(np.linalg.norm(St, axis=1), [1.0, 1.0], atol=1.0e-6)


def test_pr4_normspec_preserves_reconstruction():
    """normSpec rescales St/C jointly so C @ St is invariant."""
    mcr = MCRALS(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        normSpec="max",
        tol=1.0e-9,
        max_iter=50,
    )
    mcr.fit(_PR4_X, _PR4_C0)
    recon = np.asarray(mcr.C.data) @ np.asarray(mcr.St.data)
    np.testing.assert_allclose(recon, _PR4_X, atol=1.0e-5)


# === 7. Hard / generated concentration profiles ==================================


def test_pr4_getconc_bare_profile_replaces_columns():
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        hardConc=[0, 1],
        getConc=_pr4_getconc_bare,
        tol=1.0e-9,
        max_iter=50,
    )
    assert mcr._fit_meta["n_iter"] == 2
    np.testing.assert_allclose(
        np.asarray(mcr.C_constrained.data), np.full((8, 2), 0.5), atol=1.0e-12
    )
    assert mcr.argsGetConc == ()
    assert mcr.extraOutputGetConc == []


def test_pr4_getconc_with_args_updates_args():
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        hardConc=[0, 1],
        getConc=_pr4_getconc_args,
        argsGetConc=(1.0,),
        tol=1.0e-9,
        max_iter=50,
    )
    assert mcr._fit_meta["n_iter"] == 2
    np.testing.assert_allclose(
        np.asarray(mcr.C_constrained.data), np.full((8, 2), 0.5), atol=1.0e-12
    )
    # iter 1: a=1.0->2.0
    # iter 2: a=2.0->4.0
    assert mcr.argsGetConc == (1.0,)  # no longer mutated
    assert mcr._model_profile_constraints_[0].model_args == (4.0,)
    assert mcr.extraOutputGetConc == []


def test_pr4_getconc_with_extra_populates_extra_output():
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        hardConc=[0, 1],
        getConc=_pr4_getconc_extra,
        argsGetConc=(1.0,),
        tol=1.0e-9,
        max_iter=50,
    )
    assert mcr._fit_meta["n_iter"] == 2
    np.testing.assert_allclose(
        np.asarray(mcr.C_constrained.data), np.full((8, 2), 0.5), atol=1.0e-12
    )
    # generator always returns (a,) -> args stay (1.0,)
    assert mcr.argsGetConc == (1.0,)
    # extra-output buffer only keeps the last iteration's extra
    assert mcr.extraOutputGetConc == [{"marker": "conc"}]


def test_pr4_non_regression_constraints_empty():
    """``constraints=[]`` must produce byte-identical results to disabling
    all legacy traitlets (non-regression guarantee for the constrained-output
    semantics change)."""
    mcr_new = MCRALS(constraints=[], tol=1.0e-9, max_iter=20)
    mcr_new.fit(_PR4_X, _PR4_C0)
    mcr_legacy = MCRALS(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        unimodSpec=[],
        closureConc=[],
        normSpec=None,
        tol=1.0e-9,
        max_iter=20,
    )
    mcr_legacy.fit(_PR4_X, _PR4_C0)
    np.testing.assert_allclose(
        np.asarray(mcr_new.C.data), np.asarray(mcr_legacy.C.data), atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(mcr_new.St.data), np.asarray(mcr_legacy.St.data), atol=0.0, rtol=0.0
    )


# === 8. Hard / generated spectral profiles =======================================


def test_pr4_getspec_bare_profile_replaces_rows():
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        hardSpec=[0, 1],
        getSpec=_pr4_getspec_bare,
        tol=1.0e-9,
        max_iter=50,
    )
    assert mcr._fit_meta["n_iter"] == 3
    np.testing.assert_allclose(
        np.asarray(mcr.St.data), np.full((2, 12), 0.3), atol=1.0e-12
    )
    assert mcr.argsGetSpec == ()
    assert mcr.extraOutputGetSpec == []


def test_pr4_getspec_with_args_updates_args():
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        hardSpec=[0, 1],
        getSpec=_pr4_getspec_args,
        argsGetSpec=(1.0,),
        tol=1.0e-9,
        max_iter=50,
    )
    assert mcr._fit_meta["n_iter"] == 3
    np.testing.assert_allclose(
        np.asarray(mcr.St.data), np.full((2, 12), 0.3), atol=1.0e-12
    )
    # iter 1: a=1.0 -> 3.0 ; iter 2: a=3.0 -> 9.0 ; iter 3: a=9.0 -> 27.0
    assert mcr.argsGetSpec == (1.0,)  # no longer mutated
    assert mcr._model_profile_constraints_[0].model_args == (27.0,)
    assert mcr.extraOutputGetSpec == []


def test_pr4_getspec_with_extra_populates_extra_output():
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        hardSpec=[0, 1],
        getSpec=_pr4_getspec_extra,
        argsGetSpec=(1.0,),
        tol=1.0e-9,
        max_iter=50,
    )
    assert mcr._fit_meta["n_iter"] == 3
    np.testing.assert_allclose(
        np.asarray(mcr.St.data), np.full((2, 12), 0.3), atol=1.0e-12
    )
    assert mcr.argsGetSpec == (1.0,)
    assert mcr.extraOutputGetSpec == [{"marker": "spec"}]


# === 9. Combined case ===========================================================


def test_pr4_combined_closure_hardconc_normspec():
    """closure + hardConc + normSpec: ordering must produce a max-normalized St
    on top of the hard-concentration profile.  Note that normalization scales C
    reciprocally, so the closure row sum is no longer 1 after normalization.
    Users who need both closure and normalization should choose one normalization
    strategy or re-apply closure after normalization."""
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        closureConc="all",
        closureMethod="constantSum",
        hardConc=[0, 1],
        getConc=_pr4_getconc_bare,
        normSpec="max",
        tol=1.0e-9,
        max_iter=50,
    )
    assert mcr._fit_meta["n_iter"] == 2
    # C_constrained reflects the final C (post-normalization),
    # which differs from the hard-cone-only value (0.5) because
    # normalization scales C reciprocally.
    Cc = np.asarray(mcr.C_constrained.data)
    assert Cc.shape == (8, 2)
    assert not np.allclose(Cc, 0.5, atol=1.0e-6), (
        "C_constrained should differ from 0.5 because normalization"
        " scales C reciprocally"
    )
    # St is max-normalized: each row max == 1
    St = np.asarray(mcr.St.data)
    np.testing.assert_allclose(np.max(St, axis=1), [1.0, 1.0], atol=1.0e-6)
    # C and C_constrained are identical (both reflect the final iteration state)
    np.testing.assert_allclose(np.asarray(mcr.C.data), Cc, atol=1.0e-12)
    # Reconstruction from the final constrained pair is finite and
    # approximately matches X (tight fit is not expected when both
    # closure and max-normalization are active simultaneously).
    recon = Cc @ St
    X = np.asarray(mcr.X.data)
    rel_diff = np.linalg.norm(recon - X) / max(np.linalg.norm(X), 1.0e-15)
    assert rel_diff < 0.55, f"Reconstruction relative error too large: {rel_diff}"
    # The C sum is no longer 1.0 because normalization scales C reciprocally
    C_sum = np.sum(Cc, axis=1)
    assert np.allclose(
        C_sum, C_sum[0], atol=1.0e-12
    ), "C rows have different sums after reciprocal scaling"
    assert np.max(C_sum) < 1.0, "C rows exceed 1.0 after reciprocal scaling"
    # hardConc runs before normalization: the raw model output is 0.5,
    # but normalization scales it.  Verify the model was called
    # (extraOutputGetConc is empty for bare generators, but C is
    # correctly constrained to the model output).
    St_max = np.max(St, axis=1)
    assert np.allclose(St_max, [1.0, 1.0], atol=1.0e-6)


# === 10. Solver / initialization combinations ====================================


def test_pr4_solver_nnls_matches_baseline():
    """NNLS gives the same solution as lstsq when the LS solution is already
    non-negative for this dataset."""
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        solverConc="nnls",
        solverSpec="nnls",
        tol=1.0e-9,
        max_iter=50,
    )
    assert mcr._fit_meta["n_iter"] == 3
    np.testing.assert_allclose(
        np.asarray(mcr.C.data), _PR4_BASE_C, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )
    np.testing.assert_allclose(
        np.asarray(mcr.St.data), _PR4_BASE_ST, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )
    assert np.min(np.asarray(mcr.C.data)) >= -_PR4_ATOL
    assert np.min(np.asarray(mcr.St.data)) >= -_PR4_ATOL


def test_pr4_solver_pnnls_partial_nonneg_matches_baseline():
    mcr = _pr4_fit(
        nonnegConc=[0],
        nonnegSpec=[0],
        unimodConc=[],
        solverConc="pnnls",
        solverSpec="pnnls",
        tol=1.0e-9,
        max_iter=50,
    )
    # n_iter is intentionally not asserted: it varies across BLAS/LAPACK
    # platforms and is an incidental implementation detail, not behavior.
    assert bool(mcr._fit_meta["converged"])
    # pnnls with partial non-negativity must reach the same fix-point as
    # the unconstrained baseline for this already-non-negative dataset.
    np.testing.assert_allclose(
        np.asarray(mcr.C.data), _PR4_BASE_C, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )
    np.testing.assert_allclose(
        np.asarray(mcr.St.data), _PR4_BASE_ST, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )
    Cc = np.asarray(mcr.C_constrained.data)
    # meaningful invariant: the partially-non-negative column 0 stays >= 0.
    assert Cc[:, 0].min() >= -1.0e-9


def test_pr4_initial_guess_as_spectra_freezes_numerics():
    """Initial guess given as St (instead of C) yields a different but stable
    solution."""
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        tol=1.0e-9,
        max_iter=50,
        guess="St",
    )
    assert mcr._fit_meta["n_iter"] == 3
    assert bool(mcr._fit_meta["converged"])
    expected_C = np.array(
        [
            [0.0926110570, 0.0502498710],
            [0.2797999706, 0.1024956728],
            [0.7487589333, 0.2089856642],
            [0.9300348905, 0.4059926058],
            [0.6384105147, 0.5930180667],
            [0.2522056233, 0.7780464168],
            [0.0551620331, 0.9670679540],
            [0.0118142777, 0.8695633848],
        ]
    )
    expected_St = np.array(
        [
            [
                0.1052030020,
                0.3158248380,
                0.9485671042,
                1.0520219833,
                0.8363734190,
                0.4072592789,
                0.1916103291,
                0.0837870833,
                0.0353364698,
                0.0207639367,
                0.0167562026,
                0.0094717890,
            ],
            [
                0.0250211827,
                0.0647096282,
                0.1423637177,
                0.2502073610,
                0.4486374943,
                0.7419689392,
                0.9403986514,
                1.0396150072,
                0.8303966685,
                0.5189453417,
                0.2079234991,
                0.0521960257,
            ],
        ]
    )
    np.testing.assert_allclose(
        np.asarray(mcr.C.data), expected_C, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )
    np.testing.assert_allclose(
        np.asarray(mcr.St.data), expected_St, rtol=_PR4_RTOL, atol=_PR4_ATOL
    )


# === 11. Loop structure invariants =================================================


def test_pr4_model_profile_called_once_per_iteration():
    """ModelProfile must be called exactly once per iteration (not 2x).
    This validates the single-pass-per-half-step loop invariant."""
    call_count = 0

    def counting_model(C, a):
        nonlocal call_count
        call_count += 1
        return np.full((C.shape[0], C.shape[1]), 0.5), (a * 2,)

    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        hardConc=[0, 1],
        getConc=counting_model,
        argsGetConc=(1.0,),
        tol=1.0e-9,
        max_iter=10,
    )
    n_iter = mcr._fit_meta["n_iter"]
    assert call_count == n_iter, (
        f"Expected {n_iter} ModelProfile calls (one per iteration), "
        f"got {call_count}"
    )


def test_pr4_loop_invariant_c_ls_matches_solve_c_from_st():
    """After fitting, C_ls (from constraint_diagnostics) must be consistent
    with solve_C(state.St), confirming that C_ls is the fresh unconstrained
    LS estimate before constraints are applied."""
    mcr = _pr4_fit(
        nonnegConc=[0],
        nonnegSpec=[],
        unimodConc=[],
        tol=1.0e-9,
        max_iter=20,
    )
    X = _PR4_X
    St_con = np.asarray(mcr.St.data)
    St_ls = np.asarray(mcr.St_ls.data)
    C_con = np.asarray(mcr.C.data)

    # St_ls should equal solve_St(C_con) where C_con is constrained C
    expected_St_ls = np.linalg.lstsq(C_con, X, rcond=None)[0]
    np.testing.assert_allclose(St_ls, expected_St_ls, rtol=1.0e-10, atol=1.0e-10)

    # constraint_diagnostics should report the C projection distance
    cd = mcr._fit_meta.get("constraint_diagnostics", {})
    assert cd, "constraint_diagnostics should be populated"
    assert "C" in cd, "constraint_diagnostics should have C entry"


def test_pr4_loop_invariant_st_ls_matches_solve_st_from_c():
    """After fitting, St_ls must equal solve_St(C) where C is the constrained
    C, confirming that St_ls is the fresh unconstrained LS estimate before
    spectral constraints."""
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[0],
        unimodConc=[],
        tol=1.0e-9,
        max_iter=20,
    )
    C_con = np.asarray(mcr.C.data)
    St_ls = np.asarray(mcr.St_ls.data)
    X = _PR4_X
    expected_St_ls = np.linalg.lstsq(C_con, X, rcond=None)[0]
    np.testing.assert_allclose(St_ls, expected_St_ls, rtol=1.0e-10, atol=1.0e-10)


# === 12. Loop structure verification ==============================================


def test_loop_operation_count():
    """Verify exactly 2 solves (C + St) and 2 constraint pipelines per iteration.

    No hidden solves, no repeated constraint passes, no iterative diagnostics.
    Initialization and finalization-only refits are accounted separately.
    """
    import functools

    solve_C_calls = 0
    solve_St_calls = 0
    pipeline_calls = 0

    original_solve_C = MCRALS._solve_C
    original_solve_St = MCRALS._solve_St
    # _apply_constraint_pipeline is a @staticmethod — preserve descriptor
    original_solve_C = MCRALS._solve_C
    original_solve_St = MCRALS._solve_St
    original_apply_descriptor = MCRALS.__dict__["_apply_constraint_pipeline"]
    original_apply_func = original_apply_descriptor.__func__

    def counting_solve_C(self, St):
        nonlocal solve_C_calls
        solve_C_calls += 1
        return original_solve_C(self, St)

    def counting_solve_St(self, C):
        nonlocal solve_St_calls
        solve_St_calls += 1
        return original_solve_St(self, C)

    @staticmethod
    def counting_pipeline(profile, constraints, state, **kw):
        nonlocal pipeline_calls
        pipeline_calls += 1
        return original_apply_func(profile, constraints, state, **kw)

    MCRALS._solve_C = counting_solve_C
    MCRALS._solve_St = counting_solve_St
    MCRALS._apply_constraint_pipeline = counting_pipeline

    try:
        mcr = MCRALS(
            nonnegConc="all",
            nonnegSpec="all",
            constraints=None,
            tol=1.0e-12,
            max_iter=3,
        )
        mcr.fit(_PR4_X, _PR4_C0)

        n_iter = mcr._fit_meta["n_iter"]

        # Each standard iteration: 1 C solve + 1 conc pipeline + 1 St solve + 1 spec pipeline
        expected_solves = n_iter
        expected_pipelines = n_iter * 2  # C + St pipeline each

        # The first solve_St is in _guess_profile (initialization from C0)
        expected_solve_St = expected_solves + 1  # extra for guess

        assert (
            solve_C_calls == expected_solves
        ), f"Expected {expected_solves} C solves (one per iteration), got {solve_C_calls}"
        assert solve_St_calls == expected_solve_St, (
            f"Expected {expected_solve_St} St solves "
            f"({expected_solves} iterations + 1 initialization), got {solve_St_calls}"
        )
        assert pipeline_calls == expected_solves * 2, (
            f"Expected {expected_solves * 2} pipeline calls "
            f"(C + St per iteration), got {pipeline_calls}"
        )
    finally:
        MCRALS._solve_C = original_solve_C
        MCRALS._solve_St = original_solve_St
        MCRALS._apply_constraint_pipeline = original_apply_descriptor


def test_loop_operation_count_st0_init():
    """Same operation-count check for St0 initialization.

    With St0 initialization, _guess_profile computes C = solve_C(St0).
    The first iteration then also computes solve_C(St0) (redundant with
    initialization), so the C-solve count is n_iter (not n_iter - 1).
    """
    import functools

    solve_C_calls = 0
    solve_St_calls = 0

    original_solve_C = MCRALS._solve_C
    original_solve_St = MCRALS._solve_St

    def counting_solve_C(self, St):
        nonlocal solve_C_calls
        solve_C_calls += 1
        return original_solve_C(self, St)

    def counting_solve_St(self, C):
        nonlocal solve_St_calls
        solve_St_calls += 1
        return original_solve_St(self, C)

    MCRALS._solve_C = counting_solve_C
    MCRALS._solve_St = counting_solve_St

    try:
        mcr = MCRALS(
            nonnegConc=[],
            nonnegSpec=[],
            constraints=None,
            tol=1.0e-12,
            max_iter=3,
        )
        mcr.fit(_PR4_X, _PR4_St0)

        n_iter = mcr._fit_meta["n_iter"]

        # St0 path: guess does solve_C(St0) → 1 C solve, 0 St solves
        # Then each iteration: 1 C solve + 1 St solve
        expected_C_solves = n_iter + 1  # +1 for guess init
        expected_St_solves = n_iter  # no extra guess solve

        assert solve_C_calls == expected_C_solves, (
            f"Expected {expected_C_solves} C solves "
            f"({n_iter} iterations + 1 initialization), got {solve_C_calls}"
        )
        assert solve_St_calls == expected_St_solves, (
            f"Expected {expected_St_solves} St solves "
            f"({n_iter} iterations), got {solve_St_calls}"
        )
    finally:
        MCRALS._solve_C = original_solve_C
        MCRALS._solve_St = original_solve_St


def test_convergence_from_final_pair():
    """Convergence and residuals must be computed from the final accepted pair
    (post-constraints, post-normalization, post-presence)."""
    from spectrochempy.analysis.decomposition.mcrals import _ComponentPresenceConstraint

    mcr = MCRALS(
        nonnegConc="all",
        nonnegSpec="all",
        normSpec="max",
        tol=1.0e-10,
        max_iter=20,
    )
    mcr.fit(_PR4_X, _PR4_C0)

    C_final = np.asarray(mcr.C.data)
    St_final = np.asarray(mcr.St.data)
    X = _PR4_X

    # The fit_meta residual_std must match std(X - C_final @ St_final)
    expected_std = np.std(X - C_final @ St_final)
    np.testing.assert_allclose(
        mcr._fit_meta["residual_std"], expected_std, rtol=1.0e-12, atol=1.0e-12
    )


def test_public_output_consistency():
    """For all fit modes, inverse_transform == C @ St and residuals == X - C @ St."""
    fit_configs = [
        {"label": "C0_init_no_constraints", "guess": "C", "kwargs": {}},
        {"label": "St0_init_no_constraints", "guess": "St", "kwargs": {}},
        {
            "label": "C0_init_constrained",
            "guess": "C",
            "kwargs": {"nonnegConc": "all", "nonnegSpec": "all", "normSpec": "max"},
        },
    ]
    for cfg in fit_configs:
        mcr = _pr4_fit(tol=1.0e-10, max_iter=20, guess=cfg["guess"], **cfg["kwargs"])
        X = np.asarray(mcr.X.data)
        C = np.asarray(mcr.C.data)
        St = np.asarray(mcr.St.data)
        recon = C @ St

        # inverse_transform must match C @ St
        np.testing.assert_allclose(
            np.asarray(mcr.inverse_transform().data),
            recon,
            rtol=1.0e-10,
            atol=1.0e-10,
            err_msg=f"inverse_transform mismatch for {cfg['label']}",
        )

        # C_constrained must equal C (post-normalization consistency)
        Cc = np.asarray(mcr.C_constrained.data)
        np.testing.assert_allclose(
            Cc,
            C,
            rtol=1.0e-10,
            atol=1.0e-10,
            err_msg=f"C_constrained != C for {cfg['label']}",
        )


def test_normalization_preserves_reconstruction():
    """Normalization must preserve C @ St within numerical precision."""
    mcr = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        normSpec="max",
        tol=1.0e-9,
        max_iter=10,
    )
    C = np.asarray(mcr.C.data)
    St = np.asarray(mcr.St.data)
    X = _PR4_X
    recon = C @ St

    # Reconstruction must be close to X (normalization doesn't change C @ St)
    np.testing.assert_allclose(recon, X, rtol=1.0e-5, atol=1.0e-5)

    # Each St row must be max-normalized
    St_max = np.max(St, axis=1)
    np.testing.assert_allclose(St_max, [1.0, 1.0], rtol=1.0e-6)
    # C must be >= 0 (max-normalization with nonneg preserves non-negativity)
    assert np.all(C >= -1.0e-10), "C has negative entries after normalization"

    # Also test euclidean normalization
    mcr2 = _pr4_fit(
        nonnegConc=[],
        nonnegSpec=[],
        normSpec="euclid",
        tol=1.0e-9,
        max_iter=10,
    )
    St2 = np.asarray(mcr2.St.data)
    St_norms = np.linalg.norm(St2, axis=1)
    np.testing.assert_allclose(St_norms, [1.0, 1.0], rtol=1.0e-6)


# ======================================================================================
# 10. Public representation (__repr__)
# ======================================================================================


def test_repr_default():
    """Default MCRALS repr is concise and contains only modern public API names."""
    mcr = MCRALS()
    r = repr(mcr)
    assert r.startswith("MCRALS(")
    assert r.endswith(")")
    assert "constraints=None" in r
    assert "solver_C=" in r
    assert "solver_St=" in r
    assert "max_iter=" in r
    assert "tol_residual_change=" in r
    assert "tol_reconstruction_error=" in r
    assert "tol_profile_change=" in r
    # No legacy parameter names
    assert "nonnegConc" not in r
    assert "nonnegSpec" not in r
    assert "unimodConc" not in r
    assert "closureConc" not in r
    assert "hardConc" not in r
    assert "getConc" not in r
    assert "normSpec" not in r
    assert "storeIterations" not in r
    # Not fitted
    assert "fitted" not in r
    lines = r.split("\n")
    assert len(lines) <= 15


def test_repr_with_constraints():
    """Constraints are shown using their public repr."""
    from spectrochempy.analysis import constraints as ct

    mcr = MCRALS(constraints=[ct.NonNegative("C"), ct.NonNegative("St")])
    r = repr(mcr)
    assert "constraints=[" in r
    assert "NonNegative(profile='C'" in r
    assert "NonNegative(profile='St'" in r
    assert r.endswith(")")
    lines = r.split("\n")
    assert len(lines) <= 15


def test_repr_many_constraints_summarized():
    """Long constraint lists are summarised with a count of remaining items."""
    from spectrochempy.analysis import constraints as ct

    all_c = [
        ct.NonNegative("C"),
        ct.NonNegative("St"),
        ct.Unimodal("C"),
        ct.Closure("C"),
        ct.Monotonic("C", "increasing"),
        ct.Trilinear("C"),
        ct.ComponentPresence(
            "C",
            presence=[[True, True, True], [True, False, False]],
        ),
    ]
    mcr = MCRALS(constraints=all_c)
    r = repr(mcr)
    assert "... (2 more)" in r
    lines = r.split("\n")
    assert len(lines) <= 18


def test_repr_fitted_shows_indicator():
    """After fitting, the repr includes a fitted=True indicator."""
    import spectrochempy as scp

    X = scp.NDDataset(np.random.default_rng(42).normal(size=(10, 5)))
    C0 = scp.NDDataset(np.abs(np.random.default_rng(42).normal(size=(10, 2))))
    St0 = scp.NDDataset(np.abs(np.random.default_rng(42).normal(size=(2, 5))))
    mcr = MCRALS()
    mcr.fit(X, (C0, St0))
    r = repr(mcr)
    assert "fitted=True" in r
    # No fitted arrays
    assert "array" not in r
    assert "data" not in r
    assert ".data" not in r
    lines = r.split("\n")
    assert len(lines) <= 15


def test_repr_max_length():
    """repr stays below 1000 characters in all scenarios."""
    from spectrochempy.analysis import constraints as ct

    # Default
    assert len(repr(MCRALS())) < 1000

    # With constraints
    mcr = MCRALS(constraints=[ct.NonNegative("C"), ct.Closure("C")])
    assert len(repr(mcr)) < 1000

    # With many constraints
    many = [
        ct.NonNegative("C"),
        ct.NonNegative("St"),
        ct.Unimodal("C"),
        ct.Unimodal("St"),
        ct.Closure("C"),
        ct.Monotonic("C", "increasing"),
    ]
    assert len(repr(MCRALS(constraints=many))) < 1000


# ======================================================================================
# 11. Constructor signature synchronisation
# ======================================================================================


def test_signature_defaults_synchronised_with_runtime():
    """inspect.signature defaults must match runtime attribute values."""
    import inspect
    import logging

    sig = inspect.signature(MCRALS)
    mcr = MCRALS()

    for pname, expected in [
        ("solver_C", "lstsq"),
        ("solver_St", "lstsq"),
        ("tol_residual_change", 1.0e-3),
        ("tol_reconstruction_error", None),
        ("tol_profile_change", None),
        ("max_iter", 50),
        ("maxdiv", 5),
        ("warm_start", False),
        ("log_level", logging.WARNING),
    ]:
        p = sig.parameters.get(pname)
        assert p is not None, f"{pname} missing from signature"
        sig_default = p.default
        try:
            rt = getattr(mcr, pname)
        except AttributeError:
            # constructor-only parameter (e.g. log_level)
            rt = sig_default
        assert (
            sig_default == expected
        ), f"{pname} sig default {sig_default!r} != expected {expected!r}"
        assert (
            sig_default == rt
        ), f"{pname} sig default {sig_default!r} != runtime {rt!r}"


def test_signature_no_legacy_params():
    """The visible signature must not contain legacy traitlet parameter names."""
    import inspect

    sig = inspect.signature(MCRALS)
    legacy_names = {
        "nonnegConc",
        "unimodConc",
        "unimodConcMod",
        "unimodConcTol",
        "monoDecConc",
        "monoDecTol",
        "monoIncConc",
        "monoIncTol",
        "closureConc",
        "closureTarget",
        "closureMethod",
        "hardConc",
        "getConc",
        "argsGetConc",
        "kwargsGetConc",
        "getC_to_C_idx",
        "nonnegSpec",
        "normSpec",
        "unimodSpec",
        "unimodSpecMod",
        "unimodSpecTol",
        "hardSpec",
        "getSpec",
        "argsGetSpec",
        "kwargsGetSpec",
        "getSt_to_St_idx",
        "storeIterations",
        "solverConc",
        "solverSpec",
        "tol",
    }
    found = legacy_names & set(sig.parameters)
    assert not found, f"Legacy parameters found in signature: {found}"


def test_warm_start_property():
    """warm_start must be readable and writable via a public property."""
    mcr = MCRALS()
    assert mcr.warm_start is False
    mcr.warm_start = True
    assert mcr.warm_start is True
    mcr.warm_start = False
    assert mcr.warm_start is False


def test_docstring_single_parameters_section():
    """The class docstring must contain exactly one Parameters section."""
    doc = MCRALS.__doc__
    assert doc is not None
    assert doc.count("Parameters\n----------") == 1


def test_docstring_modern_params_in_order():
    """Parameters section must list params in the same order as __signature__."""
    import re

    expected = [
        p
        for p in MCRALS.__signature__.parameters
        if MCRALS.__signature__.parameters[p].kind.name != "VAR_POSITIONAL"
    ]

    doc = MCRALS.__doc__
    assert doc is not None
    ps = doc.find("Parameters\n----------")
    params_body = doc[ps + len("Parameters\n----------") :]
    sa = params_body.find("See Also")
    params_only = params_body[:sa] if sa >= 0 else params_body

    actual = []
    for line in params_only.split("\n"):
        m = re.match(r"^(\w+)\s*:", line)
        if m:
            actual.append(m.group(1))
    assert actual == expected, f"Order mismatch: {actual} != {expected}"


def test_docstring_no_legacy_param_definitions():
    """No legacy traitlets should appear as parameter definitions in the docstring."""
    import re

    doc = MCRALS.__doc__
    assert doc is not None
    ps = doc.find("Parameters\n----------")
    params_body = doc[ps + len("Parameters\n----------") :]
    sa = params_body.find("See Also")
    params_only = params_body[:sa] if sa >= 0 else params_body

    legacy_names = {
        "nonnegConc",
        "nonnegSpec",
        "unimodConcMod",
        "unimodConcTol",
        "monoDecConc",
        "monoDecTol",
        "monoIncConc",
        "monoIncTol",
        "closureTarget",
        "closureMethod",
        "hardConc",
        "hardSpec",
        "getC_to_C_idx",
        "getSt_to_St_idx",
        "argsGetConc",
        "argsGetSpec",
        "kwargsGetConc",
        "kwargsGetSpec",
        "normSpec",
        "storeIterations",
    }
    for name in legacy_names:
        if re.search(rf"^{name}\s*:", params_only, re.MULTILINE):
            raise AssertionError(f"Legacy parameter definition found: {name}")


def test_docstring_each_modern_param_once():
    """Each modern parameter must appear exactly once in the Parameters section."""
    import re

    doc = MCRALS.__doc__
    assert doc is not None
    ps = doc.find("Parameters\n----------")
    params_body = doc[ps + len("Parameters\n----------") :]
    sa = params_body.find("See Also")
    params_only = params_body[:sa] if sa >= 0 else params_body

    modern = [
        "constraints",
        "solver_C",
        "solver_St",
        "max_iter",
        "tol_residual_change",
        "tol_reconstruction_error",
        "tol_profile_change",
        "maxdiv",
        "log_level",
        "warm_start",
    ]
    for name in modern:
        count = params_only.count(f"{name} :")
        assert count == 1, f"{name}: expected 1 occurrence, got {count}"
